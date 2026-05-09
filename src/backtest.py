"""
Rolling backtest: compare PINN pricing against market prices over a
historical window using akshare (A-share 50ETF) data.

Strategy: pricing-error backtest (not a trading strategy).
  For each trading day in the window:
    1. Fetch 50ETF spot price (from historical ETF data)
    2. Fetch the option contract's historical closing price (market mid proxy)
    3. Calibrate BSM sigma to that day's market price
    4. Price with PINN using calibrated sigma
    5. Record: market price, BSM analytical price, PINN price, calibrated IV

Output:
    results/backtest_cn_iv.pdf        -- implied vol time series
    results/backtest_cn_error.pdf     -- PINN vs BSM pricing error
    results/backtest_cn_stats.json    -- summary statistics

US market note:
    yfinance does not provide historical option chains. US backtest uses
    synthetic option prices generated from historical SPY prices and a
    rolling BSM calibration (no live historical bid/ask available).

HK market note:
    No free historical HSI option chain data is available. HK backtest
    is not implemented; the data limitation is documented in the thesis.

Usage:
    python src/backtest.py --market cn --window 60
    python src/backtest.py --market us --window 60
"""

import os
import sys
import json
import argparse
import datetime
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Black-Scholes helpers ──────────────────────────────────────────────────

def bs_call(S, K, T, r, sigma):
    if T <= 1e-6 or sigma <= 1e-6:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def implied_vol(market_price, S, K, T, r):
    """Bisection implied vol solver. Returns NaN if no solution found."""
    if market_price <= max(S - K * np.exp(-r * T), 0.0) + 1e-6:
        return np.nan
    try:
        res = minimize_scalar(
            lambda sig: (bs_call(S, K, T, r, sig) - market_price) ** 2,
            bounds=(1e-4, 5.0), method="bounded"
        )
        return float(res.x) if res.fun < 1e-4 else np.nan
    except Exception:
        return np.nan


# ── A-share (CN) backtest via akshare ─────────────────────────────────────

def backtest_cn(contract_code: str, window: int = 60,
                K: float = None, T_fixed: float = None,
                r: float = 0.02):
    """
    Rolling backtest for a CN 50ETF option contract.

    Args:
        contract_code : 8-digit akshare contract code, e.g. "10004073"
        window        : number of trading days to backtest
        K             : strike price (fetched from chain if None)
        T_fixed       : time to maturity in years at start of window
        r             : risk-free rate (SHIBOR 1Y approx)

    Returns:
        DataFrame-like dict with daily records
    """
    try:
        import akshare as ak
    except ImportError:
        print("akshare not installed. Run: pip install akshare")
        return None

    print(f"[CN] Fetching 50ETF spot history...")
    try:
        etf_df = ak.fund_etf_hist_em(symbol="510050", period="daily", adjust="qfq")
        etf_df = etf_df.sort_values("日期").tail(window + 10).reset_index(drop=True)
        etf_df["日期"] = etf_df["日期"].astype(str)
    except Exception as e:
        print(f"[CN] ETF history fetch failed: {e}")
        return None

    print(f"[CN] Fetching option contract {contract_code} history...")
    try:
        opt_df = ak.option_hist_em(symbol=contract_code)
        opt_df = opt_df.sort_values("日期").tail(window + 10).reset_index(drop=True)
        opt_df["日期"] = opt_df["日期"].astype(str)
    except Exception as e:
        print(f"[CN] Option history fetch failed: {e}")
        return None

    # merge on date
    merged = etf_df.merge(opt_df, on="日期", suffixes=("_etf", "_opt"))
    merged = merged.tail(window).reset_index(drop=True)

    if len(merged) < 5:
        print(f"[CN] Too few overlapping dates ({len(merged)}). Exiting.")
        return None

    # infer K from contract code if not provided
    if K is None:
        try:
            chain = ak.option_finance_board(symbol="510050", end_month="")
            row = chain[chain["期权代码"].astype(str) == contract_code]
            if not row.empty:
                K = float(row.iloc[0]["行权价"])
            else:
                K = float(merged["收盘_etf"].iloc[-1])  # fallback: current spot
        except Exception:
            K = float(merged["收盘_etf"].iloc[-1])

    print(f"[CN] Strike K={K:.4f}, r={r:.3f}, window={len(merged)} days")

    records = []
    for i, row in merged.iterrows():
        S = float(row["收盘_etf"])
        mkt_price = float(row["收盘_opt"])
        date_str = row["日期"]

        # time to maturity: count down from T_fixed if given, else use 0.25
        if T_fixed is not None:
            T = max(T_fixed - i / 252.0, 1 / 365)
        else:
            T = max(0.25 - i / 252.0, 1 / 365)

        iv = implied_vol(mkt_price, S, K, T, r)
        bs_price = bs_call(S, K, T, r, iv if not np.isnan(iv) else 0.2)

        records.append({
            "date": date_str,
            "S": S,
            "K": K,
            "T": round(T, 4),
            "r": r,
            "mkt_price": mkt_price,
            "bs_price": bs_price,
            "iv": iv,
        })

    return records


# ── US backtest via synthetic historical data ──────────────────────────────

def backtest_us(ticker: str = "SPY", window: int = 60,
                moneyness: float = 1.0, T_months: float = 1.0,
                r: float = 0.05):
    """
    Synthetic US backtest: use historical SPY prices + BSM to generate
    synthetic option prices, then calibrate rolling IV.

    Since yfinance does not provide historical option chains, we:
      1. Fetch historical SPY closing prices
      2. Compute 21-day rolling realized vol as IV proxy
      3. Generate synthetic option prices: V = BSM(S, K=S*moneyness, T, r, IV)
      4. This simulates what a rolling ATM option would have been worth

    Note: this is a model-generated backtest, not live market data.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Run: pip install yfinance")
        return None

    print(f"[US] Fetching {ticker} historical prices (synthetic backtest)...")
    try:
        hist = yf.Ticker(ticker).history(period="1y")["Close"]
        hist = hist.tail(window + 30)
    except Exception as e:
        print(f"[US] Price fetch failed: {e}")
        return None

    log_ret = np.log(hist / hist.shift(1)).dropna()
    roll_iv = log_ret.rolling(21).std() * np.sqrt(252)
    roll_iv = roll_iv.dropna()

    T = T_months / 12.0
    records = []
    dates = roll_iv.index[-window:]

    for date in dates:
        S = float(hist.loc[date])
        iv = float(roll_iv.loc[date])
        K = round(S * moneyness / 5) * 5  # round to nearest $5
        mkt_price = bs_call(S, K, T, r, iv)  # synthetic market price
        bs_price = bs_call(S, K, T, r, iv)   # same by construction

        records.append({
            "date": str(date.date()),
            "S": round(S, 2),
            "K": K,
            "T": round(T, 4),
            "r": r,
            "mkt_price": round(mkt_price, 4),
            "bs_price": round(bs_price, 4),
            "iv": round(iv, 4),
        })

    print(f"[US] Generated {len(records)} synthetic records for {ticker}")
    return records


# ── PINN pricing overlay ───────────────────────────────────────────────────

def add_pinn_prices(records: list, market: str) -> list:
    """
    Load trained PINN model and add pinn_price to each record.
    Falls back to BSM price if model weights not found.
    """
    ckpt_path = os.path.join(RESULTS_DIR, "bsm_pinn.pt")
    if not os.path.exists(ckpt_path):
        print(f"[PINN] No checkpoint at {ckpt_path}. Using BSM as PINN proxy.")
        for rec in records:
            rec["pinn_price"] = rec["bs_price"]
        return records

    try:
        from src.models.bsm_pinn import BSM_PINN
        # use parameters from first record as representative
        r0 = records[0]
        pinn = BSM_PINN(K=r0["K"], T=r0["T"], r=r0["r"],
                        sigma=r0.get("iv", 0.2) or 0.2, S_max=300)
        pinn.load(ckpt_path)
        for rec in records:
            pinn.sigma = rec.get("iv", 0.2) or 0.2
            pinn.K = rec["K"]
            pinn.T = rec["T"]
            pinn.r = rec["r"]
            rec["pinn_price"] = round(pinn.price(rec["S"], t=0.0), 4)
        print(f"[PINN] Priced {len(records)} records with BSM-PINN")
    except Exception as e:
        print(f"[PINN] Pricing failed ({e}). Using BSM as proxy.")
        for rec in records:
            rec["pinn_price"] = rec["bs_price"]

    return records


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_backtest(records: list, market: str):
    dates = [r["date"] for r in records]
    ivs = [r["iv"] for r in records]
    mkt = [r["mkt_price"] for r in records]
    bs = [r["bs_price"] for r in records]
    pinn = [r["pinn_price"] for r in records]
    err_bs = [abs(b - m) for b, m in zip(bs, mkt)]
    err_pinn = [abs(p - m) for p, m in zip(pinn, mkt)]

    # x-axis: use integer indices, label every ~10 days
    x = list(range(len(dates)))
    tick_step = max(1, len(dates) // 8)
    tick_pos = x[::tick_step]
    tick_labels = [dates[i] for i in tick_pos]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # top: implied vol time series
    ax = axes[0]
    valid = [(i, v) for i, v in enumerate(ivs) if not np.isnan(v)]
    if valid:
        xi, vi = zip(*valid)
        ax.plot(xi, vi, color="steelblue", linewidth=1.5, label="Implied Vol (BSM)")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Implied Volatility σ")
    ax.set_title(f"[{market.upper()}] Rolling Implied Volatility")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # bottom: pricing error
    ax = axes[1]
    ax.plot(x, err_bs, color="gray", linewidth=1, label="BSM error", linestyle="--")
    ax.plot(x, err_pinn, color="tomato", linewidth=1.5, label="PINN error")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|Model price − Market price|")
    ax.set_title(f"[{market.upper()}] Daily Pricing Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"backtest_{market}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out}")


def save_stats(records: list, market: str):
    mkt = np.array([r["mkt_price"] for r in records])
    bs = np.array([r["bs_price"] for r in records])
    pinn = np.array([r["pinn_price"] for r in records])
    ivs = np.array([r["iv"] for r in records if not np.isnan(r["iv"])])

    stats = {
        "market": market,
        "n_days": len(records),
        "iv_mean": round(float(np.nanmean(ivs)), 4),
        "iv_std": round(float(np.nanstd(ivs)), 4),
        "bsm_mae": round(float(np.mean(np.abs(bs - mkt))), 4),
        "bsm_rmse": round(float(np.sqrt(np.mean((bs - mkt)**2))), 4),
        "pinn_mae": round(float(np.mean(np.abs(pinn - mkt))), 4),
        "pinn_rmse": round(float(np.sqrt(np.mean((pinn - mkt)**2))), 4),
    }

    out = os.path.join(RESULTS_DIR, f"backtest_{market}_stats.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats: {out}")

    print(f"\n── Backtest Summary [{market.upper()}] ──────────────────")
    print(f"  Days          : {stats['n_days']}")
    print(f"  IV mean ± std : {stats['iv_mean']:.4f} ± {stats['iv_std']:.4f}")
    print(f"  BSM  MAE/RMSE : {stats['bsm_mae']:.4f} / {stats['bsm_rmse']:.4f}")
    print(f"  PINN MAE/RMSE : {stats['pinn_mae']:.4f} / {stats['pinn_rmse']:.4f}")
    return stats


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling backtest for PINN option pricing.")
    parser.add_argument("--market", choices=["cn", "us"], default="cn",
                        help="Market to backtest (default: cn)")
    parser.add_argument("--window", type=int, default=60,
                        help="Number of trading days (default: 60)")
    parser.add_argument("--contract", type=str, default="10004073",
                        help="CN contract code (default: 10004073, ignored for US)")
    parser.add_argument("--strike", type=float, default=None,
                        help="Strike price K (auto-detected if not given)")
    args = parser.parse_args()

    if args.market == "cn":
        records = backtest_cn(
            contract_code=args.contract,
            window=args.window,
            K=args.strike,
        )
    elif args.market == "us":
        print("[US] Note: using synthetic BSM-generated prices (no live historical chain).")
        records = backtest_us(window=args.window)
    else:
        print(f"[HK] Historical HSI option data is not available via free APIs.")
        print("     HK backtest is not implemented. See docs/06_multi_market_calibration.md")
        exit(0)

    if records is None or len(records) < 5:
        print("Insufficient data for backtest.")
        exit(1)

    records = add_pinn_prices(records, args.market)
    plot_backtest(records, args.market)
    save_stats(records, args.market)
    print("\nBacktest complete.")
