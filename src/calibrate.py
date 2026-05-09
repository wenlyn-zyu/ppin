"""
Parameter calibration: fit BSM and Heston model parameters to real
option chain data from US (SPY/yfinance), A-share (50ETF/akshare),
or HK (HSI options/yfinance) markets.

Usage:
    python src/calibrate.py --market us      # S&P 500 via SPY
    python src/calibrate.py --market cn      # A-share 50ETF (510050.SH)
    python src/calibrate.py --market hk      # Hang Seng Index options

Output:
    results/calibration_<market>_bsm.json
    results/calibration_<market>_heston.json
    results/calibration_<market>_plot.pdf
"""

import json
import warnings
import datetime
import argparse
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Black-Scholes helpers ──────────────────────────────────────────────────

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_implied_vol(price, S, K, T, r, tol=1e-6, max_iter=100):
    """Newton-Raphson implied volatility solver."""
    sigma = 0.3
    for _ in range(max_iter):
        p = bs_call(S, K, T, r, sigma)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        if vega < 1e-10:
            break
        sigma -= (p - price) / vega
        sigma = max(sigma, 1e-4)
        if abs(p - price) < tol:
            break
    return sigma


# ── Heston characteristic function (Gil-Pelaez) ───────────────────────────

def heston_call(S, K, T, r, kappa, theta, xi, rho, v0):
    """Semi-analytical Heston call price via characteristic function."""
    from scipy.integrate import quad

    def char_func(phi, j):
        if j == 1:
            u, b = 0.5, kappa - rho * xi
        else:
            u, b = -0.5, kappa
        a = kappa * theta
        x = np.log(S)
        d = np.sqrt((rho * xi * phi * 1j - b)**2 - xi**2 * (2 * u * phi * 1j - phi**2))
        g = (b - rho * xi * phi * 1j + d) / (b - rho * xi * phi * 1j - d)
        C = r * phi * 1j * T + (a / xi**2) * (
            (b - rho * xi * phi * 1j + d) * T
            - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))
        )
        D = ((b - rho * xi * phi * 1j + d) / xi**2) * (
            (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
        )
        return np.exp(C + D * v0 + 1j * phi * x)

    def integrand(phi, j):
        return np.real(np.exp(-1j * phi * np.log(K)) * char_func(phi, j) / (1j * phi))

    P1 = 0.5 + (1 / np.pi) * quad(integrand, 0, 200, args=(1,), limit=200)[0]
    P2 = 0.5 + (1 / np.pi) * quad(integrand, 0, 200, args=(2,), limit=200)[0]
    return S * P1 - K * np.exp(-r * T) * P2


# ── Market data adapters ───────────────────────────────────────────────────

class MarketDataAdapter:
    """
    Base class for market data adapters.
    Subclasses implement fetch() and return a list of unified records:
        [{"K": float, "T": float, "mid": float, "S": float, "r": float}, ...]
    plus (S, r) scalars for the underlying and risk-free rate.
    """
    market_name: str = ""
    r_default: float = 0.05       # risk-free rate
    option_style: str = "european"

    def fetch(self, max_strikes: int = 8):
        """Returns (records, S, r). Override in subclasses."""
        raise NotImplementedError


class USMarketAdapter(MarketDataAdapter):
    """
    US market: SPY options via yfinance.
    - Risk-free rate: ~5% (Fed Funds rate, 2024)
    - Option style: American (SPY is ETF, options are American-style)
    - Contract multiplier: 100 shares (already reflected in yfinance bid/ask)
    - Liquidity filter: volume > 10, moneyness 80%–120%
    """
    market_name = "us"
    r_default = 0.05
    option_style = "american"

    def fetch(self, ticker="SPY", max_strikes=8):
        try:
            import yfinance as yf
        except ImportError:
            print("yfinance not installed. Run: pip install yfinance")
            return None

        tk = yf.Ticker(ticker)
        S = tk.fast_info["last_price"]
        r = self.r_default
        today = datetime.date.today()
        expirations = tk.options[:3]
        records = []

        for exp in expirations:
            chain = tk.option_chain(exp)
            calls = chain.calls
            exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
            T = max((exp_date - today).days / 365.0, 1 / 365)

            calls = calls[(calls["strike"] >= 0.80 * S) & (calls["strike"] <= 1.20 * S)]
            calls = calls[calls["volume"] > 10]
            calls = calls.nsmallest(max_strikes, key=lambda x: abs(x["strike"] - S))

            for _, row in calls.iterrows():
                mid = (row["bid"] + row["ask"]) / 2
                if mid > 0.5:
                    records.append({"K": row["strike"], "T": T, "mid": mid, "S": S, "r": r})

        print(f"[US] Fetched {len(records)} quotes for {ticker} (S={S:.2f}, r={r:.3f})")
        return records, S, r


class CNMarketAdapter(MarketDataAdapter):
    """
    A-share market: 50ETF options (510050.SH) via akshare.
    - Risk-free rate: ~2% (1-year SHIBOR, 2024)
    - Option style: European (SSE 50ETF options are European)
    - Contract multiplier: 10,000 units (prices in CNY per unit)
    - Liquidity filter: open_interest > 100, moneyness 90%–110%
    - Contract naming: e.g. "10004073" (akshare option_finance_board)
    """
    market_name = "cn"
    r_default = 0.02
    option_style = "european"

    def fetch(self, symbol="510050", max_strikes=8):
        try:
            import akshare as ak
        except ImportError:
            print("akshare not installed. Run: pip install akshare")
            return None

        r = self.r_default

        # get current ETF price
        try:
            spot_df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                          adjust="qfq")
            S = float(spot_df["收盘"].iloc[-1])
        except Exception as e:
            print(f"[CN] Failed to fetch spot price: {e}")
            return None

        # get option chain
        try:
            opt_df = ak.option_finance_board(symbol=symbol, end_month="")
        except Exception as e:
            print(f"[CN] Failed to fetch option chain: {e}")
            return None

        today = datetime.date.today()
        records = []

        # akshare returns columns: 期权代码, 名称, 最新价, 涨跌幅, 买量, 买价, 卖价, 卖量,
        #                          持仓量, 成交量, 行权价, 到期日, 剩余日历天数, Delta
        for _, row in opt_df.iterrows():
            try:
                name = str(row.get("名称", ""))
                # only take call options (购 = call in Chinese)
                if "购" not in name:
                    continue

                K = float(row["行权价"])
                # moneyness filter: 90%–110%
                if not (0.90 * S <= K <= 1.10 * S):
                    continue

                oi = float(row.get("持仓量", 0))
                if oi < 100:
                    continue

                bid = float(row.get("买价", 0))
                ask = float(row.get("卖价", 0))
                if bid <= 0 or ask <= 0:
                    continue
                mid = (bid + ask) / 2

                # parse expiry: akshare format "YYYY-MM-DD" or "YYYYMMDD"
                exp_raw = str(row.get("到期日", ""))
                exp_raw = exp_raw.replace("-", "")
                if len(exp_raw) == 8:
                    exp_date = datetime.datetime.strptime(exp_raw, "%Y%m%d").date()
                else:
                    continue
                T = max((exp_date - today).days / 365.0, 1 / 365)

                records.append({"K": K, "T": T, "mid": mid, "S": S, "r": r})
            except Exception:
                continue

        # keep nearest max_strikes strikes per expiry group
        if records:
            records.sort(key=lambda x: abs(x["K"] - S))
            records = records[:max_strikes * 3]

        print(f"[CN] Fetched {len(records)} quotes for {symbol} (S={S:.4f}, r={r:.3f})")
        return records, S, r


class HKMarketAdapter(MarketDataAdapter):
    """
    HK market: Hang Seng Index options via yfinance (^HSI).
    - Risk-free rate: ~4% (1-month HIBOR, 2024)
    - Option style: European (HKEX HSI options are European)
    - Contract multiplier: HKD 50 per index point (reflected in yfinance prices)
    - Liquidity filter: volume > 5, moneyness 85%–115%
    - Note: yfinance HSI option coverage is limited; falls back to synthetic
      data if fewer than 5 quotes are available.
    """
    market_name = "hk"
    r_default = 0.04
    option_style = "european"

    def fetch(self, ticker="^HSI", max_strikes=8):
        try:
            import yfinance as yf
        except ImportError:
            print("yfinance not installed. Run: pip install yfinance")
            return None

        tk = yf.Ticker(ticker)
        try:
            S = tk.fast_info["last_price"]
        except Exception:
            print("[HK] Could not fetch HSI spot price from yfinance.")
            return None

        r = self.r_default
        today = datetime.date.today()
        expirations = tk.options[:3] if tk.options else []
        records = []

        for exp in expirations:
            try:
                chain = tk.option_chain(exp)
                calls = chain.calls
            except Exception:
                continue

            exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
            T = max((exp_date - today).days / 365.0, 1 / 365)

            calls = calls[(calls["strike"] >= 0.85 * S) & (calls["strike"] <= 1.15 * S)]
            calls = calls[calls["volume"] > 5]
            calls = calls.nsmallest(max_strikes, key=lambda x: abs(x["strike"] - S))

            for _, row in calls.iterrows():
                mid = (row["bid"] + row["ask"]) / 2
                if mid > 1.0:
                    records.append({"K": row["strike"], "T": T, "mid": mid, "S": S, "r": r})

        # fallback: synthesize ATM quotes from BSM implied vol if data is sparse
        if len(records) < 5:
            print(f"[HK] Only {len(records)} live quotes; supplementing with BSM-implied quotes.")
            sigma_atm = 0.20  # typical HSI ATM vol
            for moneyness in [0.92, 0.96, 1.00, 1.04, 1.08]:
                K = round(S * moneyness / 100) * 100  # round to nearest 100 pts
                for T_months in [1/12, 3/12]:
                    T = T_months
                    mid = bs_call(S, K, T, r, sigma_atm)
                    if mid > 1.0:
                        records.append({"K": K, "T": T, "mid": mid, "S": S, "r": r})

        print(f"[HK] Fetched {len(records)} quotes for {ticker} (S={S:.0f}, r={r:.3f})")
        return records, S, r


MARKET_ADAPTERS = {
    "us": USMarketAdapter,
    "cn": CNMarketAdapter,
    "hk": HKMarketAdapter,
}


# ── BSM calibration ───────────────────────────────────────────────────────

def calibrate_bsm(records):
    """Fit a single flat implied vol (BSM) to minimize RMSE."""
    def objective(params):
        sigma = params[0]
        errors = []
        for rec in records:
            model = bs_call(rec["S"], rec["K"], rec["T"], rec["r"], sigma)
            errors.append((model - rec["mid"])**2)
        return np.mean(errors)

    res = minimize(objective, x0=[0.2], bounds=[(0.01, 2.0)], method="L-BFGS-B")
    return float(res.x[0])


# ── Heston calibration ────────────────────────────────────────────────────

def calibrate_heston(records):
    """Fit Heston parameters to minimize RMSE on option prices."""
    def objective(params):
        kappa, theta, xi, rho, v0 = params
        errors = []
        for rec in records:
            try:
                model = heston_call(rec["S"], rec["K"], rec["T"], rec["r"],
                                    kappa, theta, xi, rho, v0)
                errors.append((model - rec["mid"])**2)
            except Exception:
                errors.append(1e6)
        return np.mean(errors)

    x0 = [2.0, 0.04, 0.3, -0.7, 0.04]
    bounds = [(0.1, 10), (0.01, 1), (0.01, 2), (-0.99, 0), (0.001, 1)]
    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": 200})
    keys = ["kappa", "theta", "xi", "rho", "v0"]
    return dict(zip(keys, res.x.tolist())), float(res.fun)


# ── Plot: market vs model prices ──────────────────────────────────────────

def plot_calibration(records, sigma_bsm, heston_params, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (label, model_fn) in zip(axes, [
        ("BSM (σ={:.3f})".format(sigma_bsm),
         lambda r: bs_call(r["S"], r["K"], r["T"], r["r"], sigma_bsm)),
        ("Heston",
         lambda r: heston_call(r["S"], r["K"], r["T"], r["r"], **heston_params)),
    ]):
        market = [r["mid"] for r in records]
        model  = [model_fn(r) for r in records]
        strikes = [r["K"] for r in records]

        ax.scatter(strikes, market, label="Market", color="black", zorder=3)
        ax.plot(sorted(strikes), [m for _, m in sorted(zip(strikes, model))],
                label=label, color="steelblue")
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Call Price")
        ax.set_title(f"Market vs {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved calibration plot to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    parser = argparse.ArgumentParser(description="Calibrate BSM/Heston to market data.")
    parser.add_argument("--market", choices=["us", "cn", "hk"], default="us",
                        help="Market to calibrate against (default: us)")
    parser.add_argument("--max-strikes", type=int, default=8,
                        help="Max strikes per expiry (default: 8)")
    args = parser.parse_args()

    market = args.market
    adapter = MARKET_ADAPTERS[market]()
    result = adapter.fetch(max_strikes=args.max_strikes)

    if result is None:
        print(f"Cannot fetch data for market '{market}'. Exiting.")
        exit(1)

    records, S, r = result
    if len(records) < 3:
        print(f"Too few quotes ({len(records)}) for calibration. Exiting.")
        exit(1)

    print(f"\n--- BSM Calibration [{market.upper()}] ---")
    sigma_bsm = calibrate_bsm(records)
    print(f"Calibrated sigma = {sigma_bsm:.4f}")
    bsm_out = f"results/calibration_{market}_bsm.json"
    with open(bsm_out, "w") as f:
        json.dump({"sigma": sigma_bsm, "S": S, "r": r,
                   "market": market, "n_quotes": len(records)}, f, indent=2)
    print(f"Saved to {bsm_out}")

    print(f"\n--- Heston Calibration [{market.upper()}] ---")
    heston_params, rmse = calibrate_heston(records)
    print("Calibrated Heston parameters:")
    for k, v in heston_params.items():
        print(f"  {k} = {v:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    heston_out = f"results/calibration_{market}_heston.json"
    with open(heston_out, "w") as f:
        json.dump({**heston_params, "rmse": rmse, "S": S, "r": r,
                   "market": market}, f, indent=2)
    print(f"Saved to {heston_out}")

    plot_out = f"results/calibration_{market}_plot.pdf"
    plot_calibration(records, sigma_bsm, heston_params, plot_out)
    print(f"\nCalibration complete for market: {market.upper()}")
