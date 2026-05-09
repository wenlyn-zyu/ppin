"""
Parameter calibration: fit BSM and Heston model parameters to real
SPX option chain data fetched via yfinance.

Usage:
    python src/calibrate.py

Output:
    results/calibration_bsm.json   -- calibrated BSM sigma
    results/calibration_heston.json -- calibrated Heston params
    results/calibration_plot.pdf   -- market vs model IV surface
"""

import json
import warnings
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


# ── Fetch SPX option data ──────────────────────────────────────────────────

def fetch_spx_options(ticker="SPY", max_strikes=8):
    """
    Fetch near-term ATM options from yfinance.
    Returns list of dicts: {K, T, mid_price, S, r}
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Run: pip install yfinance")
        return None

    tk = yf.Ticker(ticker)
    S = tk.fast_info["last_price"]
    r = 0.05  # approximate risk-free rate

    expirations = tk.options[:3]  # use first 3 expiries
    records = []

    import datetime
    today = datetime.date.today()

    for exp in expirations:
        chain = tk.option_chain(exp)
        calls = chain.calls
        exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
        T = max((exp_date - today).days / 365.0, 1 / 365)

        # filter near-ATM strikes (80%–120% moneyness)
        calls = calls[(calls["strike"] >= 0.80 * S) & (calls["strike"] <= 1.20 * S)]
        calls = calls[calls["volume"] > 10]  # liquidity filter
        calls = calls.nsmallest(max_strikes, key=lambda x: abs(x["strike"] - S))

        for _, row in calls.iterrows():
            mid = (row["bid"] + row["ask"]) / 2
            if mid > 0.5:
                records.append({"K": row["strike"], "T": T, "mid": mid, "S": S, "r": r})

    print(f"Fetched {len(records)} option quotes for {ticker} (S={S:.1f})")
    return records, S, r


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

    result = fetch_spx_options("SPY")
    if result is None:
        print("Cannot fetch data. Exiting.")
        exit(1)

    records, S, r = result

    print("\n--- BSM Calibration ---")
    sigma_bsm = calibrate_bsm(records)
    print(f"Calibrated sigma = {sigma_bsm:.4f}")
    with open("results/calibration_bsm.json", "w") as f:
        json.dump({"sigma": sigma_bsm, "S": S, "r": r, "n_quotes": len(records)}, f, indent=2)

    print("\n--- Heston Calibration ---")
    heston_params, rmse = calibrate_heston(records)
    print("Calibrated Heston parameters:")
    for k, v in heston_params.items():
        print(f"  {k} = {v:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    with open("results/calibration_heston.json", "w") as f:
        json.dump({**heston_params, "rmse": rmse, "S": S, "r": r}, f, indent=2)

    plot_calibration(records, sigma_bsm, heston_params, "results/calibration_plot.pdf")
    print("\nCalibration complete.")
