"""
eval_compare_market.py  --  run from ~/zhuwl2022/ppin

Compare analytical models, unified PINN v15, and independent PINNs
on real SPY option chain data.

Independent PINNs are trained at fixed params (K=100, sigma=0.2/0.25, etc.).
For market evaluation we normalise each contract to S'=100, K'=100*K/S,
then scale the output back by S/100.  The independent PINN weights are reused
with the normalised inputs -- this is valid because the network learned the
shape of the price surface in normalised coordinates.

Usage:
  cd ~/zhuwl2022/ppin
  conda activate pinn_option
  python eval_compare_market.py \
      --data ../unified_pinn/data/spy_quotedata.csv \
      --unified_ckpt ../unified_pinn/results/unified_v15.pt
"""

import argparse, sys, os, re, datetime, warnings
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import norm
from scipy.integrate import quad

warnings.filterwarnings("ignore")

PPIN_DIR    = os.path.dirname(os.path.abspath(__file__))
UNIFIED_DIR = os.path.join(PPIN_DIR, "..", "unified_pinn")
sys.path.insert(0, PPIN_DIR)
sys.path.insert(0, UNIFIED_DIR)

# ── analytical helpers ────────────────────────────────────────────────────────

def bs_call(S, K, T, r, sigma):
    if sigma <= 0 or T < 1e-8:
        return max(S - K*np.exp(-r*T), 0.0)
    sqt = sigma * np.sqrt(T)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / sqt
    return float(S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d1 - sqt))

def implied_vol(mkt, S, K, T, r):
    intrinsic = max(S - K*np.exp(-r*T), 0.0)
    if mkt <= intrinsic + 1e-4 or mkt <= 0:
        return float("nan")
    try:
        return brentq(lambda s: bs_call(S, K, T, r, s) - mkt, 1e-4, 10.0, xtol=1e-7)
    except Exception:
        return float("nan")

def cev_approx(S, K, T, r, sigma, beta):
    return bs_call(S, K, T, r, max(sigma*(S/K)**(1.0-beta), 1e-4))

def calibrate_cev(calls, S, Ks, T, r):
    w = 1.0 / (np.array(calls) + 0.5)
    def obj(p):
        sig, beta = p
        if sig <= 0 or beta <= 0 or beta > 1: return 1e9
        return float(np.sum(w * (np.array([cev_approx(S,K,T,r,sig,beta) for K in Ks]) - calls)**2))
    best_val, best_p = 1e9, (0.2, 0.5)
    for s0 in [0.10, 0.15, 0.20, 0.25]:
        for b0 in [0.3, 0.5, 0.7, 0.9]:
            res = minimize(obj, [s0, b0], bounds=[(0.01,1.0),(0.01,1.0)],
                           method="L-BFGS-B", options={"maxiter":200})
            if res.fun < best_val:
                best_val, best_p = res.fun, res.x
    return float(best_p[0]), float(best_p[1])

def heston_price(S, K, T, r, kappa, theta, xi, rho, v0):
    if T < 1e-6: return max(S-K, 0.0)
    def integrand(phi, j):
        i = 1j
        u, b = (0.5, kappa-rho*xi) if j==1 else (-0.5, kappa)
        d = np.sqrt((rho*xi*i*phi-b)**2 - xi**2*(2*u*i*phi-phi**2))
        g = (b-rho*xi*i*phi+d)/(b-rho*xi*i*phi-d)
        C = (r*i*phi*T + kappa*theta/xi**2 *
             ((b-rho*xi*i*phi+d)*T - 2*np.log((1-g*np.exp(d*T))/(1-g))))
        D = (b-rho*xi*i*phi+d)/xi**2*(1-np.exp(d*T))/(1-g*np.exp(d*T))
        return np.real(np.exp(C+D*v0+i*phi*np.log(S/K))/(i*phi))
    try:
        P1 = 0.5 + (1/np.pi)*quad(integrand, 0, 100, args=(1,), limit=100)[0]
        P2 = 0.5 + (1/np.pi)*quad(integrand, 0, 100, args=(2,), limit=100)[0]
        return float(S*P1 - K*np.exp(-r*T)*P2)
    except Exception:
        return float("nan")

def calibrate_heston(calls, S, Ks, T, r):
    n = len(Ks)
    idx = np.round(np.linspace(0, n-1, min(n,20))).astype(int)
    Kc, cc = Ks[idx], np.array(calls)[idx]
    w = 1.0/(cc+0.5)
    def obj(p):
        kappa,theta,xi,rho,v0 = p
        if kappa<=0 or theta<=0 or xi<=0 or rho<=-1 or rho>=0 or v0<=0: return 1e9
        prices = np.array([heston_price(S,K,T,r,kappa,theta,xi,rho,v0) for K in Kc])
        if np.any(np.isnan(prices)): return 1e9
        return float(np.sum(w*(prices-cc)**2))
    starts = [[2.0,0.04,0.3,-0.7,0.04],[1.0,0.02,0.2,-0.5,0.02],
              [5.0,0.06,0.5,-0.8,0.06],[8.0,0.04,0.4,-0.9,0.04]]
    best_val, best_p = 1e9, starts[0]
    for x0 in starts:
        res = minimize(obj, x0, bounds=[(0.05,20),(0.001,0.5),(0.01,2.5),(-0.98,-0.01),(0.001,0.5)],
                       method="L-BFGS-B", options={"maxiter":500})
        if res.fun < best_val:
            best_val, best_p = res.fun, res.x
    return tuple(float(x) for x in best_p)

# ── CBOE parser ───────────────────────────────────────────────────────────────

def parse_cboe(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    m = re.search(r"Last:\s*([\d.]+)", lines[1])
    S = float(m.group(1))
    rows = []
    for line in lines[4:]:
        parts = line.strip().split(",")
        if len(parts) < 20: continue
        try:
            rows.append({"expiry_str": parts[0].strip(),
                         "strike": float(parts[11]),
                         "call_last": float(parts[2]) if parts[2].strip() else float("nan"),
                         "call_iv":   float(parts[7]) if parts[7].strip() else float("nan")})
        except (ValueError, IndexError):
            continue
    df = pd.DataFrame(rows)
    df["expiry_date"] = df["expiry_str"].apply(
        lambda s: datetime.datetime.strptime(s.strip(), "%a %b %d %Y").date())
    return S, df

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         default="../unified_pinn/data/spy_quotedata.csv")
    parser.add_argument("--unified_ckpt", default="../unified_pinn/results/unified_v15.pt")
    parser.add_argument("--r", type=float, default=0.043)
    parser.add_argument("--out",          default="results/eval_compare_market.csv")
    args = parser.parse_args()

    today = datetime.date.today()
    r = args.r

    S, df = parse_cboe(args.data)
    print(f"SPY spot: {S:.2f}")
    df["T"] = df["expiry_date"].apply(lambda d: (d - today).days / 365.0)
    df = df[df["T"] > 1/365].copy()
    df = df[df["call_iv"].notna() & (df["call_iv"]>0.01) & (df["call_iv"]<2.0)
            & df["call_last"].notna() & (df["call_last"]>0.10)
            & (df["strike"] >= 0.85*S) & (df["strike"] <= 1.15*S)].copy()

    # load unified PINN
    from unified_pinn_v2 import ModelParams, UnifiedPINN
    param_list = []
    for sigma in [0.13, 0.15, 0.17, 0.20, 0.25, 0.30]:
        param_list.append(ModelParams.from_bsm(K=100., T=1., r=0.05, sigma=sigma))
    for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for sigma in [0.15, 0.20]:
            param_list.append(ModelParams.from_cev(K=100., T=1., r=0.05, sigma=sigma, beta=beta))
    for kappa in [0.5, 2.0, 5.0, 8.0]:
        for xi in [0.1, 0.3, 0.5]:
            for rho in [-0.9, -0.7, -0.5]:
                param_list.append(ModelParams.from_heston(
                    K=100., T=1., r=0.05, kappa=kappa, theta=0.04,
                    xi=xi, rho=rho, v0=0.04))
    unified = UnifiedPINN(param_list, hidden=128, depth=6)
    unified.load(args.unified_ckpt)
    print(f"Loaded unified PINN: {args.unified_ckpt}")

    # load independent PINNs
    from src.models.bsm_pinn import BSM_PINN
    from src.models.cev_pinn import CEV_PINN
    from src.models.heston_pinn import Heston_PINN
    indep_bsm = BSM_PINN(K=100, T=1.0, r=0.05, sigma=0.2, S_max=300)
    indep_bsm.load(os.path.join(PPIN_DIR, "results", "bsm_pinn.pt"))
    indep_cev = CEV_PINN(K=100, T=1.0, r=0.05, sigma=0.25, beta=0.5, S_max=300)
    indep_cev.load(os.path.join(PPIN_DIR, "results", "cev_pinn.pt"))
    indep_hes = Heston_PINN(K=100, T=1.0, r=0.1,
                             kappa=1.0, theta=0.08, xi=0.39, rho=-0.93, v0=0.04,
                             S_max=400, v_max=1.0)
    indep_hes.load(os.path.join(PPIN_DIR, "results", "heston_icpinn.pt"))
    print("Loaded independent PINNs\n")

    scale = S / 100.0
    rows = []

    for exp_date in sorted(df["expiry_date"].unique()):
        grp   = df[df["expiry_date"] == exp_date]
        T_val = grp["T"].iloc[0]
        Ks    = grp["strike"].values
        calls = grp["call_last"].values
        exp_str = grp["expiry_str"].iloc[0]
        n = len(grp)
        print(f"  {exp_str}  T={T_val:.3f}y  n={n}", flush=True)

        # calibrate
        ivs = [implied_vol(c, S, K, T_val, r) for c, K in zip(calls, Ks)]
        sigma_bsm = float(np.nanmedian(ivs)) if any(~np.isnan(v) for v in ivs) else 0.15
        bsm_p = np.array([bs_call(S, K, T_val, r, sigma_bsm) for K in Ks])

        sigma_cev, beta_cev = calibrate_cev(calls, S, Ks, T_val, r)
        cev_p = np.array([cev_approx(S, K, T_val, r, sigma_cev, beta_cev) for K in Ks])

        if T_val < 0.05:
            kappa_h, theta_h, xi_h, rho_h, v0_h = 2.0, 0.04, 0.3, -0.7, sigma_bsm**2
            hes_p = bsm_p.copy()
        else:
            kappa_h, theta_h, xi_h, rho_h, v0_h = calibrate_heston(calls, S, Ks, T_val, r)
            hes_p = np.array([heston_price(S, K, T_val, r, kappa_h, theta_h, xi_h, rho_h, v0_h)
                               for K in Ks])

        # unified PINN: calibrated params, normalised to S'=100
        uni_bsm_p, uni_hes_p = [], []
        for K in Ks:
            K_n = 100.0 * K / S
            T_n = max(T_val, 0.01)
            uni_bsm_p.append(unified.price(
                ModelParams.from_bsm(K=K_n, T=T_n, r=r, sigma=sigma_bsm),
                S=100.0) * scale)
            uni_hes_p.append(unified.price(
                ModelParams.from_heston(K=K_n, T=T_n, r=r,
                                        kappa=kappa_h, theta=theta_h,
                                        xi=xi_h, rho=rho_h, v0=v0_h),
                S=100.0) * scale)

        # independent PINNs: fixed training params, normalised to S'=100
        # The network learned V/K as a function of (S/S_max, t/T).
        # For a contract with strike K_market at spot S_market:
        #   normalised spot = 100 * (S_market / S_market) = 100  (always ATM in normalised space)
        #   normalised strike = 100 * K_market / S_market
        # We pass S'=100 and override K in the model temporarily.
        ind_bsm_p, ind_cev_p, ind_hes_p = [], [], []
        for K in Ks:
            K_n = 100.0 * K / S
            T_n = max(T_val, 0.01)
            # temporarily override K and T (network weights unchanged)
            indep_bsm.K = K_n; indep_bsm.T = T_n
            ind_bsm_p.append(indep_bsm.price(100.0, t=0.0) * scale)

            indep_cev.K = K_n; indep_cev.T = T_n
            ind_cev_p.append(indep_cev.price(100.0, t=0.0) * scale)

            indep_hes.K = K_n; indep_hes.T = T_n
            ind_hes_p.append(indep_hes.price(100.0, v=v0_h, t=0.0) * scale)

        # restore original K/T
        indep_bsm.K = 100; indep_bsm.T = 1.0
        indep_cev.K  = 100; indep_cev.T  = 1.0
        indep_hes.K  = 100; indep_hes.T  = 1.0

        def mae(p): return float(np.mean(np.abs(np.array(p) - calls)))
        rows.append({
            "expiry": exp_str, "T": round(T_val,3), "n": n,
            "sigma_bsm": round(sigma_bsm,4), "beta_cev": round(beta_cev,3),
            "kappa_h": round(kappa_h,3), "xi_h": round(xi_h,3), "rho_h": round(rho_h,3),
            "MAE_BSM":        round(mae(bsm_p),4),
            "MAE_CEV":        round(mae(cev_p),4),
            "MAE_Heston":     round(mae(hes_p),4),
            "MAE_UnifiedBSM": round(mae(uni_bsm_p),4),
            "MAE_UnifiedHes": round(mae(uni_hes_p),4),
            "MAE_IndepBSM":   round(mae(ind_bsm_p),4),
            "MAE_IndepCEV":   round(mae(ind_cev_p),4),
            "MAE_IndepHes":   round(mae(ind_hes_p),4),
        })

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")
    cols = ["expiry","T","n","MAE_BSM","MAE_CEV","MAE_Heston",
            "MAE_UnifiedBSM","MAE_UnifiedHes","MAE_IndepBSM","MAE_IndepCEV","MAE_IndepHes"]
    print("\n" + df_out[cols].to_string(index=False))

if __name__ == "__main__":
    main()
