"""
eval_compare_synthetic.py  --  run from ~/zhuwl2022/ppin

Compare independent PINNs vs unified PINN v15 on the same synthetic grid.

Synthetic params (match independent PINN training):
  BSM:    K=100, T=1, r=0.05, sigma=0.2
  CEV:    K=100, T=1, r=0.05, sigma=0.25, beta=0.5
  Heston: K=100, T=1, r=0.1,  kappa=1, theta=0.08, xi=0.39, rho=-0.93, v0=0.04

Eval grid: S in [60,160], 50 pts (21 for CEV due to MC cost), t=0

Usage:
  cd ~/zhuwl2022/ppin
  conda activate pinn_option
  python eval_compare_synthetic.py
"""

import sys, os
import numpy as np
import pandas as pd

# ── import paths ─────────────────────────────────────────────────────────────
PPIN_DIR    = os.path.dirname(os.path.abspath(__file__))          # ~/zhuwl2022/ppin
UNIFIED_DIR = os.path.join(PPIN_DIR, "..", "unified_pinn")        # ~/zhuwl2022/unified_pinn
sys.path.insert(0, PPIN_DIR)
sys.path.insert(0, UNIFIED_DIR)

INDEP_CKPT  = os.path.join(PPIN_DIR,    "results")
UNIFIED_CKPT = os.path.join(UNIFIED_DIR, "results", "unified_v15.pt")
OUT_CSV     = os.path.join(PPIN_DIR,    "results", "eval_compare_synthetic.csv")

# ── reference prices ─────────────────────────────────────────────────────────

def bsm_ref(spots):
    from src.models.bsm_pinn import bs_call_price
    return np.array([bs_call_price(S, 100, 1.0, 0.05, 0.2) for S in spots])

def cev_mc_ref(spots, N_paths=500_000, N_steps=500, seed=42):
    np.random.seed(seed)
    K, T, r, sigma, beta = 100, 1.0, 0.05, 0.25, 0.5
    dt = T / N_steps
    refs = []
    print("  MC reference (500k×500 steps)...", flush=True)
    for S0 in spots:
        S = np.full(N_paths, float(S0))
        for _ in range(N_steps):
            dW = np.random.randn(N_paths) * np.sqrt(dt)
            S = S + r*S*dt + sigma*(np.abs(S)**beta)*dW
            S = np.maximum(S, 0.0)
        refs.append(float(np.exp(-r*T) * np.mean(np.maximum(S-K, 0.0))))
    return np.array(refs)

def heston_ref(spots):
    from src.models.heston_pinn import heston_call_price
    return np.array([heston_call_price(S, 100, 1.0, 0.1, 1.0, 0.08, 0.39, -0.93, 0.04)
                     for S in spots])

# ── independent PINN prices ───────────────────────────────────────────────────

def indep_bsm(spots):
    from src.models.bsm_pinn import BSM_PINN
    m = BSM_PINN(K=100, T=1.0, r=0.05, sigma=0.2, S_max=300)
    m.load(os.path.join(INDEP_CKPT, "bsm_pinn.pt"))
    return np.array([m.price(S, t=0.0) for S in spots])

def indep_cev(spots):
    from src.models.cev_pinn import CEV_PINN
    m = CEV_PINN(K=100, T=1.0, r=0.05, sigma=0.25, beta=0.5, S_max=300)
    m.load(os.path.join(INDEP_CKPT, "cev_pinn.pt"))
    return np.array([m.price(S, t=0.0) for S in spots])

def indep_heston(spots):
    from src.models.heston_pinn import Heston_PINN
    m = Heston_PINN(K=100, T=1.0, r=0.1,
                    kappa=1.0, theta=0.08, xi=0.39, rho=-0.93, v0=0.04,
                    S_max=400, v_max=1.0)
    m.load(os.path.join(INDEP_CKPT, "heston_icpinn.pt"))
    return np.array([m.price(S, v=0.04, t=0.0) for S in spots])

# ── unified PINN ─────────────────────────────────────────────────────────────

def load_unified():
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
    m = UnifiedPINN(param_list, hidden=128, depth=6)
    m.load(UNIFIED_CKPT)
    return m

def unified_bsm(spots, m):
    from unified_pinn_v2 import ModelParams
    p = ModelParams.from_bsm(K=100., T=1., r=0.05, sigma=0.2)
    return np.array([m.price(p, S=S, t=0.0) for S in spots])

def unified_cev(spots, m):
    from unified_pinn_v2 import ModelParams
    p = ModelParams.from_cev(K=100., T=1., r=0.05, sigma=0.25, beta=0.5)
    return np.array([m.price(p, S=S, t=0.0) for S in spots])

def unified_heston(spots, m):
    from unified_pinn_v2 import ModelParams
    # Note: unified PINN trained with r=0.05; indep Heston uses r=0.1
    # We use the same params as indep for fair comparison
    p = ModelParams.from_heston(K=100., T=1., r=0.1,
                                kappa=1.0, theta=0.08, xi=0.39, rho=-0.93, v0=0.04)
    return np.array([m.price(p, S=S, v=0.04, t=0.0) for S in spots])

# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(pred, ref, thr=0.5):
    err = np.abs(pred - ref)
    mae  = float(np.mean(err))
    rmse = float(np.sqrt(np.mean((pred-ref)**2)))
    mask = ref > thr
    relmae = float(np.mean(err[mask]/ref[mask])) if mask.any() else float("nan")
    return mae, rmse, relmae

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    spots50 = np.linspace(60, 160, 50)
    spots21 = np.linspace(60, 160, 21)

    print("Loading unified PINN v15...", flush=True)
    uni = load_unified()

    rows = []

    # BSM
    print("\n── BSM ──────────────────────────────────────────────────")
    ref = bsm_ref(spots50)
    for label, pred in [("IndepPINN-BSM",   indep_bsm(spots50)),
                        ("UnifiedPINN-BSM", unified_bsm(spots50, uni))]:
        mae, rmse, relmae = metrics(pred, ref)
        print(f"  {label:<22}  MAE={mae:.4f}  RMSE={rmse:.4f}  RelMAE={100*relmae:.2f}%")
        rows.append(dict(model="BSM", method=label, ref="analytical",
                         n_pts=50, MAE=round(mae,4), RMSE=round(rmse,4),
                         RelMAE_pct=round(100*relmae,2)))

    # CEV
    print("\n── CEV ──────────────────────────────────────────────────")
    ref = cev_mc_ref(spots21)
    for label, pred in [("IndepPINN-CEV",   indep_cev(spots21)),
                        ("UnifiedPINN-CEV", unified_cev(spots21, uni))]:
        mae, rmse, relmae = metrics(pred, ref)
        print(f"  {label:<22}  MAE={mae:.4f}  RMSE={rmse:.4f}  RelMAE={100*relmae:.2f}%")
        rows.append(dict(model="CEV", method=label, ref="MC(500k×500)",
                         n_pts=21, MAE=round(mae,4), RMSE=round(rmse,4),
                         RelMAE_pct=round(100*relmae,2)))

    # Heston
    print("\n── Heston ───────────────────────────────────────────────")
    ref = heston_ref(spots50)
    for label, pred in [("IndepPINN-Heston",   indep_heston(spots50)),
                        ("UnifiedPINN-Heston", unified_heston(spots50, uni))]:
        mae, rmse, relmae = metrics(pred, ref)
        print(f"  {label:<24}  MAE={mae:.4f}  RMSE={rmse:.4f}  RelMAE={100*relmae:.2f}%")
        rows.append(dict(model="Heston", method=label, ref="char-func",
                         n_pts=50, MAE=round(mae,4), RMSE=round(rmse,4),
                         RelMAE_pct=round(100*relmae,2)))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print("\n" + df.to_string(index=False))

if __name__ == "__main__":
    main()
