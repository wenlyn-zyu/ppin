"""
eval_heston_icpinn.py — Evaluate Heston ICPINN checkpoint vs semi-analytical reference.

Usage:
  python eval_heston_icpinn.py [--ckpt results/heston_icpinn.pt] [--out results/]
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.models.heston_pinn import Heston_PINN, heston_call_price


def metrics(pred, ref):
    err = np.abs(pred - ref)
    mask = ref > 0.5
    rel = float(np.mean(err[mask] / ref[mask])) if mask.any() else float("nan")
    return {
        "MAE":    float(np.mean(err)),
        "RMSE":   float(np.sqrt(np.mean(err**2))),
        "MaxErr": float(np.max(err)),
        "RelMAE": rel,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="results/heston_icpinn.pt")
    parser.add_argument("--out",  type=str, default="results/")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Must match train_heston_icpinn.py params exactly
    K, T, r = 100.0, 1.0, 0.1
    kappa, theta, xi, rho, v0 = 1.0, 0.08, 0.39, -0.93, 0.04

    model = Heston_PINN(
        K=K, T=T, r=r,
        kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0,
        S_max=400.0, v_max=1.0,
        device=device,
    )
    model.load(args.ckpt)
    print(f"Loaded checkpoint: {args.ckpt}")

    # Evaluate at t=0 (tau=T), S in [60, 160]
    S_grid = np.linspace(60, 160, 50)

    print("Computing PINN prices...")
    pred = np.array([model.price(S, v=v0, t=0.0) for S in S_grid])

    print("Computing semi-analytical reference prices...")
    ref = np.array([
        heston_call_price(S, K, T, r, kappa, theta, xi, rho, v0)
        for S in S_grid
    ])

    m = metrics(pred, ref)
    print(f"\n=== Heston ICPINN Evaluation (Tan & Zhang 2026 params) ===")
    print(f"  K={K}, T={T}, r={r}, kappa={kappa}, theta={theta}")
    print(f"  xi={xi}, rho={rho}, v0={v0}")
    print(f"  MAE    = {m['MAE']:.4f}")
    print(f"  RMSE   = {m['RMSE']:.4f}")
    print(f"  MaxErr = {m['MaxErr']:.4f}")
    print(f"  RelMAE = {m['RelMAE']:.4f}  (ref > 0.5 only)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(S_grid, ref,  "--", label="Semi-analytical", lw=2, color="black")
        axes[0].plot(S_grid, pred,        label="ICPINN",          lw=2, color="tab:blue")
        axes[0].set_title("Heston Call Price (t=0, paper params)")
        axes[0].set_xlabel("S")
        axes[0].set_ylabel("V")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(S_grid, np.abs(pred - ref), lw=2, color="tab:red")
        axes[1].set_title(f"Absolute Error  (MAE={m['MAE']:.4f})")
        axes[1].set_xlabel("S")
        axes[1].set_ylabel("|PINN - Ref|")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(args.out, "heston_icpinn_curve.pdf")
        plt.savefig(fig_path)
        print(f"\nPlot saved to {fig_path}")
    except Exception as e:
        print(f"[Plot skipped] {e}")


if __name__ == "__main__":
    main()
