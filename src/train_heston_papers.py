"""
Train and evaluate the two paper-faithful Heston PINN reproductions.

Model 1: Tan & Zhang (2026) ICPINN  -> results/heston_icpinn.pt
Model 2: Hainaut & Casas (2024)     -> results/heston_hainaut.pt

Run:
  python src/train_heston_papers.py --model icpinn
  python src/train_heston_papers.py --model hainaut
  python src/train_heston_papers.py --model all
  python src/train_heston_papers.py --eval-only --model all
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.heston_icpinn import HestonICPINN, heston_call_price as icpinn_call
from src.models.heston_hainaut import HestonHainaut, heston_put_price

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(CKPT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# ICPINN (Tan & Zhang 2026) — paper params: K=1, r=0.1, kappa=1, theta=0.08,
#   xi=0.39, rho=-0.93, v0=0.04, S_max=4, v_max=1, T=1
# We use K=1 directly to match the paper's scale exactly.
# ---------------------------------------------------------------------------
ICPINN_PARAMS = dict(
    K=1.0, T=1.0, r=0.1,
    kappa=1.0, theta=0.08, xi=0.39, rho=-0.93, v0=0.04,
    S_max=4.0, v_max=1.0,
)


def train_icpinn():
    print("=" * 60)
    print("Training ICPINN (Tan & Zhang 2026) — K=1 paper params")
    print("=" * 60)
    model = HestonICPINN(**ICPINN_PARAMS)
    model.train(
        epochs=20000,
        log_every=1000,
        n_r=10000,
        n_bc=500,
        lr=1e-3,
        pretrain_epochs=10000,
    )
    path = os.path.join(CKPT_DIR, "heston_icpinn.pt")
    model.save(path)
    print(f"ICPINN checkpoint saved: {path}\n")
    return model


def eval_icpinn(model=None):
    print("\n── ICPINN evaluation (Tan & Zhang 2026) ───────────────")
    if model is None:
        model = HestonICPINN(**ICPINN_PARAMS)
        model.load(os.path.join(CKPT_DIR, "heston_icpinn.pt"))

    # Evaluate at tau=T (pricing today), v=v0=0.04
    # Paper uses K=1, S in [0.5, 1.5] for evaluation grid
    spots = np.linspace(0.5, 1.5, 21)
    pinn_prices, ref_prices, errors = [], [], []
    for S in spots:
        p = model.price(S, v=ICPINN_PARAMS["v0"], tau=ICPINN_PARAMS["T"])
        ref = icpinn_call(
            S, ICPINN_PARAMS["K"], ICPINN_PARAMS["T"], ICPINN_PARAMS["r"],
            ICPINN_PARAMS["kappa"], ICPINN_PARAMS["theta"],
            ICPINN_PARAMS["xi"], ICPINN_PARAMS["rho"], ICPINN_PARAMS["v0"],
        )
        pinn_prices.append(p)
        ref_prices.append(ref)
        errors.append(abs(p - ref))

    mae  = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    rel  = np.mean(np.array(errors) / (np.array(ref_prices) + 1e-8))
    print(f"  MAE  = {mae:.6f}")
    print(f"  RMSE = {rmse:.6f}")
    print(f"  MRE  = {rel*100:.2f}%")
    print(f"  Max  = {max(errors):.6f}")

    # Print table
    print(f"\n  {'S':>6}  {'PINN':>10}  {'Ref':>10}  {'|err|':>10}")
    print("  " + "-" * 42)
    for S, p, r, e in zip(spots, pinn_prices, ref_prices, errors):
        print(f"  {S:6.3f}  {p:10.6f}  {r:10.6f}  {e:10.6f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(spots, ref_prices, "k-", label="Semi-analytical")
    axes[0].plot(spots, pinn_prices, "r--", label="ICPINN")
    axes[0].set_xlabel("S (K=1 units)")
    axes[0].set_ylabel("Call price")
    axes[0].set_title("ICPINN vs Semi-analytical (Tan & Zhang 2026)")
    axes[0].legend()
    axes[1].plot(spots, errors, "b-o")
    axes[1].set_xlabel("S")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_title(f"ICPINN error (MAE={mae:.4e})")
    plt.tight_layout()
    out = os.path.join(CKPT_DIR, "heston_icpinn_eval.pdf")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot saved: {out}")
    return mae, rmse


# ---------------------------------------------------------------------------
# Hainaut & Casas (2024) — parametric PUT, K=100
# Table 6 evaluation params: kappa=1.15, r=0.04, theta=0.202, xi=0.20, rho=-0.40
# V0 (initial variance) not in Table 6; use midpoint of V_RANGE = 0.276
# ---------------------------------------------------------------------------
HAINAUT_EVAL = dict(
    S=100.0, V=0.276, t=1.0, T=1.0,   # t=T means "now" (no time elapsed)
    r=0.04, kappa=1.15, theta=0.202, xi=0.20, rho=-0.40,
)


def train_hainaut():
    print("=" * 60)
    print("Training Hainaut & Casas (2024) parametric PINN")
    print("=" * 60)
    model = HestonHainaut()
    model.train(log_every=500)
    path = os.path.join(CKPT_DIR, "heston_hainaut.pt")
    model.save(path)
    print(f"Hainaut checkpoint saved: {path}\n")
    return model


def eval_hainaut(model=None):
    print("\n── Hainaut & Casas (2024) evaluation ──────────────────")
    if model is None:
        model = HestonHainaut()
        model.load(os.path.join(CKPT_DIR, "heston_hainaut.pt"))

    # Evaluate across spot prices, fixed params from Table 6
    spots = np.linspace(60, 140, 21)
    ep = HAINAUT_EVAL
    pinn_prices, ref_prices, errors = [], [], []
    for S in spots:
        p = model.price(S=S, V=ep["V"], t=ep["t"], T=ep["T"],
                        r=ep["r"], kappa=ep["kappa"], theta=ep["theta"],
                        xi=ep["xi"], rho=ep["rho"])
        ref = heston_put_price(S, HestonHainaut.K_STRIKE, ep["T"],
                               ep["r"], ep["kappa"], ep["theta"],
                               ep["xi"], ep["rho"], ep["V"])
        pinn_prices.append(p)
        ref_prices.append(ref)
        errors.append(abs(p - ref))

    mae  = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    rel  = np.mean(np.array(errors) / (np.array(ref_prices) + 1e-8))
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MRE  = {rel*100:.2f}%")
    print(f"  Max  = {max(errors):.4f}")

    print(f"\n  {'S':>6}  {'PINN':>10}  {'Ref':>10}  {'|err|':>10}")
    print("  " + "-" * 42)
    for S, p, r, e in zip(spots, pinn_prices, ref_prices, errors):
        print(f"  {S:6.1f}  {p:10.4f}  {r:10.4f}  {e:10.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(spots, ref_prices, "k-", label="Semi-analytical")
    axes[0].plot(spots, pinn_prices, "r--", label="Hainaut PINN")
    axes[0].set_xlabel("Spot price S")
    axes[0].set_ylabel("Put price")
    axes[0].set_title("Hainaut & Casas (2024) vs Semi-analytical")
    axes[0].legend()
    axes[1].plot(spots, errors, "b-o")
    axes[1].set_xlabel("S")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_title(f"Hainaut error (MAE={mae:.4f})")
    plt.tight_layout()
    out = os.path.join(CKPT_DIR, "heston_hainaut_eval.pdf")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot saved: {out}")
    return mae, rmse


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["icpinn", "hainaut", "all"],
                        default="all")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, load checkpoints and evaluate only")
    args = parser.parse_args()

    results = {}

    if args.model in ("icpinn", "all"):
        if not args.eval_only:
            m = train_icpinn()
            mae, rmse = eval_icpinn(m)
        else:
            mae, rmse = eval_icpinn()
        results["ICPINN"] = (mae, rmse)

    if args.model in ("hainaut", "all"):
        if not args.eval_only:
            m = train_hainaut()
            mae, rmse = eval_hainaut(m)
        else:
            mae, rmse = eval_hainaut()
        results["Hainaut"] = (mae, rmse)

    print("\n── Summary ─────────────────────────────────────────────")
    print(f"{'Model':<12} {'MAE':>12} {'RMSE':>12}")
    print("-" * 38)
    for name, (mae, rmse) in results.items():
        print(f"{name:<12} {mae:>12.6f} {rmse:>12.6f}")
