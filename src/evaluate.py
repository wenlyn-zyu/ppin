"""
Evaluate trained PINN models against analytical / FD reference prices.
Generates comparison tables and plots saved to results/.

Run: python src/evaluate.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.bsm_pinn import BSM_PINN, bs_call_price
from src.models.cev_pinn import CEV_PINN, cev_fd_call
from src.models.heston_pinn import Heston_PINN, heston_call_price

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
RESULTS_DIR = CKPT_DIR


def eval_bsm():
    print("\n── BSM evaluation ──────────────────────────────────────")
    m = BSM_PINN(K=100, T=1.0, r=0.05, sigma=0.2, S_max=300)
    m.load(os.path.join(CKPT_DIR, "bsm_pinn.pt"))

    spots = np.linspace(60, 160, 50)
    pinn_prices, bs_prices, errors = [], [], []
    for S in spots:
        p = m.price(S, t=0.0)
        ref = bs_call_price(S, m.K, m.T, m.r, m.sigma)
        pinn_prices.append(p)
        bs_prices.append(ref)
        errors.append(abs(p - ref))

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  Max  = {max(errors):.4f}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(spots, bs_prices, "k-", label="Black-Scholes")
    axes[0].plot(spots, pinn_prices, "r--", label="BSM-PINN")
    axes[0].set_xlabel("Spot price S")
    axes[0].set_ylabel("Option price V")
    axes[0].set_title("BSM-PINN vs Analytical")
    axes[0].legend()
    axes[1].plot(spots, errors, "b-o")
    axes[1].set_xlabel("Spot price S")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_title(f"BSM absolute error (MAE={mae:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "bsm_eval.pdf"), dpi=150)
    plt.close()
    print("  Plot saved: results/bsm_eval.pdf")
    return mae, rmse


def eval_cev():
    print("\n── CEV evaluation ──────────────────────────────────────")
    m = CEV_PINN(K=100, T=1.0, r=0.05, sigma=0.25, beta=0.5, S_max=300)
    m.load(os.path.join(CKPT_DIR, "cev_pinn.pt"))

    spots = np.linspace(60, 160, 21)
    pinn_prices, fd_prices, errors = [], [], []
    for S in spots:
        p = m.price(S, t=0.0)
        ref = cev_fd_call(S, m.K, m.T, m.r, m.sigma, m.beta)
        pinn_prices.append(p)
        fd_prices.append(ref)
        errors.append(abs(p - ref))

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  Max  = {max(errors):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(spots, fd_prices, "k-", label="FD reference")
    axes[0].plot(spots, pinn_prices, "r--", label="CEV-PINN")
    axes[0].set_xlabel("Spot price S")
    axes[0].set_ylabel("Option price V")
    axes[0].set_title(f"CEV-PINN vs FD (beta={m.beta})")
    axes[0].legend()
    axes[1].plot(spots, errors, "b-o")
    axes[1].set_xlabel("Spot price S")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_title(f"CEV absolute error (MAE={mae:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cev_eval.pdf"), dpi=150)
    plt.close()
    print("  Plot saved: results/cev_eval.pdf")
    return mae, rmse


def eval_heston():
    print("\n── Heston-ICPINN evaluation (Tan & Zhang 2026 params) ──")
    m = Heston_PINN(K=100, T=1.0, r=0.1,
                    kappa=1.0, theta=0.08, xi=0.39, rho=-0.93, v0=0.04,
                    S_max=400, v_max=1.0)
    m.load(os.path.join(CKPT_DIR, "heston_icpinn.pt"))

    spots = np.linspace(60, 160, 21)
    pinn_prices, heston_prices, errors = [], [], []
    for S in spots:
        p = m.price(S, v=m.v0, t=0.0)
        ref = heston_call_price(S, m.K, m.T, m.r,
                                m.kappa, m.theta, m.xi, m.rho, m.v0)
        pinn_prices.append(p)
        heston_prices.append(ref)
        errors.append(abs(p - ref))

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  Max  = {max(errors):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(spots, heston_prices, "k-", label="Heston semi-analytical")
    axes[0].plot(spots, pinn_prices, "r--", label="Heston-PINN")
    axes[0].set_xlabel("Spot price S")
    axes[0].set_ylabel("Option price V")
    axes[0].set_title("Heston-ICPINN vs Semi-analytical (Tan & Zhang 2026)")
    axes[0].legend()
    axes[1].plot(spots, errors, "b-o")
    axes[1].set_xlabel("Spot price S")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_title(f"Heston-ICPINN absolute error (MAE={mae:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "heston_eval.pdf"), dpi=150)
    plt.close()
    print("  Plot saved: results/heston_eval.pdf")
    return mae, rmse


if __name__ == "__main__":
    bsm_mae, bsm_rmse = eval_bsm()
    cev_mae, cev_rmse = eval_cev()
    heston_mae, heston_rmse = eval_heston()

    print("\n── Summary ─────────────────────────────────────────────")
    print(f"{'Model':<10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 32)
    print(f"{'BSM':<10} {bsm_mae:>10.4f} {bsm_rmse:>10.4f}")
    print(f"{'CEV':<10} {cev_mae:>10.4f} {cev_rmse:>10.4f}")
    print(f"{'Heston':<10} {heston_mae:>10.4f} {heston_rmse:>10.4f}")
