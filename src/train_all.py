"""
Train all three PINN models and save checkpoints.
Run: python src/train_all.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.bsm_pinn import BSM_PINN
from src.models.cev_pinn import CEV_PINN
from src.models.heston_pinn import Heston_PINN

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(CKPT_DIR, exist_ok=True)


def train_bsm():
    print("=" * 60)
    print("Training BSM-PINN")
    print("=" * 60)
    model = BSM_PINN(K=100, T=1.0, r=0.05, sigma=0.2, S_max=300)
    model.train(epochs=20000, log_every=2000)
    model.save(os.path.join(CKPT_DIR, "bsm_pinn.pt"))
    print("BSM checkpoint saved.\n")
    return model


def train_cev():
    print("=" * 60)
    print("Training CEV-PINN (beta=0.5)")
    print("=" * 60)
    model = CEV_PINN(K=100, T=1.0, r=0.05, sigma=0.25, beta=0.5, S_max=300)
    model.train(epochs=20000, log_every=2000)
    model.save(os.path.join(CKPT_DIR, "cev_pinn.pt"))
    print("CEV checkpoint saved.\n")
    return model


def train_heston():
    print("=" * 60)
    print("Training Heston-PINN")
    print("=" * 60)
    model = Heston_PINN(
        K=100, T=1.0, r=0.05,
        kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04,
        S_max=300, v_max=1.0,
    )
    model.train(epochs=30000, log_every=2000)
    model.save(os.path.join(CKPT_DIR, "heston_pinn.pt"))
    print("Heston checkpoint saved.\n")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bsm", "cev", "heston", "all"],
                        default="all")
    args = parser.parse_args()

    if args.model in ("bsm", "all"):
        train_bsm()
    if args.model in ("cev", "all"):
        train_cev()
    if args.model in ("heston", "all"):
        train_heston()
