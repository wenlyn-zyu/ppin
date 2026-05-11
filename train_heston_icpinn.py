"""
train_heston_icpinn.py — Train Heston ICPINN with Tan & Zhang (2026) paper params.

Paper params (Section 3.2):
  K=1 (we use K=100 with scale-invariant BCs), r=0.1
  kappa=1, theta=0.08, xi=0.39, rho=-0.93, v0=0.04
  S_max=4*K=400, v_max=1
  N_r=10000 interior, N_bc=500 each boundary
  Pretrain aux: 10000 epochs
  Main train: 20000 epochs, Adam lr=1e-3, StepLR gamma=0.75 step=5000

Usage:
  python train_heston_icpinn.py [--epochs 20000] [--out results/heston_icpinn.pt]
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.models.heston_pinn import Heston_PINN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=20000)
    parser.add_argument("--pretrain_epochs", type=int,   default=10000)
    parser.add_argument("--n_r",             type=int,   default=10000)
    parser.add_argument("--n_bc",            type=int,   default=500)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--w_data",          type=float, default=100.0)
    parser.add_argument("--out",             type=str,   default="results/heston_icpinn.pt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Paper params: K=100 (scale-invariant from K=1 paper), r=0.1
    model = Heston_PINN(
        K=100.0, T=1.0, r=0.1,
        kappa=1.0, theta=0.08, xi=0.39, rho=-0.93, v0=0.04,
        S_max=400.0, v_max=1.0,
        device=device,
    )

    print(f"\nTraining Heston ICPINN (Tan & Zhang 2026 params)")
    print(f"  K=100, T=1, r=0.1, kappa=1, theta=0.08, xi=0.39, rho=-0.93, v0=0.04")
    print(f"  S_max=400, v_max=1")
    print(f"  pretrain_epochs={args.pretrain_epochs}, epochs={args.epochs}")
    print(f"  n_r={args.n_r}, n_bc={args.n_bc}, lr={args.lr}\n")

    model.train(
        epochs=args.epochs,
        pretrain_epochs=args.pretrain_epochs,
        n_r=args.n_r,
        n_bc=args.n_bc,
        lr=args.lr,
        w_data=args.w_data,
        log_every=2000,
    )

    model.save(args.out)
    print(f"\nCheckpoint saved to {args.out}")


if __name__ == "__main__":
    main()
