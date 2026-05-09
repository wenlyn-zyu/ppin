"""
BSM-PINN: Physics-Informed Neural Network for Black-Scholes-Merton option pricing.

PDE: dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V = 0
Terminal condition (call): V(S, T) = max(S - K, 0)
Boundary conditions:
    V(0, t) = 0
    V(S_max, t) = S_max - K*exp(-r*(T-t))  (deep ITM approximation)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm


# ── analytical Black-Scholes price for validation ──────────────────────────
def bs_call_price(S, K, T, r, sigma):
    S, K, T, r, sigma = map(float, [S, K, T, r, sigma])
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ── network architecture (gated, from Dhiman & Hu 2023) ────────────────────
class GatedPINN(nn.Module):
    """Two-branch gated network: wide shallow branch + deep branch."""

    def __init__(self, hidden=64, depth=4):
        super().__init__()
        # shallow branch: 1 hidden layer, captures simple features
        self.branch_shallow = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # deep branch: `depth` hidden layers, captures complex features
        deep_layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            deep_layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.branch_deep = nn.Sequential(*deep_layers)

        self.gate = nn.Linear(hidden, 1)          # scalar gate weight
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h_s = self.branch_shallow(x)
        h_d = self.branch_deep(x)
        g = torch.sigmoid(self.gate(h_s))         # gate in [0,1]
        h = g * h_s + (1 - g) * h_d
        return self.out(h)


# ── PINN trainer ────────────────────────────────────────────────────────────
class BSM_PINN:
    def __init__(self, K=100.0, T=1.0, r=0.05, sigma=0.2,
                 S_max=300.0, device=None):
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.S_max = S_max
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.net = GatedPINN().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5000, gamma=0.5
        )

    # ── sampling helpers ────────────────────────────────────────────────────
    def _sample_collocation(self, n=10000):
        S = torch.FloatTensor(n, 1).uniform_(0.01, self.S_max)
        t = torch.FloatTensor(n, 1).uniform_(0.0, self.T)
        return S.to(self.device), t.to(self.device)

    def _sample_terminal(self, n=2000):
        S = torch.FloatTensor(n, 1).uniform_(0.01, self.S_max)
        t = torch.full((n, 1), self.T)
        V = torch.clamp(S - self.K, min=0.0)
        return S.to(self.device), t.to(self.device), V.to(self.device)

    def _sample_boundary(self, n=1000):
        # lower boundary: S=0 -> V=0
        t_lo = torch.FloatTensor(n, 1).uniform_(0.0, self.T)
        S_lo = torch.zeros(n, 1)
        V_lo = torch.zeros(n, 1)
        # upper boundary: S=S_max -> V ≈ S_max - K*exp(-r*(T-t))
        t_hi = torch.FloatTensor(n, 1).uniform_(0.0, self.T)
        S_hi = torch.full((n, 1), self.S_max)
        V_hi = self.S_max - self.K * torch.exp(-self.r * (self.T - t_hi))
        return (
            S_lo.to(self.device), t_lo.to(self.device), V_lo.to(self.device),
            S_hi.to(self.device), t_hi.to(self.device), V_hi.to(self.device),
        )

    # ── PDE residual ────────────────────────────────────────────────────────
    def _pde_residual(self, S, t):
        S.requires_grad_(True)
        t.requires_grad_(True)
        x = torch.cat([S / self.S_max, t / self.T], dim=1)  # normalise inputs
        V = self.net(x)

        V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V),
                                  create_graph=True)[0]
        V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V),
                                  create_graph=True)[0]
        V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S),
                                   create_graph=True)[0]

        residual = (V_t
                    + 0.5 * self.sigma ** 2 * S ** 2 * V_SS
                    + self.r * S * V_S
                    - self.r * V)
        return residual

    # ── forward pass (normalised inputs) ────────────────────────────────────
    def _predict(self, S, t):
        x = torch.cat([S / self.S_max, t / self.T], dim=1)
        return self.net(x)

    # ── training loop ────────────────────────────────────────────────────────
    def train(self, epochs=20000, log_every=1000,
              w_pde=1.0, w_ic=10.0, w_bc=5.0):
        from tqdm import tqdm
        losses = []
        pbar = tqdm(range(1, epochs + 1), desc="BSM", unit="epoch",
                    dynamic_ncols=True)
        for epoch in pbar:
            self.optimizer.zero_grad()

            S_c, t_c = self._sample_collocation()
            res = self._pde_residual(S_c, t_c)
            loss_pde = torch.mean(res ** 2)

            S_ic, t_ic, V_ic = self._sample_terminal()
            V_pred_ic = self._predict(S_ic, t_ic)
            loss_ic = torch.mean((V_pred_ic - V_ic) ** 2)

            S_lo, t_lo, V_lo, S_hi, t_hi, V_hi = self._sample_boundary()
            loss_bc = (torch.mean((self._predict(S_lo, t_lo) - V_lo) ** 2)
                       + torch.mean((self._predict(S_hi, t_hi) - V_hi) ** 2))

            loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.3e}",
                             pde=f"{loss_pde.item():.3e}",
                             ic=f"{loss_ic.item():.3e}")
            if epoch % log_every == 0:
                losses.append(loss.item())

        return losses

    # ── inference ────────────────────────────────────────────────────────────
    def price(self, S, t=0.0):
        """Price a European call at spot S and current time t."""
        self.net.eval()
        with torch.no_grad():
            S_t = torch.tensor([[S]], dtype=torch.float32).to(self.device)
            t_t = torch.tensor([[t]], dtype=torch.float32).to(self.device)
            return self._predict(S_t, t_t).item()

    def save(self, path):
        torch.save({
            "state_dict": self.net.state_dict(),
            "params": dict(K=self.K, T=self.T, r=self.r,
                           sigma=self.sigma, S_max=self.S_max),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
