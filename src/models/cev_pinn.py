"""
CEV-PINN: Physics-Informed Neural Network for Constant Elasticity of Variance
option pricing.

PDE: dV/dt + 0.5*sigma^2*S^(2*beta)*d2V/dS2 + r*S*dV/dS - r*V = 0
  beta=1  -> reduces to BSM
  beta=0.5 -> square-root process (most stable, used as default)
  beta<1  -> volatility decreases as S increases (leverage effect)

No closed-form solution for general beta; we validate against finite-difference
reference prices computed in this file.
"""

import torch
import torch.nn as nn
import numpy as np


# ── finite-difference reference (Crank-Nicolson) ───────────────────────────
def cev_fd_call(S0, K, T, r, sigma, beta, N_S=400, N_t=400):
    """Crank-Nicolson FD for CEV call price. Used as ground truth."""
    S_max = 3.0 * K
    dS = S_max / N_S
    dt = T / N_t
    S = np.linspace(0, S_max, N_S + 1)

    # terminal payoff
    V = np.maximum(S - K, 0.0)

    # CEV diffusion coefficient (fixed for all time steps)
    alpha = 0.5 * sigma ** 2 * S ** (2 * beta)

    # Crank-Nicolson left-side matrix coefficients (implicit half-step)
    a_diag = -0.5 * dt * (alpha / dS ** 2 - r * S / (2 * dS))
    b_diag = 1.0 + dt * (alpha / dS ** 2 + r)
    c_diag = -0.5 * dt * (alpha / dS ** 2 + r * S / (2 * dS))

    # right-side explicit half-step coefficients
    a_rhs = 0.5 * dt * (alpha / dS ** 2 - r * S / (2 * dS))
    b_rhs = 1.0 - dt * (alpha / dS ** 2 + r)
    c_rhs = 0.5 * dt * (alpha / dS ** 2 + r * S / (2 * dS))

    for step in range(N_t):
        # current time for boundary (backward in time: t = T - step*dt)
        tau = (step + 1) * dt   # time elapsed from T

        # build RHS from explicit half-step
        rhs = np.zeros_like(V)
        rhs[1:-1] = (a_rhs[1:-1] * V[:-2]
                     + b_rhs[1:-1] * V[1:-1]
                     + c_rhs[1:-1] * V[2:])
        rhs[0] = 0.0
        rhs[-1] = S_max - K * np.exp(-r * tau)

        # Thomas algorithm — copy coefficients fresh each step
        aa = a_diag.copy()
        bb = b_diag.copy()
        cc = c_diag.copy()
        n = len(rhs)

        # forward sweep
        for i in range(1, n):
            if abs(bb[i - 1]) < 1e-14:
                break
            m = aa[i] / bb[i - 1]
            bb[i] -= m * cc[i - 1]
            rhs[i] -= m * rhs[i - 1]

        # back substitution
        V_new = np.zeros(n)
        V_new[-1] = rhs[-1] / bb[-1]
        for i in range(n - 2, -1, -1):
            V_new[i] = (rhs[i] - cc[i] * V_new[i + 1]) / bb[i]
        V = V_new

    idx = int(S0 / dS)
    idx = min(idx, N_S - 1)
    w = (S0 - S[idx]) / dS
    return float((1 - w) * V[idx] + w * V[idx + 1])


# ── shared network (same gated architecture as BSM) ────────────────────────
class GatedPINN(nn.Module):
    def __init__(self, in_dim=2, hidden=64, depth=4):
        super().__init__()
        self.branch_shallow = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        deep_layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            deep_layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.branch_deep = nn.Sequential(*deep_layers)
        self.gate = nn.Linear(hidden, 1)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h_s = self.branch_shallow(x)
        h_d = self.branch_deep(x)
        g = torch.sigmoid(self.gate(h_s))
        return self.out(g * h_s + (1 - g) * h_d)


# ── PINN trainer ────────────────────────────────────────────────────────────
class CEV_PINN:
    def __init__(self, K=100.0, T=1.0, r=0.05, sigma=0.25, beta=0.5,
                 S_max=300.0, device=None):
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.beta = beta          # elasticity parameter
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
        t_lo = torch.FloatTensor(n, 1).uniform_(0.0, self.T)
        S_lo = torch.zeros(n, 1)
        V_lo = torch.zeros(n, 1)
        t_hi = torch.FloatTensor(n, 1).uniform_(0.0, self.T)
        S_hi = torch.full((n, 1), self.S_max)
        V_hi = self.S_max - self.K * torch.exp(-self.r * (self.T - t_hi))
        return (
            S_lo.to(self.device), t_lo.to(self.device), V_lo.to(self.device),
            S_hi.to(self.device), t_hi.to(self.device), V_hi.to(self.device),
        )

    def _pde_residual(self, S, t):
        S.requires_grad_(True)
        t.requires_grad_(True)
        x = torch.cat([S / self.S_max, t / self.T], dim=1)
        V = self.net(x)

        V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V),
                                  create_graph=True)[0]
        V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V),
                                  create_graph=True)[0]
        V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S),
                                   create_graph=True)[0]

        # CEV diffusion term: sigma^2 * S^(2*beta)
        diffusion = 0.5 * self.sigma ** 2 * S ** (2 * self.beta)
        residual = V_t + diffusion * V_SS + self.r * S * V_S - self.r * V
        return residual

    def _predict(self, S, t):
        x = torch.cat([S / self.S_max, t / self.T], dim=1)
        return self.net(x)

    def train(self, epochs=20000, log_every=1000,
              w_pde=1.0, w_ic=10.0, w_bc=5.0):
        from tqdm import tqdm
        losses = []
        pbar = tqdm(range(1, epochs + 1), desc=f"CEV β={self.beta}",
                    unit="epoch", dynamic_ncols=True)
        for epoch in pbar:
            self.optimizer.zero_grad()

            S_c, t_c = self._sample_collocation()
            loss_pde = torch.mean(self._pde_residual(S_c, t_c) ** 2)

            S_ic, t_ic, V_ic = self._sample_terminal()
            loss_ic = torch.mean((self._predict(S_ic, t_ic) - V_ic) ** 2)

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

    def price(self, S, t=0.0):
        self.net.eval()
        with torch.no_grad():
            S_t = torch.tensor([[S]], dtype=torch.float32).to(self.device)
            t_t = torch.tensor([[t]], dtype=torch.float32).to(self.device)
            return self._predict(S_t, t_t).item()

    def save(self, path):
        torch.save({
            "state_dict": self.net.state_dict(),
            "params": dict(K=self.K, T=self.T, r=self.r, sigma=self.sigma,
                           beta=self.beta, S_max=self.S_max),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
