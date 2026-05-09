"""
Heston-PINN: Physics-Informed Neural Network for Heston stochastic volatility
option pricing.

PDE (2D, inputs: S, v, t):
  dV/dt + 0.5*v*S^2*d2V/dS2 + rho*xi*v*S*d2V/dSdv + 0.5*xi^2*v*d2V/dv2
        + r*S*dV/dS + kappa*(theta-v)*dV/dv - r*V = 0

Parameters:
  kappa : mean-reversion speed
  theta : long-run variance
  xi    : vol-of-vol
  rho   : correlation between S and v Brownian motions
  r     : risk-free rate
  K, T  : strike and maturity

Boundary / terminal conditions:
  V(S, v, T) = max(S - K, 0)          terminal payoff
  V(0, v, t) = 0                       S=0 boundary
  V(S_max, v, t) = S_max - K*exp(-r*(T-t))   deep ITM
  dV/dv(S, 0, t) = 0                   Feller boundary (reflecting)
  V(S, v_max, t) = S                   large-v approximation

Architecture: ICPINN (hard-constraint) from Guo et al. 2024.
  The terminal condition is enforced exactly via output transformation:
    V_hat(S,v,t) = payoff(S) + (T-t) * net(S,v,t)
  so V_hat(S,v,T) = payoff(S) automatically.
"""

import torch
import torch.nn as nn
import numpy as np


# ── Heston semi-analytical price (characteristic function) ─────────────────
def heston_call_price(S0, K, T, r, kappa, theta, xi, rho, v0,
                      n_integration=200):
    """
    Heston (1993) semi-analytical call price via characteristic function.
    Uses the standard Heston formula with Gauss-Laguerre quadrature.
    """
    def char_func(phi, j):
        # j=1 or j=2 selects the two characteristic functions
        if j == 1:
            u, b = 0.5, kappa - rho * xi
        else:
            u, b = -0.5, kappa
        a = kappa * theta
        x = np.log(S0)
        d = np.sqrt((rho * xi * phi * 1j - b) ** 2
                    - xi ** 2 * (2 * u * phi * 1j - phi ** 2))
        g = (b - rho * xi * phi * 1j + d) / (b - rho * xi * phi * 1j - d)
        C = (r * phi * 1j * T
             + a / xi ** 2 * ((b - rho * xi * phi * 1j + d) * T
                               - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))))
        D = ((b - rho * xi * phi * 1j + d) / xi ** 2
             * (1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
        return np.exp(C + D * v0 + 1j * phi * x)

    def integrand(phi, j):
        return np.real(np.exp(-1j * phi * np.log(K))
                       * char_func(phi, j) / (1j * phi))

    phi_grid = np.linspace(1e-5, 200, n_integration)
    dphi = phi_grid[1] - phi_grid[0]
    P = np.zeros(2)
    for j in [1, 2]:
        P[j - 1] = 0.5 + (1 / np.pi) * np.trapz(
            [integrand(p, j) for p in phi_grid], phi_grid
        )
    return S0 * P[0] - K * np.exp(-r * T) * P[1]


# ── hard-constraint network (ICPINN) ───────────────────────────────────────
class HestonNet(nn.Module):
    """
    3-input (S_norm, v_norm, t_norm) network with hard terminal constraint.
    Output: V(S,v,t) = payoff(S) + (T-t)*net(S,v,t)
    Supports call and put payoffs.
    """

    def __init__(self, hidden=128, depth=6):
        super().__init__()
        layers = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, S_n, v_n, t_n, S_raw, t_raw, K, T, is_call=True):
        x = torch.cat([S_n, v_n, t_n], dim=1)
        raw = self.net(x)
        if is_call:
            payoff = torch.clamp(S_raw - K, min=0.0)
        else:
            payoff = torch.clamp(K - S_raw, min=0.0)
        return payoff + (T - t_raw) * raw


# ── PINN trainer ────────────────────────────────────────────────────────────
class Heston_PINN:
    OPTION_TYPES = {"european_call", "european_put", "american_call", "american_put"}

    def __init__(self, K=100.0, T=1.0, r=0.05,
                 kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04,
                 S_max=300.0, v_max=1.0, option_type="european_call", device=None):
        assert option_type in self.OPTION_TYPES
        self.K = K
        self.T = T
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
        self.S_max = S_max
        self.v_max = v_max
        self.option_type = option_type
        self.is_call = "call" in option_type
        self.is_american = "american" in option_type
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.net = HestonNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5000, gamma=0.5
        )

    def _to(self, t):
        return t.to(self.device)

    def _payoff(self, S):
        if self.is_call:
            return torch.clamp(S - self.K, min=0.0)
        else:
            return torch.clamp(self.K - S, min=0.0)

    def _forward(self, S, v, t):
        return self.net(
            S / self.S_max, v / self.v_max, t / self.T,
            S, t, self.K, self.T, self.is_call
        )

    def _sample_collocation(self, n=15000):
        S = self._to(torch.FloatTensor(n, 1).uniform_(0.01, self.S_max))
        v = self._to(torch.FloatTensor(n, 1).uniform_(1e-4, self.v_max))
        t = self._to(torch.FloatTensor(n, 1).uniform_(0.0, self.T * 0.999))
        return S, v, t

    def _sample_boundary(self, n=1000):
        # S=0 boundary
        t_s0 = self._to(torch.FloatTensor(n, 1).uniform_(0, self.T))
        v_s0 = self._to(torch.FloatTensor(n, 1).uniform_(1e-4, self.v_max))
        S_s0 = self._to(torch.zeros(n, 1))
        V_s0 = self._to(torch.full((n, 1), 0.0 if self.is_call
                                   else self.K))  # put: K at S=0

        # S=S_max boundary
        t_sm = self._to(torch.FloatTensor(n, 1).uniform_(0, self.T))
        v_sm = self._to(torch.FloatTensor(n, 1).uniform_(1e-4, self.v_max))
        S_sm = self._to(torch.full((n, 1), self.S_max))
        if self.is_call:
            if self.is_american:
                V_sm = self._to(torch.full((n, 1), self.S_max - self.K))
            else:
                V_sm = self.S_max - self.K * torch.exp(-self.r * (self.T - t_sm))
        else:
            V_sm = self._to(torch.zeros(n, 1))  # deep OTM put

        # v=v_max boundary
        t_vm = self._to(torch.FloatTensor(n, 1).uniform_(0, self.T))
        S_vm = self._to(torch.FloatTensor(n, 1).uniform_(0.01, self.S_max))
        v_vm = self._to(torch.full((n, 1), self.v_max))
        V_vm = S_vm if self.is_call else self._to(torch.zeros(n, 1))

        return (S_s0, v_s0, t_s0, V_s0,
                S_sm, v_sm, t_sm, V_sm,
                S_vm, v_vm, t_vm, V_vm)

    # ── PDE residual ─────────────────────────────────────────────────────────
    def _pde_residual(self, S, v, t):
        S.requires_grad_(True)
        v.requires_grad_(True)
        t.requires_grad_(True)
        V = self._forward(S, v, t)

        V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V),
                                  create_graph=True)[0]
        V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V),
                                  create_graph=True)[0]
        V_v = torch.autograd.grad(V, v, grad_outputs=torch.ones_like(V),
                                  create_graph=True)[0]
        V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S),
                                   create_graph=True)[0]
        V_vv = torch.autograd.grad(V_v, v, grad_outputs=torch.ones_like(V_v),
                                   create_graph=True)[0]
        V_Sv = torch.autograd.grad(V_S, v, grad_outputs=torch.ones_like(V_S),
                                   create_graph=True)[0]

        residual = (V_t
                    + 0.5 * v * S ** 2 * V_SS
                    + self.rho * self.xi * v * S * V_Sv
                    + 0.5 * self.xi ** 2 * v * V_vv
                    + self.r * S * V_S
                    + self.kappa * (self.theta - v) * V_v
                    - self.r * V)
        return residual

    def train(self, epochs=30000, log_every=1000,
              w_pde=1.0, w_bc=5.0, w_american=50.0):
        from tqdm import tqdm
        losses = []
        pbar = tqdm(range(1, epochs + 1),
                    desc=f"Heston-{self.option_type}", unit="epoch",
                    dynamic_ncols=True)
        for epoch in pbar:
            self.optimizer.zero_grad()

            S_c, v_c, t_c = self._sample_collocation()
            res = self._pde_residual(S_c, v_c, t_c)
            loss_pde = torch.mean(res ** 2)

            (S_s0, v_s0, t_s0, V_s0,
             S_sm, v_sm, t_sm, V_sm,
             S_vm, v_vm, t_vm, V_vm) = self._sample_boundary()

            loss_bc = (
                torch.mean((self._forward(S_s0, v_s0, t_s0) - V_s0) ** 2)
                + torch.mean((self._forward(S_sm, v_sm, t_sm) - V_sm) ** 2)
                + torch.mean((self._forward(S_vm, v_vm, t_vm) - V_vm) ** 2)
            )

            loss = w_pde * loss_pde + w_bc * loss_bc

            if self.is_american:
                S_a, v_a, t_a = self._sample_collocation(n=5000)
                V_pred = self._forward(S_a, v_a, t_a)
                loss_american = torch.mean(
                    torch.clamp(self._payoff(S_a) - V_pred, min=0.0) ** 2
                )
                loss = loss + w_american * loss_american

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.3e}",
                             pde=f"{loss_pde.item():.3e}",
                             bc=f"{loss_bc.item():.3e}")
            if epoch % log_every == 0:
                losses.append(loss.item())
        return losses

    def price(self, S, v=None, t=0.0):
        """Price a European call. v defaults to v0 if not given."""
        if v is None:
            v = self.v0
        self.net.eval()
        with torch.no_grad():
            S_t = self._to(torch.tensor([[S]], dtype=torch.float32))
            v_t = self._to(torch.tensor([[v]], dtype=torch.float32))
            t_t = self._to(torch.tensor([[t]], dtype=torch.float32))
            return self._forward(S_t, v_t, t_t).item()

    def save(self, path):
        torch.save({
            "state_dict": self.net.state_dict(),
            "params": dict(K=self.K, T=self.T, r=self.r, kappa=self.kappa,
                           theta=self.theta, xi=self.xi, rho=self.rho,
                           v0=self.v0, S_max=self.S_max, v_max=self.v_max,
                           option_type=self.option_type),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
