"""
Heston-PINN v2: improved convergence via log-price transform,
adaptive loss weighting, and residual-based adaptive sampling (RAR).

Key changes from v1:
  1. Log-price transform: x = ln(S/K), domain x in [-4.6, 1.1] instead of S in [0,300]
     Eliminates S^2 scale imbalance in PDE diffusion term.
  2. Adaptive loss weights: updated every `reweight_every` steps using
     gradient-norm balancing (Wang et al. 2022, NTK-based).
  3. RAR sampling: every `rar_every` steps, add collocation points in
     high-residual regions (deep OTM where V ~ 0).
  4. Deeper network (8 layers x 128) + Fourier feature embedding for
     better approximation of near-zero payoff region.
  5. Feller boundary at v=0: dV/dv = 0 enforced as soft constraint.

PDE in log-price coordinates (x = ln(S), S = e^x):
  dV/dt + 0.5*v*d2V/dx2 + (r - 0.5*v)*dV/dx
        + rho*xi*v*d2V/dxdv + 0.5*xi^2*v*d2V/dv2
        + kappa*(theta-v)*dV/dv - r*V = 0

References:
  Hainaut & Casas (2024) Annals of Finance
  Guo et al. (2024) ICPINN
  Wang et al. (2022) NTK-based adaptive weights
  Lu et al. (2021) RAR sampling
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import quad


# -- Heston semi-analytical price (characteristic function) -----------------
def heston_call_price(S0, K, T, r, kappa, theta, xi, rho, v0,
                      n_integration=500):
    def char_func(phi, j):
        if j == 1:
            u, b = 0.5, kappa - rho * xi
        else:
            u, b = -0.5, kappa
        d = np.sqrt((rho * xi * phi * 1j - b) ** 2
                    - xi ** 2 * (2 * u * phi * 1j - phi ** 2))
        g = (b - rho * xi * phi * 1j + d) / (b - rho * xi * phi * 1j - d)
        C = (r * phi * 1j * T
             + kappa * theta / xi ** 2
             * ((b - rho * xi * phi * 1j + d) * T
                - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))))
        D = ((b - rho * xi * phi * 1j + d) / xi ** 2
             * (1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
        return np.exp(C + D * v0 + 1j * phi * np.log(S0))

    def integrand(phi, j):
        return np.real(np.exp(-1j * phi * np.log(K))
                       * char_func(phi, j) / (1j * phi))

    P = np.zeros(2)
    for j in [1, 2]:
        P[j - 1] = 0.5 + (1 / np.pi) * quad(
            integrand, 1e-6, 200, args=(j,), limit=500)[0]
    return S0 * P[0] - K * np.exp(-r * T) * P[1]


# -- Fourier feature embedding -----------------------------------------------
class FourierEmbedding(nn.Module):
    """Random Fourier features to help learn high-frequency components."""
    def __init__(self, in_dim, n_freq=32, scale=1.0):
        super().__init__()
        B = torch.randn(in_dim, n_freq) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# -- ICPINN network with log-price input ------------------------------------
class HestonNetV2(nn.Module):
    """
    Inputs: (x_norm, v_norm, t_norm) where x = ln(S/K)
    Hard terminal constraint: V = payoff(S) + (T-t)*net(...)
    """
    def __init__(self, hidden=128, depth=8, n_freq=16):
        super().__init__()
        self.embed = FourierEmbedding(3, n_freq=n_freq, scale=1.0)
        in_dim = 2 * n_freq
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_n, v_n, t_n, S_raw, t_raw, K, T):
        inp = torch.cat([x_n, v_n, t_n], dim=1)
        emb = self.embed(inp)
        raw = self.net(emb)
        payoff = torch.clamp(S_raw - K, min=0.0)
        # hard constraint: V(S,v,T) = payoff(S) exactly
        return payoff + (T - t_raw) * raw


# -- PINN trainer ------------------------------------------------------------
class Heston_PINN:
    def __init__(self, K=100.0, T=1.0, r=0.05,
                 kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04,
                 S_max=300.0, v_max=1.0, device=None):
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
        # log-price domain
        self.x_min = np.log(1.0 / K)
        self.x_max = np.log(S_max / K)
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.net = HestonNetV2().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50000, eta_min=1e-5
        )
        self.w_pde = 1.0
        self.w_bc = 10.0

    def _to(self, t):
        return t.to(self.device)

    def _forward_x(self, x, v, t):
        """Forward pass using log-price x = ln(S/K)."""
        S = torch.exp(x) * self.K
        x_n = (x - self.x_min) / (self.x_max - self.x_min)
        v_n = v / self.v_max
        t_n = t / self.T
        return self.net(x_n, v_n, t_n, S, t, self.K, self.T)

    def _forward(self, S, v, t):
        """Forward pass using raw S (for evaluate.py compatibility)."""
        S = torch.clamp(S, min=1e-3)
        x = torch.log(S / self.K)
        return self._forward_x(x, v, t)

    def _sample_collocation(self, n=15000):
        x = self._to(torch.FloatTensor(n, 1).uniform_(self.x_min, self.x_max))
        v = self._to(torch.FloatTensor(n, 1).uniform_(1e-4, self.v_max))
        t = self._to(torch.FloatTensor(n, 1).uniform_(0.0, self.T * 0.999))
        return x, v, t

    def _sample_boundary(self, n=1000):
        # S=0 boundary (x=x_min): V=0
        t_s0 = self._to(torch.FloatTensor(n, 1).uniform_(0, self.T))
        v_s0 = self._to(torch.FloatTensor(n, 1).uniform_(1e-4, self.v_max))
        x_s0 = self._to(torch.full((n, 1), self.x_min))
        V_s0 = self._to(torch.zeros(n, 1))

        # S=S_max boundary (x=x_max): V = S - K*exp(-r*(T-t))
        t_sm = self._to(torch.FloatTensor(n, 1).uniform_(0, self.T))
        v_sm = self._to(torch.FloatTensor(n, 1).uniform_(1e-4, self.v_max))
        x_sm = self._to(torch.full((n, 1), self.x_max))
        V_sm = self.S_max - self.K * torch.exp(-self.r * (self.T - t_sm))

        # v=0 Feller boundary points (for dV/dv=0 penalty)
        t_v0 = self._to(torch.FloatTensor(n, 1).uniform_(0, self.T))
        x_v0 = self._to(torch.FloatTensor(n, 1).uniform_(self.x_min, self.x_max))
        v_v0 = self._to(torch.full((n, 1), 1e-4))

        # NOTE: v=v_max boundary (V~S) is dropped — it is a rough approximation
        # with MSE magnitude ~S_max^2 that dominates and destabilises training.
        # The PDE naturally handles large-v behaviour via the diffusion term.

        return (x_s0, v_s0, t_s0, V_s0,
                x_sm, v_sm, t_sm, V_sm,
                x_v0, v_v0, t_v0)

    def _pde_residual(self, x, v, t):
        """PDE residual in log-price coordinates."""
        x.requires_grad_(True)
        v.requires_grad_(True)
        t.requires_grad_(True)
        V = self._forward_x(x, v, t)

        ones = torch.ones_like(V)
        V_t  = torch.autograd.grad(V, t,  grad_outputs=ones, create_graph=True)[0]
        V_x  = torch.autograd.grad(V, x,  grad_outputs=ones, create_graph=True)[0]
        V_v  = torch.autograd.grad(V, v,  grad_outputs=ones, create_graph=True)[0]
        V_xx = torch.autograd.grad(V_x, x, grad_outputs=torch.ones_like(V_x), create_graph=True)[0]
        V_vv = torch.autograd.grad(V_v, v, grad_outputs=torch.ones_like(V_v), create_graph=True)[0]
        V_xv = torch.autograd.grad(V_x, v, grad_outputs=torch.ones_like(V_x), create_graph=True)[0]

        residual = (V_t
                    + 0.5 * v * V_xx
                    + (self.r - 0.5 * v) * V_x
                    + self.rho * self.xi * v * V_xv
                    + 0.5 * self.xi ** 2 * v * V_vv
                    + self.kappa * (self.theta - v) * V_v
                    - self.r * V)
        return residual

    def _rar_resample(self, x_c, v_c, t_c, n_add=3000):
        """Add collocation points in high-residual regions (RAR)."""
        x_c2 = x_c.detach().requires_grad_(True)
        v_c2 = v_c.detach().requires_grad_(True)
        t_c2 = t_c.detach().requires_grad_(True)
        res = self._pde_residual(x_c2, v_c2, t_c2).detach().abs().squeeze()
        probs = res / (res.sum() + 1e-10)
        idx = torch.multinomial(probs, n_add, replacement=True)
        x_new = torch.cat([x_c, x_c[idx].detach()], dim=0)
        v_new = torch.cat([v_c, v_c[idx].detach()], dim=0)
        t_new = torch.cat([t_c, t_c[idx].detach()], dim=0)
        return x_new, v_new, t_new

    def _update_weights(self, loss_pde, loss_bc):
        """Gradient-norm adaptive weighting (Wang et al. 2022)."""
        def grad_norm(loss):
            grads = torch.autograd.grad(
                loss, list(self.net.parameters()),
                retain_graph=True, allow_unused=True)
            norms = [g.norm() for g in grads if g is not None]
            return torch.stack(norms).mean().item() if norms else 1.0

        n_pde = grad_norm(loss_pde)
        n_bc  = grad_norm(loss_bc)
        total = n_pde + n_bc + 1e-10
        self.w_pde = float(np.clip(total / (n_pde + 1e-10), 0.1, 50.0))
        self.w_bc  = float(np.clip(total / (n_bc  + 1e-10), 0.1, 50.0))

    def train(self, epochs=50000, log_every=2000,
              rar_every=2000, reweight_every=1000):
        losses = []
        x_c, v_c, t_c = self._sample_collocation()

        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()

            x_c2 = x_c.detach().requires_grad_(True)
            v_c2 = v_c.detach().requires_grad_(True)
            t_c2 = t_c.detach().requires_grad_(True)
            res = self._pde_residual(x_c2, v_c2, t_c2)
            loss_pde = torch.mean(res ** 2)

            (x_s0, v_s0, t_s0, V_s0,
             x_sm, v_sm, t_sm, V_sm,
             x_v0, v_v0, t_v0) = self._sample_boundary()

            loss_bc = (
                torch.mean((self._forward_x(x_s0, v_s0, t_s0) - V_s0) ** 2)
                # normalise S_max boundary by S_max^2 to match scale of V~0 boundary
                + torch.mean(((self._forward_x(x_sm, v_sm, t_sm) - V_sm) / self.S_max) ** 2)
            )

            # Feller boundary: dV/dv = 0 at v=0
            x_v0r = x_v0.detach().requires_grad_(True)
            v_v0r = v_v0.detach().requires_grad_(True)
            t_v0r = t_v0.detach().requires_grad_(True)
            V_v0 = self._forward_x(x_v0r, v_v0r, t_v0r)
            V_dv = torch.autograd.grad(V_v0, v_v0r,
                                       grad_outputs=torch.ones_like(V_v0),
                                       create_graph=True)[0]
            loss_feller = torch.mean(V_dv ** 2)

            if epoch % reweight_every == 0:
                self._update_weights(loss_pde, loss_bc)

            loss = (self.w_pde * loss_pde
                    + self.w_bc * loss_bc
                    + 2.0 * loss_feller)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            if epoch % rar_every == 0:
                x_c, v_c, t_c = self._rar_resample(
                    x_c.detach(), v_c.detach(), t_c.detach(), n_add=3000)
                if x_c.shape[0] > 30000:
                    idx = torch.randperm(x_c.shape[0])[:20000]
                    x_c, v_c, t_c = x_c[idx], v_c[idx], t_c[idx]

            if epoch % log_every == 0:
                print(f"[Heston v2] epoch {epoch:6d} | "
                      f"loss={loss.item():.4e}  "
                      f"pde={loss_pde.item():.4e}  "
                      f"bc={loss_bc.item():.4e}  "
                      f"w_pde={self.w_pde:.2f}  w_bc={self.w_bc:.2f}")
                losses.append(loss.item())
        return losses

    def price(self, S, v=None, t=0.0):
        if v is None:
            v = self.v0
        self.net.eval()
        with torch.no_grad():
            S_t = self._to(torch.tensor([[float(S)]], dtype=torch.float32))
            v_t = self._to(torch.tensor([[float(v)]], dtype=torch.float32))
            t_t = self._to(torch.tensor([[float(t)]], dtype=torch.float32))
            return self._forward(S_t, v_t, t_t).item()

    def save(self, path):
        torch.save({
            "state_dict": self.net.state_dict(),
            "params": dict(K=self.K, T=self.T, r=self.r, kappa=self.kappa,
                           theta=self.theta, xi=self.xi, rho=self.rho,
                           v0=self.v0, S_max=self.S_max, v_max=self.v_max),
            "weights": dict(w_pde=self.w_pde, w_bc=self.w_bc),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
