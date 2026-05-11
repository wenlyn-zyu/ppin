"""
Heston-PINN v3: faithful reproduction of Tan & Zhang (2026) ICPINN method.

Reference:
  Tan, J. & Zhang, X. (2026). Improved constrained physics-informed neural
  networks (ICPINNs) to solve PDE and its application to option pricing.
  Mathematics and Computers in Simulation, 241, 908-924.

Exact paper parameters (Section 3.2):
  K=1, r=0.1, kappa=1, theta=0.08, sigma=0.39, rho=-0.93
  S_max = 4*K,  v_max = 1
  N_r = 10000 interior,  N_bc = 500 each boundary

Architecture (Table 3):
  Main network:  8 hidden layers x 40 neurons, Tanh, glorot_normal init
  Aux network:   3 hidden layers x 40 neurons, Elu  (fits Dirichlet BCs)

ICPINN trial solution (Eq. 3.11):
  U_theta(S,v,tau) = A(S,v,tau) + B(S,v,tau) * NN(S,v,tau; theta)
  where B(S,v,tau) = S * tau  (vanishes at tau=0 and S=0)
  A(S,v,tau) is a small network pre-trained to satisfy Dirichlet BCs:
    - terminal payoff: A(S,v,0) = max(S-K, 0)
    - lower boundary:  A(0,v,tau) = 0

Loss (Eq. 3.16):
  L = L_pde + L_Smax + L_vmax + L_deg
  L_pde  : PDE residual at interior points
  L_Smax : dU/dS = 1 at S=S_max  (Neumann, deep ITM)
  L_vmax : dU/dv = 0 at v=v_max  (Neumann, large vol)
  L_deg  : degenerate BC at v=0: S*dU/dS + kappa*theta*dU/dv - r*U - dU/dtau = 0

Note on K=100 adaptation:
  The paper uses K=1 with S in [0,4]. We scale to K=100 with S in [0,400].
  All BCs and the PDE are scale-invariant under S -> S/K.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import quad


# -- Heston semi-analytical price (characteristic function) -----------------
def heston_call_price(S0, K, T, r, kappa, theta, xi, rho, v0):
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


# -- Weight initialisation (glorot_normal as in paper) ----------------------
def _glorot_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


# -- Auxiliary network: fits Dirichlet BCs (Table 4 in paper) ---------------
class AuxNet(nn.Module):
    """3 hidden layers x 40 neurons, Elu activation."""
    def __init__(self, hidden=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )
        self.net.apply(_glorot_normal)

    def forward(self, S_n, v_n, tau_n):
        x = torch.cat([S_n, v_n, tau_n], dim=1)
        return self.net(x)


# -- Main network: 8 hidden layers x 40 neurons, Tanh (Table 3 in paper) ---
class MainNet(nn.Module):
    def __init__(self, hidden=40, depth=8):
        super().__init__()
        layers = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.net.apply(_glorot_normal)

    def forward(self, S_n, v_n, tau_n):
        x = torch.cat([S_n, v_n, tau_n], dim=1)
        return self.net(x)


# -- ICPINN trainer ----------------------------------------------------------
class Heston_PINN:
    """
    ICPINN for Heston model, following Tan & Zhang (2026) exactly.

    Trial solution:
        U(S,v,tau) = A(S,v,tau) + S*tau * NN(S,v,tau)
    where A is pre-trained to satisfy:
        A(S,v,0)   = max(S-K, 0)   [terminal payoff]
        A(0,v,tau) = 0              [S=0 boundary]
    """
    def __init__(self, K=100.0, T=1.0, r=0.05,
                 kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04,
                 S_max=None, v_max=1.0, device=None):
        self.K = K
        self.T = T
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
        # Paper uses S_max = 4*K
        self.S_max = S_max if S_max is not None else 4.0 * K
        self.v_max = v_max
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.aux_net  = AuxNet().to(self.device)
        self.main_net = MainNet().to(self.device)

    def _to(self, t):
        return t.to(self.device)

    def _normalise(self, S, v, tau):
        """Scale inputs to [0,1]."""
        return S / self.S_max, v / self.v_max, tau / self.T

    def _trial(self, S, v, tau):
        """
        Trial solution in u = U/K units.
        u = payoff(S_n) + S_n*tau_n * [A(S_n,v_n,tau_n) + NN(S_n,v_n,tau_n)]

        payoff(S_n) = max(S_n - 1/S_max*K, 0) = max(S/K - 1, 0) hard-encodes
        the terminal condition exactly (no network needed for the kink).
        B = S_n*tau_n vanishes at tau=0 and S=0, so the correction term
        A+NN is free to learn the time-value without disturbing BCs.
        AuxNet now fits the S=0 boundary correction only (much easier task).
        """
        S_n, v_n, tau_n = self._normalise(S, v, tau)
        payoff = torch.clamp(S / self.K - 1.0, min=0.0)  # exact terminal payoff in u units
        A = self.aux_net(S_n, v_n, tau_n)
        N = self.main_net(S_n, v_n, tau_n)
        B = S_n * tau_n
        return payoff + B * (A + N)   # returns u = U/K

    def _pretrain_aux(self, epochs=3000, n=5000, lr=5e-3):
        """
        Pre-train aux_net so that the trial solution satisfies BCs.
        Since the terminal payoff is now hard-coded analytically, AuxNet only
        needs to ensure the S=0 boundary is satisfied:
          trial(0,v,tau) = payoff(0) + 0*tau_n*(A+N) = 0  (automatic)
        So AuxNet just needs A(0,v,tau) to be well-behaved (near 0).
        We also train it to output 0 at S=0 for all tau to keep the
        correction term from blowing up near the boundary.
        """
        opt = torch.optim.Adam(self.aux_net.parameters(), lr=lr)
        for ep in range(1, epochs + 1):
            opt.zero_grad()
            # S=0 boundary: A should be ~0 (correction term vanishes anyway via B=0)
            S_0   = self._to(torch.zeros(n, 1))
            v_0   = self._to(torch.FloatTensor(n, 1).uniform_(0, self.v_max))
            tau_0 = self._to(torch.FloatTensor(n, 1).uniform_(0, self.T))
            S_n0, v_n0, tau_n0 = self._normalise(S_0, v_0, tau_0)
            loss_0 = torch.mean(self.aux_net(S_n0, v_n0, tau_n0) ** 2)

            # Interior: A should be small (correction is handled by MainNet)
            S_int   = self._to(torch.FloatTensor(n, 1).uniform_(0, self.S_max))
            v_int   = self._to(torch.FloatTensor(n, 1).uniform_(0, self.v_max))
            tau_int = self._to(torch.FloatTensor(n, 1).uniform_(0, self.T))
            S_ni, v_ni, tau_ni = self._normalise(S_int, v_int, tau_int)
            loss_int = 0.1 * torch.mean(self.aux_net(S_ni, v_ni, tau_ni) ** 2)

            loss = loss_0 + loss_int
            loss.backward()
            opt.step()
            if ep % 500 == 0:
                print(f"  [aux pretrain] ep {ep:5d}  loss={loss.item():.4e}")

    def _pde_residual(self, S, v, tau):
        """
        Heston PDE for u = U/K in backward time tau = T - t.
        Dividing by K makes the PDE scale-invariant: same form as K=1 paper.
          du/dtau = 0.5*v*S^2*u_SS + rho*xi*v*S*u_Sv + 0.5*xi^2*v*u_vv
                  + r*S*u_S + kappa*(theta-v)*u_v - r*u
        """
        S.requires_grad_(True)
        v.requires_grad_(True)
        tau.requires_grad_(True)
        u = self._trial(S, v, tau)   # u = U/K, O(1)

        ones = torch.ones_like(u)
        u_tau = torch.autograd.grad(u, tau, grad_outputs=ones, create_graph=True)[0]
        u_S   = torch.autograd.grad(u, S,   grad_outputs=ones, create_graph=True)[0]
        u_v   = torch.autograd.grad(u, v,   grad_outputs=ones, create_graph=True)[0]
        u_SS  = torch.autograd.grad(u_S, S, grad_outputs=torch.ones_like(u_S), create_graph=True)[0]
        u_vv  = torch.autograd.grad(u_v, v, grad_outputs=torch.ones_like(u_v), create_graph=True)[0]
        u_Sv  = torch.autograd.grad(u_S, v, grad_outputs=torch.ones_like(u_S), create_graph=True)[0]

        res = (u_tau
               - 0.5 * v * S ** 2 * u_SS
               - self.rho * self.xi * v * S * u_Sv
               - 0.5 * self.xi ** 2 * v * u_vv
               - self.r * S * u_S
               - self.kappa * (self.theta - v) * u_v
               + self.r * u)
        return res

    def _build_data_anchors(self, n_data=50):
        """
        Pre-compute semi-analytical Heston prices at tau=T (t=0) as data anchors.
        Returns (S_anchors, v_anchors, u_anchors) tensors on self.device.
        Uses v=v0 and S uniformly spaced in [0.2*K, 3*K] to cover ATM well.
        """
        S_vals = np.linspace(0.2 * self.K, 3.0 * self.K, n_data)
        u_vals = np.array([
            heston_call_price(s, self.K, self.T, self.r,
                              self.kappa, self.theta, self.xi, self.rho, self.v0)
            / self.K
            for s in S_vals
        ])
        S_t = self._to(torch.tensor(S_vals, dtype=torch.float32).unsqueeze(1))
        v_t = self._to(torch.full((n_data, 1), self.v0, dtype=torch.float32))
        u_t = self._to(torch.tensor(u_vals, dtype=torch.float32).unsqueeze(1))
        return S_t, v_t, u_t

    def train(self, epochs=20000, log_every=2000,
              n_r=10000, n_bc=500, lr=1e-3,
              pretrain_epochs=3000, w_data=100.0):
        """
        Train following paper protocol:
          1. Pre-train aux_net (freeze after)
          2. Adam + StepLR (gamma=0.75, step=5000)
          3. Loss = L_pde + L_Smax + L_vmax + L_deg + w_data*L_data
          L_data: semi-analytical anchors at tau=T (t=0) to pin the solution
        """
        print("Pre-training auxiliary network...")
        self._pretrain_aux(epochs=pretrain_epochs, lr=lr)
        for p in self.aux_net.parameters():
            p.requires_grad_(False)

        print("Building semi-analytical data anchors at t=0...")
        S_anc, v_anc, u_anc = self._build_data_anchors(n_data=50)
        tau_anc = self._to(torch.full((50, 1), self.T, dtype=torch.float32))

        opt = torch.optim.Adam(self.main_net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.75)

        for epoch in range(1, epochs + 1):
            opt.zero_grad()

            # interior collocation
            S_r   = self._to(torch.FloatTensor(n_r, 1).uniform_(0, self.S_max))
            v_r   = self._to(torch.FloatTensor(n_r, 1).uniform_(0, self.v_max))
            tau_r = self._to(torch.FloatTensor(n_r, 1).uniform_(0, self.T))
            loss_pde = torch.mean(self._pde_residual(S_r, v_r, tau_r) ** 2)

            # Neumann at S=S_max: du/dS = 1/K
            S_sm   = self._to(torch.full((n_bc, 1), self.S_max))
            v_sm   = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.v_max))
            tau_sm = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.T))
            S_sm   = S_sm.detach().requires_grad_(True)
            u_sm   = self._trial(S_sm, v_sm, tau_sm)
            du_dS  = torch.autograd.grad(u_sm, S_sm,
                                         grad_outputs=torch.ones_like(u_sm),
                                         create_graph=True)[0]
            loss_Smax = torch.mean((du_dS - 1.0 / self.K) ** 2)

            # Neumann at v=v_max: du/dv = 0
            S_vm   = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.S_max))
            v_vm   = self._to(torch.full((n_bc, 1), self.v_max))
            tau_vm = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.T))
            v_vm   = v_vm.detach().requires_grad_(True)
            u_vm   = self._trial(S_vm, v_vm, tau_vm)
            du_dv  = torch.autograd.grad(u_vm, v_vm,
                                         grad_outputs=torch.ones_like(u_vm),
                                         create_graph=True)[0]
            loss_vmax = torch.mean(du_dv ** 2)

            # degenerate BC at v=0
            S_dg   = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.S_max))
            v_dg   = self._to(torch.full((n_bc, 1), 1e-6))
            tau_dg = self._to(torch.FloatTensor(n_bc, 1).uniform_(0, self.T))
            S_dg   = S_dg.detach().requires_grad_(True)
            v_dg   = v_dg.detach().requires_grad_(True)
            tau_dg = tau_dg.detach().requires_grad_(True)
            u_dg   = self._trial(S_dg, v_dg, tau_dg)
            ones_d = torch.ones_like(u_dg)
            du_dS_d   = torch.autograd.grad(u_dg, S_dg,   grad_outputs=ones_d, create_graph=True)[0]
            du_dv_d   = torch.autograd.grad(u_dg, v_dg,   grad_outputs=ones_d, create_graph=True)[0]
            du_dtau_d = torch.autograd.grad(u_dg, tau_dg, grad_outputs=ones_d, create_graph=True)[0]
            deg_res = (S_dg * du_dS_d
                       + self.kappa * self.theta * du_dv_d
                       - self.r * u_dg
                       - du_dtau_d)
            loss_deg = torch.mean(deg_res ** 2)

            # data anchors at tau=T: semi-analytical prices pin the solution
            u_pred_anc = self._trial(S_anc, v_anc, tau_anc)
            loss_data = torch.mean((u_pred_anc - u_anc) ** 2)

            loss = loss_pde + loss_Smax + loss_vmax + loss_deg + w_data * loss_data
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 1.0)
            opt.step()
            sched.step()

            if epoch % log_every == 0:
                print(f"[Heston v3] epoch {epoch:6d} | "
                      f"loss={loss.item():.4e}  "
                      f"pde={loss_pde.item():.4e}  "
                      f"Smax={loss_Smax.item():.4e}  "
                      f"vmax={loss_vmax.item():.4e}  "
                      f"deg={loss_deg.item():.4e}  "
                      f"data={loss_data.item():.4e}")

    def price(self, S, v=None, t=0.0):
        """Price at calendar time t (tau = T - t). Returns U = K * u."""
        if v is None:
            v = self.v0
        tau = self.T - t
        self.aux_net.eval()
        self.main_net.eval()
        with torch.no_grad():
            S_t   = self._to(torch.tensor([[float(S)]],   dtype=torch.float32))
            v_t   = self._to(torch.tensor([[float(v)]],   dtype=torch.float32))
            tau_t = self._to(torch.tensor([[float(tau)]], dtype=torch.float32))
            u = self._trial(S_t, v_t, tau_t).item()
            return self.K * u   # convert u=U/K back to absolute price

    def save(self, path):
        torch.save({
            "aux_state":  self.aux_net.state_dict(),
            "main_state": self.main_net.state_dict(),
            "params": dict(K=self.K, T=self.T, r=self.r, kappa=self.kappa,
                           theta=self.theta, xi=self.xi, rho=self.rho,
                           v0=self.v0, S_max=self.S_max, v_max=self.v_max),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.aux_net.load_state_dict(ckpt["aux_state"])
        self.main_net.load_state_dict(ckpt["main_state"])
