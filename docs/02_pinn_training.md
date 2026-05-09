# 02 — PINN 三大模型训练指南

> **目标**：理解三大期权定价 PINN 的原理、代码结构，并在服务器上完成训练。

---

## 1. 为什么用 PINN 而不是有限差分

传统数值方法（有限差分 FD、有限元 FEM）需要在网格上离散化 PDE，网格越细精度越高但计算量指数增长。对于 Heston 这样的二维 PDE，网格点数是 $N_S \times N_v \times N_t$，内存和时间开销很大。

PINN 的核心思想（Raissi et al., 2019）：**用神经网络近似 PDE 的解，把 PDE 残差直接写进损失函数**。

$$\mathcal{L} = w_\text{pde}\underbrace{\frac{1}{N_c}\sum_{i=1}^{N_c}|\mathcal{F}[V](x_i)|^2}_{\text{PDE 残差}} + w_\text{ic}\underbrace{\frac{1}{N_T}\sum_j|V(S_j,T)-\text{payoff}(S_j)|^2}_{\text{终值条件}} + w_\text{bc}\underbrace{\frac{1}{N_b}\sum_k|V_\text{boundary}|^2}_{\text{边界条件}}$$

优势：无需网格，天然支持高维，训练一次可对任意 $(S,t)$ 即时推断。

---

## 2. 三大模型的 PDE 与边界条件

### BSM（一维，`src/models/bsm_pinn.py`）

**PDE**：
$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

**条件**：
| 条件 | 表达式 | 含义 |
|------|--------|------|
| 终值 | $V(S,T)=\max(S-K,0)$ | 到期日看涨期权收益 |
| 下边界 | $V(0,t)=0$ | 股价为零时期权无价值 |
| 上边界 | $V(S_\max,t)=S_\max - Ke^{-r(T-t)}$ | 深度实值近似 |

**验证方法**：与 Black-Scholes 解析解对比（`bs_call_price` 函数）。

---

### CEV（一维，`src/models/cev_pinn.py`）

**PDE**：
$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^{2\beta} \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

与 BSM 的唯一区别：扩散项从 $\sigma^2 S^2$ 变为 $\sigma^2 S^{2\beta}$。

**参数选择**：默认 $\beta=0.5$（平方根过程），最稳定。$\beta<0$ 时 $S\to 0$ 处有奇异性，避免使用。

**验证方法**：与 Crank-Nicolson 有限差分参考解对比（`cev_fd_call` 函数，代码内置）。

---

### Heston（二维，`src/models/heston_pinn.py`）

**PDE**（输入为 $S, v, t$ 三维）：
$$\frac{\partial V}{\partial t} + \frac{1}{2}vS^2\frac{\partial^2 V}{\partial S^2} + \rho\xi vS\frac{\partial^2 V}{\partial S\partial v} + \frac{1}{2}\xi^2 v\frac{\partial^2 V}{\partial v^2} + rS\frac{\partial V}{\partial S} + \kappa(\theta-v)\frac{\partial V}{\partial v} - rV = 0$$

**关键设计——硬约束（ICPINN）**：

普通 PINN 把终值条件放进损失函数（软约束），网络不能精确满足。ICPINN 通过输出变换**硬编码**终值条件：

```python
# 网络输出变换：V_hat 在 t=T 时精确等于 payoff
V_hat(S,v,t) = payoff(S) + (T - t) * net(S,v,t)
# 当 t=T 时：V_hat = payoff + 0 * net = payoff  ✓ 精确满足
```

这样终值条件从损失函数中消失，网络只需学习 PDE 残差和边界条件，收敛更快。

**验证方法**：与 Heston 半解析解（特征函数方法）对比（`heston_call_price` 函数）。

---

## 3. 网络架构：门控网络（Gated PINN）

BSM 和 CEV 使用门控网络（参考 Dhiman & Hu, 2023）：

```
输入 (S_norm, t_norm)
       ↓              ↓
  浅层分支          深层分支
  (1层隐藏)        (4层隐藏)
       ↓              ↓
      h_s            h_d
       ↓
  gate = sigmoid(W * h_s)   ← 门控权重由浅层决定
       ↓
  h = gate * h_s + (1-gate) * h_d   ← 加权融合
       ↓
  输出 V
```

**为什么用门控？** 期权价格在深度实值/虚值区域接近线性（简单），在平值附近高度非线性（复杂）。浅层分支捕捉简单线性部分，深层分支捕捉复杂非线性部分，门控自动分配权重。

Heston 使用更深的全连接网络（6层，128个神经元），因为输入维度更高。

---

## 4. 训练参数

| 参数 | BSM | CEV | Heston |
|------|-----|-----|--------|
| Epochs | 20,000 | 20,000 | 30,000 |
| 配点数 $N_c$ | 10,000 | 10,000 | 15,000 |
| 终值点 $N_T$ | 2,000 | 2,000 | — (硬约束) |
| 边界点 $N_b$ | 1,000 | 1,000 | 1,000×3 |
| $w_\text{pde}$ | 1.0 | 1.0 | 1.0 |
| $w_\text{ic}$ | 10.0 | 10.0 | — |
| $w_\text{bc}$ | 5.0 | 5.0 | 5.0 |
| 优化器 | Adam lr=1e-3 | Adam lr=1e-3 | Adam lr=5e-4 |
| 学习率衰减 | StepLR ×0.5/5000步 | 同左 | 同左 |

**为什么 $w_\text{ic}=10$？** 终值条件是最强的约束（直接决定解的形状），权重高可以让网络优先学会正确的边界行为，再逐步满足 PDE 残差。

---

## 5. 如何运行训练

### 训练全部模型（推荐，约 2-3 小时）

```bash
# 在服务器上
cd /home/yz2026/zhuwl2022/ppin
nohup /home/yz2026/anaconda3/envs/pinn_option/bin/python -u src/train_all.py \
  > results/train.log 2>&1 &
echo $!   # 记录进程 ID

# 查看训练进度
tail -f results/train.log
```

### 只训练某一个模型

```bash
python src/train_all.py --model bsm      # 只训练 BSM
python src/train_all.py --model cev      # 只训练 CEV
python src/train_all.py --model heston   # 只训练 Heston
```

### 查看 GPU 使用情况

```bash
nvidia-smi   # 查看 GPU 利用率和显存
# 训练时应看到 GPU 利用率 ~80-95%，显存占用 ~15-20GB
```

### 训练完成的标志

```
results/
├── bsm_pinn.pt      # BSM 模型权重（~74KB）
├── cev_pinn.pt      # CEV 模型权重（~74KB）
├── heston_pinn.pt   # Heston 模型权重（~331KB，更大因为网络更深）
└── train.log        # 训练日志
```

---

## 6. 训练结果（本次运行）

| 模型 | 最终 Loss | PDE 残差 | 边界误差 | 训练时间 |
|------|-----------|---------|---------|---------|
| BSM | 4.2e-1 | 1.75e-1 | 1.2e-3 | ~7 分钟 |
| CEV (β=0.5) | 1.7e-1 | 2.4e-2 | — | ~7 分钟 |
| Heston | 6.1e+3 | 1.86e+2 | 1.19e+3 | ~17 分钟 |

**Heston loss 量级大是正常的**：二维 PDE 的数值量级本来就比一维大（$S$ 最大 300，$v$ 最大 1，乘积项量级在 $10^4$）。评估时看相对误差（MAE/价格）才有意义。

---

## 7. 如何评估模型精度

```bash
cd /home/yz2026/zhuwl2022/ppin
/home/yz2026/anaconda3/envs/pinn_option/bin/python src/evaluate.py
```

输出：
- 终端打印 MAE、RMSE、最大误差
- `results/bsm_eval.pdf`、`cev_eval.pdf`、`heston_eval.pdf`：对比图和误差图

这些图可以直接用于论文第四章实验部分。
