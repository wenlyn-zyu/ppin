# 04 — 第三章（方法）写作指南

> **目标**：理解 `Tex/Chap_3_Method.tex` 的结构逻辑，掌握其中用到的 LaTeX 技巧，能够自己修改和扩充方法章节。

---

## 1. 章节结构总览

```
第三章 方法
├── 3.1 PINN 统一框架
│   ├── 3.1.1 问题形式化（一般 PDE 形式）
│   ├── 3.1.2 损失函数（三项复合损失）
│   └── 3.1.3 硬约束终值条件（ICPINN）
├── 3.2 网络架构
│   ├── 3.2.1 门控网络（BSM 与 CEV）
│   ├── 3.2.2 深层全连接网络（Heston）
│   └── 3.2.3 输入归一化
├── 3.3 BSM-PINN（PDE 算子 + 边界条件 + 解析解）
├── 3.4 CEV-PINN（PDE 算子 + β 参数说明）
├── 3.5 Heston-PINN（PDE 算子 + 边界条件 + 半解析解）
└── 3.6 LLM 路由层
    ├── 3.6.1 设计动机
    ├── 3.6.2 路由规则
    ├── 3.6.3 参数提取（JSON 示例）
    └── 3.6.4 系统流程
```

**写作逻辑**：先给出统一框架（所有模型共用的 PDE 形式和损失函数），再逐一介绍每个模型的具体算子，最后介绍 LLM 路由层。这样读者先建立整体认知，再看细节。

---

## 2. 关键 LaTeX 技巧

### 2.1 一般形式 PDE（抽象算子写法）

```latex
\begin{equation}\label{eq:general_pde}
  \mathcal{F}[V](x, t) = 0, \quad (x, t) \in \Omega \times [0, T),
\end{equation}
```

- `\mathcal{F}` 是花体字母，用于表示微分算子
- `\quad` 是一个 em 宽度的空格，用于在公式内分隔条件
- `\Omega` 是求解域，`\partial\Omega` 是边界

### 2.2 复合损失函数（多行对齐）

```latex
\begin{equation}\label{eq:loss}
  \mathcal{L}(\theta) = w_\text{pde}\,\mathcal{L}_\text{pde}
                      + w_\text{ic}\,\mathcal{L}_\text{ic}
                      + w_\text{bc}\,\mathcal{L}_\text{bc},
\end{equation}
```

- `\,` 是细空格（thin space），用于改善乘法符号周围的间距
- `w_\text{pde}` 中 `\text{}` 确保下标是直立罗马体，而不是斜体

### 2.3 求和公式（带绝对值）

```latex
\mathcal{L}_\text{pde} &= \frac{1}{N_c}\sum_{i=1}^{N_c}
  \left|\mathcal{F}[\hat{V}_\theta](x_i, t_i)\right|^2,
```

- `\left|...\right|` 自动调整绝对值符号大小以匹配内容
- `\hat{V}` 是带帽子的 V，表示神经网络近似值
- `\sum_{i=1}^{N_c}` 是求和符号，下标上标分别是求和范围

### 2.4 硬约束输出变换

```latex
\begin{equation}\label{eq:hard_constraint}
  \hat{V}_\theta(S, v, t) = \phi(S) + (T - t)\cdot\text{net}_\theta(S, v, t).
\end{equation}
```

- `\cdot` 是居中点乘号，比 `*` 更正式
- `\phi` 是希腊字母 φ，用于表示终值收益函数

### 2.5 门控网络方程组（align 环境）

```latex
\begin{align}
  h_s &= f_\text{shallow}(x), \\
  h_d &= f_\text{deep}(x), \\
  g   &= \sigma(W_g h_s + b_g), \\
  h   &= g \odot h_s + (1-g) \odot h_d, \\
  \hat{V} &= W_\text{out} h + b_\text{out},
\end{align}
```

- `align` 环境用 `&` 对齐等号，`\\` 换行
- `\odot` 是逐元素乘法符号（Hadamard 积）
- `\sigma(\cdot)` 表示 sigmoid 函数

### 2.6 输入归一化（多变量同行）

```latex
\begin{equation}
  \tilde{S} = \frac{S}{S_\max}, \quad
  \tilde{v} = \frac{v}{v_\max}, \quad
  \tilde{t} = \frac{t}{T},
\end{equation}
```

- `\tilde{S}` 是带波浪号的 S，表示归一化后的变量
- 多个公式用 `\quad` 分隔写在同一行，节省空间

### 2.7 verbatim 代码块（JSON 示例）

```latex
\begin{verbatim}
{
  "model": "BSM",
  "S": 100.0
}
\end{verbatim}
```

- `verbatim` 环境原样输出文本，不解析 LaTeX 命令
- 适合展示 JSON、代码片段等需要保留格式的内容

### 2.8 有序列表（系统流程）

```latex
\begin{enumerate}
  \item 用户输入自然语言描述；
  \item LLM 解析输入，提取参数；
  \item PINN 模型推断；
  \item 返回结果。
\end{enumerate}
```

### 2.9 无序列表（路由规则）

```latex
\begin{itemize}
  \item 若用户仅提供常数波动率 $\sigma$，选择 \textbf{BSM}；
  \item 若用户提及弹性参数 $\beta$，选择 \textbf{CEV}；
\end{itemize}
```

- `\textbf{}` 加粗，用于强调关键词

---

## 3. 公式与代码的对应关系

| 论文公式 | 代码位置 | 说明 |
|---------|---------|------|
| 式(3.1) 一般 PDE | 概念性，无直接代码 | 统一框架的抽象表示 |
| 式(3.2) 损失函数 | `bsm_pinn.py:train()` | `loss = w_pde*loss_pde + w_ic*loss_ic + w_bc*loss_bc` |
| 式(3.3-3.5) 各损失项 | `_pde_residual()`, `_sample_terminal()` | 自动微分计算 PDE 残差 |
| 式(3.6) 硬约束 | `heston_pinn.py:HestonNet.forward()` | `payoff + (T-t)*net_out` |
| 式(3.7-3.11) 门控网络 | `GatedPINN.forward()` | `g*h_s + (1-g)*h_d` |
| 式(3.12) 输入归一化 | `_pde_residual()` 中 `S/S_max, t/T` | 所有模型都做了归一化 |
| 式(3.13) BSM PDE | `bsm_pinn.py:_pde_residual()` | `V_t + 0.5*sigma^2*S^2*V_SS + r*S*V_S - r*V` |
| 式(3.14) CEV PDE | `cev_pinn.py:_pde_residual()` | `diffusion = 0.5*sigma^2*S^(2*beta)` |
| 式(3.15) Heston PDE | `heston_pinn.py:_pde_residual()` | 包含交叉偏导 `V_Sv` |

---

## 4. 如何修改方法章节

### 添加新模型（例如 SABR 模型）

1. 在 `sec:pinn_framework` 之后添加新的 `\section{SABR-PINN}`
2. 写出 PDE 算子（参考 BSM/CEV 的写法）
3. 说明边界条件
4. 在 `sec:llm_router` 的路由规则中添加新模型的判断条件

### 修改网络架构描述

如果你改变了网络层数或神经元数，在 `sec:architecture` 对应的 `\subsection` 中更新数字即可。

### 添加系统架构图

在章节开头引用图片：

```latex
系统整体架构如图~\ref{fig:system}~所示。
```

然后在 `Img/` 目录放入图片，在 `Tex/Chap_3_Method.tex` 中添加：

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{Img/system_arch.pdf}
  \caption{系统整体架构图}
  \label{fig:system}
\end{figure}
```

---

## 5. 编译后检查清单

- [ ] 所有 `\label{eq:xxx}` 都有对应的 `\eqref{eq:xxx}` 引用
- [ ] `align` 环境中每行末尾有 `\\`，最后一行不加
- [ ] `\citep{guo2024icpinn}` 等引用的 key 在 `ref.bib` 中存在
- [ ] 公式中的变量名与代码中的变量名一致（如 `beta` vs `β`）
- [ ] 每个 `\subsection` 都有实质内容，不是空标题
