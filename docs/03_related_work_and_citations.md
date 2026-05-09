# 03 — Related Work 写作指南与引用规范

> **目标**：理解 `Tex/Chap_2_RelatedWork.tex` 的结构逻辑，掌握 LaTeX 引用的正确写法，能够自己修改和扩充相关工作章节。

---

## 1. Related Work 的结构逻辑

本文的 Related Work（第二章）按三个方向组织，每个方向的写法遵循同一个模式：

```
背景与意义（1-2句）→ 代表性工作（按贡献递进）→ 局限性（引出本文）
```

三个方向的选择依据：

| 方向 | 为什么要写 | 对应本文哪个部分 |
|------|-----------|----------------|
| 期权定价模型（BSM/CEV/Heston） | 给出三大 PDE 的数学形式，是后续方法章的基础 | 第三章方法 |
| PINN 期权定价 | 直接相关工作，说明本文站在哪些工作的肩膀上 | 第三章方法 |
| LLM 辅助科学计算 | 支撑 LLM 路由层设计的合理性 | 第三章 LLM 路由 |

末尾的"局限性"段落非常重要——它直接回答"你的工作和已有工作有什么不同"，是审稿人最关注的地方。

---

## 2. 参考文献数据库（`Biblio/ref.bib`）

### BibTeX 条目类型

| 类型 | 用于 | 必填字段 |
|------|------|---------|
| `@article` | 期刊论文、arXiv | author, title, journal, year |
| `@inproceedings` | 会议论文 | author, title, booktitle, year |
| `@mastersthesis` | 硕士论文 | author, title, school, year |
| `@book` | 书籍 | author, title, publisher, year |

### Key 命名规范

格式：`作者姓氏年份关键词`，全小写，例如：

```
black1973pricing      ← Black (1973) 期权定价
raissi2019physics     ← Raissi (2019) PINN
hainaut2024heston     ← Hainaut (2024) Heston
```

### 专有名词大写保护

BibTeX 默认把标题转为首字母大写，专有名词会被错误地小写。用 `{}` 包裹保护：

```bibtex
% 错误：会变成 "A closed-form solution for options with stochastic volatility"
title = {A Closed-Form Solution for Options with Stochastic Volatility}

% 正确：{H}eston 保护大写 H
title = {A Closed-Form Solution for Options with {S}tochastic {V}olatility}

% 更常见的写法：保护整个专有名词
title = {Option Pricing in the {Heston} Model with {PINN}}
```

### arXiv 论文写法

```bibtex
@article{dhiman2023pinn,
  author  = {Dhiman, Ashish and Hu, Yibei},
  title   = {Physics Informed Neural Network for Option Pricing},
  journal = {arXiv preprint arXiv:2312.06711},
  year    = {2023}
}
```

### 期刊论文必须填 DOI

```bibtex
@article{hainaut2024heston,
  ...
  doi = {10.1007/s10436-024-00452-7}
}
```

---

## 3. 正文中的引用命令

本模板使用 `natbib` 宏包，引用样式为 `gbt7714-unsrt`（国标顺序引用）。

### 两种基本命令

```latex
\citet{key}   % 作者作主语：Raissi 等 [1] 提出了……
\citep{key}   % 括号引用：……（[1]）或 ……[1]
```

实际显示效果取决于模板设置（本模板是上标数字）：

```latex
Raissi 等人\citet{raissi2019physics}于 2019 年提出了 PINN。
% → Raissi 等人[1]于 2019 年提出了 PINN。

PINN 已被广泛应用于科学计算\citep{raissi2019physics}。
% → PINN 已被广泛应用于科学计算[1]。
```

### 同时引用多篇

```latex
\citep{black1973pricing, merton1973theory}
% → [1,2]
```

### 引用特定页码（一般不用）

```latex
\citep[p.~327]{heston1993closed}
% → [3, p. 327]
```

---

## 4. 公式写法

Related Work 里的三大 PDE 公式是论文的核心，写法如下：

### 行内公式

```latex
BSM 模型假设波动率为常数 $\sigma$，股价满足 $dS = \mu S\,dt + \sigma S\,dW_t$。
```

### 独立编号公式

```latex
\begin{equation}\label{eq:bsm}
  \frac{\partial V}{\partial t}
  + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
  + rS\frac{\partial V}{\partial S}
  - rV = 0
\end{equation}
```

### 多行对齐公式（Heston 的两个 SDE）

```latex
\begin{align}
  dS &= \mu S\,dt + \sqrt{v}\,S\,dW_t^S, \\
  dv &= \kappa(\theta - v)\,dt + \xi\sqrt{v}\,dW_t^v.
\end{align}
```

### 交叉引用公式

```latex
如式~\eqref{eq:bsm} 所示，BSM 方程……
% → 如式 (1) 所示，BSM 方程……
```

**注意**：`\eqref` 自动加括号，不要写成 `式(\ref{eq:bsm})`。

---

## 5. 本文已有的 13 条参考文献

| Key | 作者/年 | 类型 | 用于 |
|-----|---------|------|------|
| `black1973pricing` | Black & Scholes 1973 | 期刊 | BSM 模型 |
| `merton1973theory` | Merton 1973 | 期刊 | BSM 模型 |
| `cox1976valuation` | Cox & Ross 1976 | 期刊 | CEV 模型 |
| `heston1993closed` | Heston 1993 | 期刊 | Heston 模型 |
| `raissi2019physics` | Raissi et al. 2019 | 期刊 | PINN 基础 |
| `dhiman2023pinn` | Dhiman & Hu 2023 | arXiv | BSM-PINN，门控网络 |
| `zamxaka2023cev` | Zamxaka 2023 | 硕士论文 | BSM+CEV PINN |
| `louskos2023american` | Louskos 2023 | arXiv | 美式期权 PINN |
| `guo2024icpinn` | Guo et al. 2024 | arXiv | ICPINN 硬约束 |
| `hainaut2024heston` | Hainaut & Casas 2024 | 期刊 | Heston PINN |
| `bae2024localvol` | Bae et al. 2024 | 期刊 | CEV+局部波动率 PINN |
| `dcpinns2024` | Hadiasadeh et al. 2024 | arXiv | DC-PINN |
| `liu2019pricing` | Liu et al. 2019 | 期刊 | NN 期权定价 |
| `liu2025pdeagent` | Liu et al. 2025 | arXiv | LLM+PDE |
| `xiao2024tradingagents` | Xiao et al. 2024 | arXiv | LLM 金融 |

---

## 6. 如何添加新文献

1. 在 `Biblio/ref.bib` 末尾添加新条目
2. 在正文中用 `\citep{key}` 引用
3. 重新做完整四步编译（保存 `.tex` 文件触发自动编译）

**快速获取 BibTeX 条目的方法**：
- Google Scholar → 点击引用图标 → 选 BibTeX → 复制粘贴
- DOI 链接 → `https://doi.org/10.xxxx/xxxxx` → 在 doi2bib.org 转换
- arXiv 页面 → 右侧 "Export BibTeX citation"

---

## 7. 编译后检查清单

- [ ] 所有 `\citep{}` 的 key 都在 `ref.bib` 中存在
- [ ] 参考文献列表末尾没有 `[?]` 或 `??`
- [ ] 公式编号连续，没有跳号
- [ ] 专有名词大写正确（Heston、PINN、BSM 等）
- [ ] 每章末尾有 `\section{本章小结}` 段落
