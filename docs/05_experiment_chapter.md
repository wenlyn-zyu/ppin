# 05 — 第四章（实验）写作指南

> **目标**：理解 `Tex/Chap_4_Experiment.tex` 的结构，掌握 LaTeX 表格写法，能够用 `evaluate.py` 的输出更新实验数据。

---

## 1. 章节结构总览

```
第四章 实验
├── 4.1 实验设置（硬件、训练参数表、评估指标公式、基准参数）
├── 4.2 BSM-PINN 实验结果（精度表 + 收敛分析）
├── 4.3 CEV-PINN 实验结果（精度表 + 杠杆效应验证）
├── 4.4 Heston-PINN 实验结果（精度表 + ICPINN 效果）
└── 4.5 LLM 路由演示（交互示例 + 准确率 + 延迟）
```

---

## 2. 如何运行评估脚本

```bash
cd /home/yz2026/zhuwl2022/ppin
/home/yz2026/anaconda3/envs/pinn_option/bin/python src/evaluate.py
```

输出：
- 终端打印 MAE、RMSE、最大误差（直接填入论文表格）
- `results/bsm_eval.pdf`、`cev_eval.pdf`、`heston_eval.pdf`：对比图和误差图

### 当前评估结果

| 模型 | MAE | RMSE | 最大误差 |
|------|-----|------|---------|
| BSM | 0.056 | 0.071 | 0.183 |
| CEV (β=0.5) | 1.099 | 1.312 | 2.847 |
| Heston | 2.357 | 2.891 | 6.124 |

---

## 3. LaTeX 表格写法

### 基本三线表（booktabs 风格）

本模板使用 `\hline` 画线（不用 booktabs），格式如下：

```latex
\begin{table}[htbp]
  \centering
  \caption{表格标题}
  \label{tab:xxx}
  \begin{tabular}{lcc}   % l=左对齐, c=居中, r=右对齐
    \hline
    列1 & 列2 & 列3 \\
    \hline
    数据1 & 数据2 & 数据3 \\
    \hline
  \end{tabular}
\end{table}
```

- `[htbp]` 是浮动体位置参数：h=here, t=top, b=bottom, p=单独一页
- `\caption` 在 `\begin{tabular}` 之前（表格标题在上方）
- `\label` 紧跟 `\caption`，用于交叉引用

### 数字中的千位分隔符

```latex
20{,}000   % 正确：20,000（逗号不会被解析为参数分隔符）
20,000     % 错误：LaTeX 会把逗号当作普通字符，但在数学模式中可能出问题
```

### 引用表格

```latex
如表~\ref{tab:train_params} 所示，……
```

---

## 4. 如何添加评估图表

### 将服务器上的 PDF 图复制到本地

```bash
# 在本地终端运行（或用 WinSCP）
scp yz2026@10.19.126.135:/home/yz2026/zhuwl2022/ppin/results/bsm_eval.pdf \
    D:/Willing/Study/Course/final/sym/paper/latex_m/Img/
```

### 在 LaTeX 中插入图片

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{Img/bsm_eval.pdf}
  \caption{BSM-PINN 预测值与 Black-Scholes 解析解对比}
  \label{fig:bsm_eval}
\end{figure}
```

### 并排两张图

```latex
\begin{figure}[htbp]
  \centering
  \begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{Img/bsm_eval.pdf}
    \caption{BSM 对比图}
    \label{fig:bsm_eval}
  \end{minipage}
  \hfill
  \begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{Img/cev_eval.pdf}
    \caption{CEV 对比图}
    \label{fig:cev_eval}
  \end{minipage}
\end{figure}
```

---

## 5. 如何更新实验数据

如果重新训练了模型（更多 epochs 或调整权重），按以下步骤更新论文：

1. 在服务器运行 `python src/evaluate.py`，记录新的 MAE/RMSE/MaxErr
2. 在 `Chap_4_Experiment.tex` 中找到对应的 `\begin{tabular}` 块，更新数字
3. 如果图也更新了，重新 scp 图片到 `Img/` 目录
4. 保存 `.tex` 文件触发自动四步编译

---

## 6. 评估结果的解读

### 为什么 Heston 误差最大？

Heston PDE 是二维的（$S, v, t$），PDE 中的项量级更大：
- BSM 扩散项：$\sigma^2 S^2 \approx 0.04 \times 10^4 = 400$
- Heston 交叉项：$\rho\xi vS \approx 0.7 \times 0.3 \times 0.04 \times 100 = 0.84$（量级小）
- Heston 方差扩散：$\xi^2 v \approx 0.09 \times 0.04 = 0.0036$（量级小）

损失函数的绝对量级由最大项决定，二维 PDE 的训练更难收敛。

### 相对误差更有意义

平值期权（$S=K=100$）的参考价格约为 10 元，因此：
- BSM 相对误差：0.056/10 ≈ 0.56%
- CEV 相对误差：1.099/10 ≈ 11%（偏高，可通过增加 epochs 改善）
- Heston 相对误差：2.357/10 ≈ 23.6%（偏高，可通过增加 epochs 改善）

### 如何改善 CEV/Heston 精度

```bash
# 增加训练轮数（在 train_all.py 中修改 epochs 参数）
python src/train_all.py --model cev    # 默认 20000 轮，可改为 50000
python src/train_all.py --model heston # 默认 30000 轮，可改为 60000
```

或者调整损失权重（在 `cev_pinn.py` 的 `train()` 方法中）：
```python
# 增大 w_ic 让网络更重视终值条件
losses = model.train(epochs=50000, w_pde=1.0, w_ic=20.0, w_bc=10.0)
```

---

## 7. LLM 路由演示的运行方法

```bash
cd /home/yz2026/zhuwl2022/ppin
/home/yz2026/anaconda3/envs/pinn_option/bin/python src/demo.py
```

按提示输入自然语言描述，系统会：
1. 调用 DeepSeek API 解析参数
2. 加载对应的 PINN 模型
3. 返回期权价格

**注意**：需要网络连接才能调用 DeepSeek API。如果服务器无法访问外网，可以在本地运行 `demo.py`（需要先把 `.pt` 权重文件 scp 到本地）。

---

## 8. 编译后检查清单

- [ ] 所有 `\ref{tab:xxx}` 的 label 都在对应 `\begin{table}` 中定义
- [ ] 表格数字与 `evaluate.py` 输出一致
- [ ] 图片文件已放入 `Img/` 目录（如果有图）
- [ ] `\eqref{eq:hard_constraint}` 等交叉引用指向第三章的公式
- [ ] 每节末尾逻辑完整，不是突然截断
