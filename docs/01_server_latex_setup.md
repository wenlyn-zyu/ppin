# 01 — 服务器环境搭建与 LaTeX 编译配置

> **目标**：在 idata2 服务器上完成 TeX Live 安装、LaTeX Workshop 配置，实现保存即自动四步编译。

---

## 1. 为什么在服务器上编译

本地 Windows 用的是 TeX Live 2023，服务器用 Ubuntu 22.04。两边都能编译，但：

- 服务器 GPU 资源充足，训练代码和论文编译可以同时进行
- Remote-SSH 连接后，VSCode 的 LaTeX Workshop 插件直接在服务器上运行，编译结果实时预览，不需要来回传文件
- 统一在服务器上编译，避免本地/服务器字体不一致导致的排版差异

---

## 2. 已安装的 TeX Live 包

通过 `apt` 安装了以下包（需要 sudo 权限）：

```bash
sudo apt-get install -y \
  texlive-xetex \          # XeTeX 引擎，支持 Unicode 和系统字体
  texlive-lang-chinese \   # 中文支持（xeCJK、ctex 等）
  texlive-fonts-recommended \  # 常用字体
  texlive-latex-extra \    # 额外宏包（caption、subcaption 等）
  texlive-fonts-extra \    # 扩展字体（newtxmath 等数学字体）
  texlive-science \        # 科学类宏包（algorithmicx、algorithm 等）
  texlive-plain-generic    # 通用宏包
```

**为什么分多个包？** Ubuntu 的 TeX Live 按功能拆分，不像 Windows 版一次装全。每次遇到 `File xxx.sty not found` 错误，就用 `apt-file search xxx.sty` 找到对应包再安装。

---

## 3. LaTeX Workshop 插件配置

插件安装命令（只需运行一次）：

```bash
~/.vscode-server/bin/07ff9d6178ede9a1bd12ad3399074d726ebe6e43/bin/code-server \
  --install-extension james-yu.latex-workshop
```

> **注意**：`07ff9d6...` 是 VSCode Server 的版本 hash，如果 VSCode 升级后 hash 会变。
> 可以用 `ls ~/.vscode-server/bin/` 查看当前 hash。

配置文件位于项目根目录的 `.vscode/settings.json`，核心内容解释如下：

```jsonc
{
  // 定义编译工具（xelatex 和 bibtex 的调用方式）
  "latex-workshop.latex.tools": [...],

  // 定义 recipe（工具的执行顺序）
  "latex-workshop.latex.recipes": [
    {
      "name": "xelatex -> bibtex -> xelatex x2 (完整编译)",
      "tools": ["xelatex", "bibtex", "xelatex", "xelatex"]
      // 为什么要跑四步？
      // 第1步 xelatex：生成 .aux 文件（记录引用关系）
      // bibtex：读取 .aux，从 ref.bib 生成 .bbl（参考文献列表）
      // 第2步 xelatex：把 .bbl 编译进 PDF，生成目录和交叉引用
      // 第3步 xelatex：解决第2步产生的新引用，确保页码和目录正确
    }
  ],

  // 默认使用完整编译 recipe
  "latex-workshop.latex.recipe.default": "xelatex -> bibtex -> xelatex x2 (完整编译)",

  // 保存 .tex 文件时自动触发编译
  "latex-workshop.latex.autoBuild.run": "onSave",

  // PDF 在 VSCode 侧边 tab 中预览
  "latex-workshop.view.pdf.viewer": "tab"
}
```

---

## 4. 如何使用

### 日常编辑流程

1. 用 VSCode Remote-SSH 连接 `idata2`
2. 打开 `/home/yz2026/zhuwl2022/ppin/` 文件夹
3. 编辑任意 `Tex/*.tex` 文件
4. **按 `Ctrl+S` 保存** → LaTeX Workshop 自动触发四步编译
5. 编译完成后，点击右上角的 PDF 图标（或按 `Ctrl+Alt+V`）预览

### 手动触发编译

- 按 `Ctrl+Shift+P` → 输入 `LaTeX: Build with recipe` → 选择编译方式
- 或点击左侧 LaTeX 面板 → `Build LaTeX project`

### 查看编译错误

- 左侧 LaTeX 面板 → `View Log Messages`
- 或直接看底部状态栏，红色表示编译失败

### 命令行编译（不依赖 VSCode）

```bash
cd /home/yz2026/zhuwl2022/ppin
xelatex -interaction=nonstopmode Thesis.tex
bibtex Thesis
xelatex -interaction=nonstopmode Thesis.tex
xelatex -interaction=nonstopmode Thesis.tex
```

---

## 5. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `File xxx.sty not found` | 对应 TeX Live 包未安装 | `sudo apt-get install texlive-xxx` |
| 目录/引用显示 `??` | 只编译了一次 | 用完整四步 recipe |
| 中文乱码 | 字体缺失 | `sudo apt-get install fonts-wqy-zenhei` |
| PDF 预览空白 | 编译失败 | 查看 Log Messages |

---

## 6. 项目文件结构

```
ppin/
├── Thesis.tex              # 主文件，不要直接在这里写内容
├── .vscode/
│   └── settings.json       # LaTeX Workshop 配置（本文档配置的）
├── Tex/
│   ├── Mainmatter.tex      # 章节入口，\input 各章
│   ├── Chap_1_Intro.tex    # 第一章（引言）
│   ├── Chap_2_RelatedWork.tex  # 第二章（相关工作）✓ 已完成
│   ├── Chap_3_Method.tex   # 第三章（方法）
│   ├── Chap_4_Experiment.tex   # 第四章（实验）
│   └── Chap_5_Conclusion.tex   # 第五章（结论）
├── Biblio/
│   └── ref.bib             # 参考文献数据库
├── Style/                  # 模板样式文件（不要修改）
├── src/                    # PINN 代码
└── results/                # 训练结果和图表
```
