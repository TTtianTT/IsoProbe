# 第六轮相关工作笔记

## Macocco et al. 2025：outlier 维度（OD）的定义

Iuri Macocco, Nora Graichen, Gemma Boleda, Marco Baroni. *Not a nuisance but a useful heuristic: Outlier dimensions favor
frequent tokens in language models.* arXiv 2503.21718（v4，2025-10-03；BlackboxNLP @ EMNLP 2025）。
代码：github.com/imacocco/llms_outlier_dimensions（main 分支，MIT）。2026-11-17 在登录节点联网核对了论文 HTML（v4）与代码。

**论文（Section 3, "OD identification"，原文摘录）**

- 数据："we sample 50k non-overlapping fragments of 101 (typographic) words which do not cross document boundaries"（WikiText-103），
  前 100 词作为上下文，取其后一个位置的表示；只研究最后一层。
- "We consider extreme values those that are in the top 1% when ranking the absolute values of the activations of all
  dimensions in a given layer for our whole dataset."
- "We then define an OD as a dimension whose median activation across the 50k samples in our dataset is an extreme value."
- 即：一个 OD 在至少 50% 的输入上的激活落在"所有输入 × 所有维度"的前 1% 绝对值之内。

**代码（`src/evaluate.py`，`outlier_dims_quantile(X, loc=0.5, glo=0.99)`，原文摘录）**

```python
X = abs(X)
qt = np.quantile(X, q=loc, axis=0)      # 每个维度在输入上的 |激活| 中位数
threshold = np.quantile(X, q=glo)       # 全部输入 × 全部维度的 |激活| 的 0.99 分位数
idx = np.where(qt > threshold)[0]
```

`config.yaml`：`qt: 0.5`，`threshold: 0.99`。表示取 HF `hidden_states[num_layers]`（对 GPT-2 / Pythia 即 final norm 之后），
每个片段一个位置。论文与代码的选取规则一致；默认配置的数据文件名为 `wiki5k`，与论文的 5 万个片段是否一致未核对。

**本轮的近似（S3）**：用 `cache/banks/` 中每个模型的 10000 个 bank token（1000 条 wikitext-103 test 段落 × 每条 10 个随机位置，
去掉位置 0 后约 9940 个），而不是 5 万个独立片段的固定位置；同一段落内的 10 个 token 不独立。规则与代码完全相同
（|h| 的逐维中位数 > 全局 0.99 分位数）。
