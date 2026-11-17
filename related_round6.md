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

---

## 阶段二 T0：文献核对（2026-11-17 18:00 前后；用 web 检索 + 子代理读原文 HTML）

### 1. Karasan 2026（arXiv 2608.27020）

Abdullah Karasan, *Representation Measurements Under Function-Preserving Reparameterizations*（stat.ML / cs.LG，2026-08-27，v1）。
主题：保持函数不变的隐藏基变换下，column-permutation parallel analysis（PA）给出的成分数会变（5 个模型、75 个变换，成分数不一致率中位数 0.79），
而右正交不变的比较量稳定；证明数据内部的参照过程不能同时保持边缘分布、正交等变并去掉跨坐标协方差。
- 各向异性 / isotropy：**没有涉及**（"isotropic" 只出现在 CKA 的不变性和参照分布中）。
- 余弦相似度：**没有涉及**。
- final norm 折算：只有一句用来界定可容许的变换族（Section 3："For the RMSNorm models considered here, coordinatewise gain parameters can be
  absorbed into adjacent linear maps, allowing any R∈O(d)."），没有讨论 unembedding，也没有分析折算对指标的影响。
- 与本项目的关系：同一个原则（测量应对保持函数的重参数化不变），但对象是 PA 的成分数，变换族是正交族。
- 注意：HTML 只有前约 4 万字符逐字读到（摘要至 Section 3.2），其余部分的"没有涉及"依据的是模型摘要。

### 2. Macocco et al. 2025（arXiv 2503.21718）的 OD 定义

阶段一已对照论文 v4 的 Section 3 与代码 `src/evaluate.py`（`outlier_dims_quantile`），逐字记录在本文件上方。S3 的实现
（|h| 逐维中位数 > 全体 |h| 的 0.99 分位数）与两者一致，不需要修正。仍然存在的差别只是数据（bank 的约 9940 个 token，而非 5 万个独立片段的固定位置），
已在 REPORT 中写明。

### 3. Park, Choe & Veitch（ICML 2024，arXiv 2311.03658）与 h* 的关系

*The Linear Representation Hypothesis and the Geometry of Large Language Models*，Kiho Park、Yo Joong Choe、Victor Veitch（芝加哥大学）。读的是 v2 HTML（2024-07-17）。
- Section 3 "Unidentifiability of the inner product"：g(y) ← Aγ(y) + β，l(x) ← A^{−T}λ(x)（Eq. 3.1）不改变 softmax，
  所以 "the representation γ is identified (at best) only up to some invertible affine transformation"，欧氏内积不是由训练确定的。
- 因果内积（Definition 3.1）：使因果可分的概念的方向正交的内积。Theorem 3.4 给出一族解（M^{−1} = GGᵀ，GᵀCov(γ)^{−1}G = D），
  其中 γ 是"从词表中均匀抽取的 unembedding 向量"（Assumption 3.3，不按词频）。他们取 D = I，即 ⟨γ̄, γ̄′⟩_C = γ̄ᵀ Cov(γ)^{−1} γ̄′（Eq. 3.3），
  并用 A = M^{1/2} = Cov(γ)^{−1/2} 变换，使因果内积成为变换后空间的欧氏内积。
- **与 h* 的关系（我们的推导）**：同一个 A 作用在 embedding（隐藏状态）一侧是 λ ↦ A^{−T}λ = Cov(γ)^{1/2} λ。
  我们的均匀版规范度量 M = W_Uᵀ(I − 11ᵀ/V)W_U = V·Cov(γ)（γ 为 W_U 的行，词表上均匀），所以 **h* = M^{1/2} h = √V · Cov(γ)^{1/2} h**，
  与 Park 等人变换后的 embedding 只差一个常数因子（不影响任何余弦类指标）。区别只有两点：(a) 我们用增广形式 [h; 1] 把 LayerNorm 的 β / logit 偏置
  一并纳入，使 h* 对平移和 β 折算也不变；(b) 我们另算了按词频加权的 P_π 版本，他们没有。
  所以阶段一的 h* 不是新构造，应表述为"Park 等人的因果内积在隐藏状态一侧诱导的几何"；阶段一的新内容是用它测 isotropy 并与 V1 / V3 比较。
- 他们的出发点（内积不可识别）与本项目线 A 相同，但他们讨论的是概念方向的内积，没有讨论各向异性指标或跨模型比较。
- 未逐字核对的部分（Section 4 之后与附录，例如 Eq. 4.1、附录 D.6 的 LLaMA-2 词表检查）来自模型摘要。

### 4. 检索："anisotropy + reparameterization / LayerNorm folding / gauge / whitening invariance"

检索式（web 检索，2026-11-17）：
1. `anisotropy language model hidden states invariant reparameterization LayerNorm folding rescaling isotropy not identifiable`
2. `"anisotropy" language models comparison across models final layer normalization gain unembedding equivalent parameterization cosine similarity ranking changes`
3. `whitening Mahalanobis cosine affine invariance sentence embeddings whitening invariant linear transformation`
4. `arXiv 2608.27020`

相关命中：
- **Cain 2026，*Gauge Freedom and Metric Dependence in Neural Representation Spaces*（arXiv 2603.06774，单作者预印本，无发表信息；全文读到）**：
  与本项目线 A 的机制**部分重合**。它指出 h → Dh、W → WD^{−1}（D ∈ GL(d)）不改变网络函数，"Cosine similarity is not invariant under general gauge
  transformations of the representation space"（Section 4），并把"anisotropy in embedding spaces"列为可由此解释的现象（摘要）；把白化 Σ^{−1/2}
  称为一个规范选择（Section 4.1、7.4）。但：实验只有 Digits 上的两层 MLP 和 CIFAR-10 上的小 CNN；**没有语言模型、没有 LayerNorm / RMSNorm 增益折算、
  没有 final norm 或 unembedding、没有跨模型的 isotropy 排序**，也没有给出不变的度量；Section 8 明确把归一化层和 Transformer 列为未检验。
  → **不触发 T0 的停机条件**（停机条件要求已有工作报告"isotropy 的跨模型排序在保持输出的重缩放下不可识别"）。写作时应引用它作为"余弦 / 各向异性依赖坐标"的一般性论述。
- Timkey & van Schijndel 2021（EMNLP，*All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality*）：
  少数"rogue"维度主导 GPT-2 等最后一层的余弦，但不主导模型行为；建议逐维标准化。与阶段一 S3（OD ≈ 均值维度）和 P6-4（zscore 不变）直接相关。
- Machina & Mercer 2024（已在第四轮讨论）、Godey et al. 2024 的 Pythia 饱和（第五轮已讨论）：未讨论规范问题。
- 其余命中（Ethayarajh 2019、Anisotropy Is Inherent to Self-Attention、Parupudi 2026 等）都不涉及重参数化。
- 局限：只做了 web 检索，没有系统检索 ACL Anthology / OpenReview；检索未命中不等于不存在。

### 5. 检索："白化 / Mahalanobis 余弦的仿射不变性用于句向量"（T3 用）

- Su et al. 2021，*Whitening Sentence Representations for Better Semantics and Faster Retrieval*（arXiv 2103.15316）：把句向量均值变为 0、协方差变为单位阵（BERT-whitening），
  动机是缓解各向异性。WhitenedCSE（ACL 2023）在对比学习中使用白化。两者都没有陈述"白化余弦对可逆线性变换不变"。
- Cain 2026 Section 4.1："cosine similarity in the original coordinates can be interpreted as angular similarity under the Mahalanobis metric induced by Σ"，
  但也没有明确写出不变性。
- 不变性本身是标准的线性代数（我们的推导，T3 用作 P7-3 的依据）：若拟合集协方差 Σ 满秩，对任意可逆 A，变换后协方差为 AΣAᵀ，
  ZCA 白化矩阵 (AΣAᵀ)^{−1/2} 满足 (AΣAᵀ)^{−1/2} A = O Σ^{−1/2}，其中 O 正交（极分解），所以白化后的向量只差一个正交变换，余弦不变。
  对均值的平移同样消去。前提是 Σ 满秩且条件数允许数值上稳定求逆。
