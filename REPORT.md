# IsoProbe 预实验报告：现代 LLM 隐藏表征的各向同性

> **这是预实验。** 模型集合不构成按规模或家族设计的对照。第 1 轮只用本地已有模型（GPT-2 small 除外，为正确性检查下载）；
> 第 2 轮按需补下载了 gpt2-medium/large/xl 和两个 base 模型（Llama-3.1-8B、Qwen3-8B-Base），见 `models_local.md`。
> 现代模型中只有这两对是 base 与 post-trained 的同架构配对，其余 4 个现代模型只有 instruct / post-trained 版本。

日期：2026-10-28。Slurm array 9725（第 1 轮，7 个任务）、9743（第 2 轮，12 个任务：新模型的 `measure.py` + 全部模型的
final-norm 分解）和 9769（第 3 轮，12 个任务），均为 `%2`、每任务 1 张 B300，全部 COMPLETED，单任务 1–4 分钟；
第 3 轮另有 CPU 任务 9770（词频）。

## 1. 设置

| 项 | 设置 |
|---|---|
| 数据 | `Salesforce/wikitext` / `wikitext-103-raw-v1` / test。原论文用 WIT 的维基文本，这里用 wikitext 代替。段落 = raw 文件中非空、非标题的一行；≥100 词的候选只有 1085 段，seed=0 抽 1000 段（`data/samples.jsonl`），即几乎覆盖整个候选池。文本保留 wikitext 原始的 ` @-@ `、` , ` 等分词痕迹 |
| 分词 | 各模型自己的 tokenizer，`add_special_tokens=True`，截断到 256；平均长度 185–195 token（最短 100–103） |
| BOS | Llama-3.1（base 与 instruct，`<\|begin_of_text\|>`）和 Mistral（`<s>`）在 1000/1000 条上加了 BOS；GPT-2 系列、Qwen2.5、Qwen3（含 Base）、Olmo-3.1 都没加，位置 0 是普通文本 token（最常见为 `The`，226/1000） |
| 前向 | bf16，batch_size=1（无 padding），`output_hidden_states=True`，`torch.no_grad()`；指标全部 fp32；SDPA 关闭 cuDNN 后端（B300 上 cuDNN SDPA 无可用执行计划，与 TurnPrint 的做法一致） |
| 层 | `hidden_states[0]`（embedding 输出）、`round(k·L/8)` (k=1..7)、`hidden_states[-1]`。对每个模型检查了 `lm_head(hidden_states[-1])` 与 logits 的最大绝对差：12 个模型全部为 0.0，即最后一层确实是 **final norm 之后、LM head 之前** 的表示（图中用 ◆ 标注）|
| Token bank | 每条序列随机抽 10 个位置（均匀抽样），共 10000 个；其中约 62–64 个是位置 0（0.6%）。"去掉位置 0"版本把这些剔除 |
| 环境 | torch 2.11.0+cu128，transformers 5.16.1，seed=0，`/dataset1/zailong/envs/peft-sft-lab` |

指标定义按任务书：A = 1 − mean_x s(x)，s(x) = (‖Σu_i‖² − T)/(T(T−1))；B = 2 万对跨序列 token 的平均余弦（原始 / 减 bank 均值）；C = 中心化 bank 的奇异值谱，RankMe = exp(−Σ p log p)，p = σ/Σσ；D = 各维度 mean|h| 的 max/median，以及 ‖h₀‖/median‖h_t,t>0‖。

**Final-norm 分解**（`diag_final_norm.py`，第 2 轮新增）：用 hook 取 final norm 的输入 x（= 最后一个 block 的输出，不在 `hidden_states` 里），按 norm(x) = γ⊙x̂ + β 逐步重算 A/B/C：x → x̂（只归一化）→ γ⊙x̂ → γ⊙x̂+β（= `hidden_states[-1]`）；对带偏置的 LayerNorm 另算 post − β。所用序列与 bank 位置和 `measure.py` 完全相同。fp32 重算与模块输出的相对误差 ≤ 2e-3。

正确性：`tests/test_metrics.py`（9 个测试全过）验证了 O(Td) 公式与暴力两两余弦一致（含 bf16 输入、带均值偏移的输入）、跨序列配对不会抽到同一序列、RankMe 在已知秩数据上的取值。

## 2. 结果

### 2.1 最后一层（`hidden_states[-1]`，final norm 之后）

**全部位置**

| model | params | 版本 | BOS | A isotropy | B cos | B cos（中心化） | RankMe/d | top-1 var | top-10 var |
|---|---|---|---|---|---|---|---|---|---|
| gpt2 | 0.12B | base | no | **0.068** | 0.931 | 0.009 | 0.321 | 0.788 | 0.972 |
| gpt2-medium | 0.35B | base | no | 0.319 | 0.674 | 0.037 | 0.208 | 0.920 | 0.991 |
| gpt2-large | 0.77B | base | no | 0.708 | 0.163 | 0.000 | 0.840 | 0.040 | 0.204 |
| gpt2-xl | 1.56B | base | no | 0.723 | 0.149 | 0.000 | 0.836 | 0.035 | 0.185 |
| Qwen2.5-7B-Instruct | 7.62B | instruct | no | 0.600 | 0.293 | -0.001 | 0.670 | 0.164 | 0.475 |
| Llama-3.1-8B | 8.03B | **base** | yes | 0.703 | 0.140 | -0.000 | 0.724 | 0.052 | 0.234 |
| Llama-3.1-8B-Instruct | 8.03B | instruct | yes | 0.657 | 0.206 | -0.001 | 0.726 | 0.052 | 0.235 |
| Qwen3-8B-Base | 8.19B | **base** | no | **0.137** | 0.848 | 0.015 | 0.682 | 0.429 | 0.653 |
| Qwen3-8B | 8.19B | post-trained | no | 0.403 | 0.542 | 0.008 | 0.713 | 0.260 | 0.460 |
| Mistral-Small-24B-Instruct-2501 | 23.57B | instruct | yes | 0.596 | 0.291 | -0.000 | 0.648 | 0.127 | 0.418 |
| Olmo-3.1-32B-Instruct | 32.23B | instruct | no | 0.859 | 0.050 | -0.000 | 0.846 | 0.020 | 0.094 |
| Qwen3-32B | 32.76B | post-trained | no | 0.657 | 0.268 | -0.001 | 0.673 | 0.340 | 0.609 |

"去掉位置 0"版本见 `results/summary.md`：所有 12 个模型最后一层各指标变化 ≤ 0.014（最大为 gpt2-medium 的原始 B，其余 ≤ 0.007），不改变上表任何结论。

![fig1](figs/fig1_last_layer_isotropy.png)

### 2.2 Final norm 分解（最后一层）

格式：A isotropy / B 原始跨序列余弦（全部位置）。

| model | norm | 偏置 β | 共享 emb | x（norm 前） | x̂ | γ⊙x̂ | γ⊙x̂+β（= 2.1 表） | post − β | max/median \|γ\| |
|---|---|---|---|---|---|---|---|---|---|
| gpt2 | LayerNorm | yes | yes | 0.153 / 0.825 | 0.153 / 0.825 | 0.076 / 0.923 | 0.068 / 0.931 | 0.076 / 0.923 | 13.9 |
| gpt2-medium | LayerNorm | yes | yes | 0.212 / 0.772 | 0.212 / 0.772 | 0.335 / 0.657 | 0.319 / 0.674 | 0.335 / 0.657 | 12.7 |
| gpt2-large | LayerNorm | yes | yes | 0.299 / 0.649 | 0.297 / 0.651 | 0.727 / 0.140 | 0.708 / 0.163 | 0.727 / 0.140 | 1.6 |
| gpt2-xl | LayerNorm | yes | yes | 0.416 / 0.545 | 0.318 / 0.630 | 0.740 / 0.130 | 0.723 / 0.149 | 0.740 / 0.130 | 1.5 |
| Qwen2.5-7B-Instruct | RMSNorm | no | no | 0.674 / 0.206 | 0.674 / 0.206 | 0.600 / 0.293 | 0.600 / 0.293 | – | |
| Llama-3.1-8B | RMSNorm | no | no | 0.687 / 0.166 | 0.687 / 0.166 | 0.703 / 0.140 | 0.703 / 0.140 | – | 1.2 |
| Llama-3.1-8B-Instruct | RMSNorm | no | no | 0.635 / 0.239 | 0.635 / 0.239 | 0.657 / 0.206 | 0.657 / 0.206 | – | 1.2 |
| Qwen3-8B-Base | RMSNorm | no | no | 0.340 / 0.613 | 0.340 / 0.613 | 0.137 / 0.848 | 0.137 / 0.848 | – | 5.2 |
| Qwen3-8B | RMSNorm | no | no | 0.391 / 0.557 | 0.391 / 0.557 | 0.403 / 0.542 | 0.403 / 0.542 | – | 5.2 |
| Mistral-Small-24B-Instruct-2501 | RMSNorm | no | no | 0.682 / 0.180 | 0.682 / 0.180 | 0.596 / 0.291 | 0.596 / 0.291 | – | |
| Olmo-3.1-32B-Instruct | RMSNorm | no | no | 0.857 / 0.053 | 0.857 / 0.053 | 0.859 / 0.050 | 0.859 / 0.050 | – | |
| Qwen3-32B | RMSNorm | no | no | 0.703 / 0.199 | 0.703 / 0.199 | 0.657 / 0.268 | 0.657 / 0.268 | – | |

γ 的 max/median 直接从 safetensors 里的 final norm 权重读出，只对 GPT-2 系列和两对 base/post-trained 模型算了（空格 = 未算）。
"共享 emb"= 模型加载后输入 embedding 与 LM head 是同一个张量（GPT-2 系列是；现代模型的 `tie_word_embeddings` 都是 False）。
含 top-1 方差和 β 的统计见 `results/summary.md` 末尾。

![fig5](figs/fig5_final_norm.png)

### 2.3 随深度变化

GPT-2 系列和现代模型分开画（每图 ≤ 8 条线）：

![fig2 gpt2](figs/fig2_depth_gpt2.png)
![fig2 modern](figs/fig2_depth_modern.png)
![fig3 gpt2](figs/fig3_centered_spectrum_gpt2.png)
![fig3 modern](figs/fig3_centered_spectrum_modern.png)
![fig4 gpt2](figs/fig4_pos0_outliers_gpt2.png)
![fig4 modern](figs/fig4_pos0_outliers_modern.png)

逐层数值见 `results/summary.md`，机器可读版见 `results/summary.csv`、`results/<model>.json`、`results/diag/<model>.json`。

## 3. 观察

### 3.1 与原论文的对照
- GPT-2 small 最后一层 isotropy = 0.068（50 条冒烟测试为 0.080），明显低于 0.25，与 Su & Collier 报告的"GPT-2 small 各向异性"一致。
- GPT-2 系列随规模：small 0.068 → medium 0.319 → large 0.708 → xl 0.723。分界在 medium 与 large 之间。medium 在本设置下略高于 0.25，但它的中间层 isotropy（0.19–0.33）是 12 个模型里最低的，最后一层 top-1 方差占比 0.92、离群维度比 630，都与 small 同类；large 和 xl 则在所有这些指标上都与现代模型同类。
- 原论文的结论"各向异性不是自回归 LM 的普遍性质"在这里成立，但**第 1 轮报告里"现代模型最后一层都高于 0.25"的说法不再成立**：Qwen3-8B-Base 最后一层 isotropy 只有 0.137，与 GPT-2 small 同一量级。

### 3.2 GPT-2 final LayerNorm 的偏置 β：不是主要原因
- 去掉 β 后，4 个 GPT-2 模型的原始跨序列余弦只变化 0.008–0.023（small 0.931 → 0.923），isotropy 变化 ≤ 0.02。β 的范数约为平均 token 向量范数的 5%（small/medium）到 21%（large/xl），且与平均 token 方向的余弦为 0.47–0.89，但它对 A、B 的贡献都很小。
- 该假设被否定，但 final norm 本身确实起作用：**作用来自逐维缩放 γ，而不是 β**（见 3.3）。
- 另外，norm 之前的最后一个 block 输出在 GPT-2 small 已经相当集中（x 的原始余弦 0.825，isotropy 0.153，而第 10 层只有 0.499 / 0.452）。第 1 轮报告"GPT-2 的各向异性只出现在最后一层"的说法不够准确：选中的层里没有第 11、12 个 block 的输出，集中至少从最后一个 block 已经开始。

### 3.3 逐维缩放 γ 决定最后一层是变窄还是变宽
- 只归一化（x → x̂）对 A、B 几乎没有影响（RMSNorm 是逐 token 缩放，A、B 严格不变；LayerNorm 的逐 token 去均值只在 gpt2-xl 上有可见作用）。
- γ 的作用方向因模型而异：
  - **变窄**：GPT-2 small（余弦 0.825 → 0.923）、Qwen3-8B-Base（0.613 → 0.848，isotropy 0.340 → 0.137）；Qwen2.5-7B-Instruct、Mistral、Qwen3-32B 也有小幅变窄（+0.07 – +0.11）。
  - **变宽**：gpt2-large（0.651 → 0.140）、gpt2-xl（0.630 → 0.130）、gpt2-medium（0.772 → 0.657）；Llama 两个版本小幅变宽。
  - 基本不变：Qwen3-8B、Olmo-3.1。
- γ 的尖峰程度（max/median |γ|）与结果一致：GPT-2 small/medium 为 13.9 / 12.7（各向异性），large/xl 为 1.6 / 1.5（各向同性）；Qwen3-8B 为 5.2，Llama-3.1-8B 为 1.2。在 GPT-2 small/medium 和 Qwen3-8B(-Base) 里，最后一层的离群维度正好就是 |γ| 最大的那几维（例如 GPT-2 small 的 496、430、314；Qwen3 的 2284、2202、3169），且与前一个选中层的离群维度不同。**（第 3 轮补充：这些维度在乘 γ 之前就已是均值主导的维度，离群不是只由 γ 造成，见 R3.2、R3.5。）**
- 注意：large/xl 在 norm 之前也相当集中（余弦 0.65 / 0.55），它们最后一层的各向同性是 γ 压平之后的结果，不代表最后一个 block 的输出本身是各向同性的。

### 3.4 共享输入/输出 embedding：不能解释本次的差异
- 4 个 GPT-2 模型全都共享 embedding，但 large/xl 最后一层是各向同性的（0.708 / 0.723）。
- 所有现代模型都不共享（`tie_word_embeddings=False`，加载后也确认输入 embedding 与 LM head 不是同一张量），但 Qwen3-8B-Base 最后一层是各向异性的（0.137），Qwen3-8B 也是现代模型中最低的（0.403）。
- 所以在本次 12 个模型里，共享 embedding 既不是最后一层各向异性的必要条件，也不是充分条件。这不否定 Gao 等人（2019）关于共享 embedding 训练出各向异性**词嵌入矩阵**的结论；本次测的是上下文隐藏表征。

### 3.5 base vs post-trained
- **Qwen3-8B**：base 0.137 → post-trained 0.403。差异几乎全部在最后一步：两者中间层曲线接近（base 0.48–0.62，post 0.43–0.66），norm 之前的 x 也接近（0.340 vs 0.391）；差别在 γ 这一步：base 被压到 0.137，post 保持 0.403。两者的 γ 几乎相同（‖Δγ‖/‖γ‖ = 0.3%，余弦 1.0000），所以差别来自 norm 输入 x 在大 γ 维度上的分布：最后一层最大离群维度（2284）的 mean|h| 从 156 降到 61。**（第 3 轮修改：只平移这一维只弥合 19% 的差距，需要约 8 个维度，见 R3.2、R3.5。）**
- **Llama-3.1-8B**：base 0.703 → instruct 0.657，各层曲线几乎重合（中间层 0.66–0.79 vs 0.63–0.78），γ 同样几乎没变（0.2%）。
- 只有两对，方向也不一致（Qwen3 post-training 提高 isotropy，Llama 略降），不能据此推断"后训练"对几何的一般影响。

### 3.6 位置 0 / 均值偏移的影响（第 1 轮结论，新模型同样成立）
- 位置 0 在中间层的范数远大于其他位置：中位比值峰值 Qwen2.5-7B 402×、Qwen3-32B 218×、Qwen3-8B 199×、Llama-Instruct 175×、Llama base 157×、Qwen3-8B-Base 134×、Mistral 119×、GPT-2 系列 25–42×。位置 0 是 BOS（Llama、Mistral）还是普通文本 token（Qwen、GPT-2）都有。**例外：Olmo-3.1-32B 全程不超过 1.7×。**最后一层（final norm 后）所有模型的比值回到 0.3–1.1。
- 对 **A 和原始 B** 几乎没有影响：12 个模型所有层，去掉位置 0 后 A 的变化 ≤ 0.012，原始 B 的变化 ≤ 0.014（gpt2-medium；其余模型 ≤ 0.006）。
- 对 **中心化 B 和 C（谱）影响很大**，尽管位置 0 只占 bank 的 0.6%：例如 Qwen3-32B 第 8 层中心化余弦 0.763 → 0.008，RankMe/d 0.102 → 0.664；Llama-Instruct 第 4 层中心化余弦 0.591 → 0.000，RankMe/d 0.196 → 0.784。即中间层"中心化后仍各向异性 / 低有效秩"的读数主要来自少数大范数 token。最后一层不受影响。

### 3.7 其他未解释的现象
- Qwen2.5-7B-Instruct 去掉位置 0 后，第 4–21 层仍然 top-1 方差占比 0.97–1.00、RankMe/d 只有 0.15–0.39，第 4 层中心化余弦 0.34。Qwen3-8B（第 18、22 层，top-1 0.88 / 0.77）和 Qwen3-32B（第 48 层，0.82）也有类似的局部尖峰。推测 bank 里还有其他位置的大范数 token，但**本次没有验证是哪些 token**。
- Olmo-3.1-32B 在所有指标上都不同于其他现代模型（全程 isotropy 0.77–0.86、无位置 0 范数尖峰、离群维度比 ≤ 21），原因未查。

## 第三轮：γ 折算、逐维分解、功能性干预

Slurm array 9769（12 个任务，`%2`，每任务 1 张 B300；gpt2 与 Qwen3-8B-Base 直接沿用 srun 冒烟测试的完整输出，任务内跳过）和 CPU 任务 9770（RTXq，无 `--gres`，词频统计），全部 COMPLETED。代码：`round3_lib.py`（共享函数）、`extract_round3.py`（GPU：步骤 0、步骤 1 的折算测试与 V0–V3 指标、步骤 3）、`analyze_round3.py`（登录节点：步骤 2、3b、汇总与图）、`count_unigrams.py`。完整数值见 `results/round3/summary.md` 与 `results/round3/analysis.json`。

与任务书的一处出入：步骤 1 的 V0–V3 指标（含 10000×d 的 SVD）放在 GPU 任务里用 `round3_lib.py` 计算，`analyze_round3.py` 读取结果，没有在登录节点重算。步骤 2 的维度排名在 GPU 任务（供步骤 3 使用）和 `analyze_round3.py` 中各算一次，12 个模型的前 8 维集合完全一致。

### R3.0 预期与结果对照

| # | 预期（假设） | 实际结果 | 是否一致 |
|---|---|---|---|
| 0a | 由 x 重建的 V3 与 `hidden_states[-1]` 相对误差 < 1e-2 | 12 个模型 1.7e-3 – 2.5e-3 | 一致 |
| 0b | bank 估计的 A 与第二轮全序列 A 相差 < 0.01；‖ū‖² 与第二轮 B 相差 < 0.01 | V0–V3 全部通过；最大 dA = 0.0066（gpt2-medium），最大 dB = 0.0037 | 一致 |
| 1a | γ 折算：最大相对误差 < 1e-4，argmax 一致率 ≥ 99.99% | 最大相对误差 ≤ 3.1e-6，argmax 一致率 100%（12/12） | 一致 |
| 1b | RMSNorm 模型 V0 与 V1 的 A、B 数值一致 | 8 个 RMSNorm 模型最大差 1.2e-7 | 一致 |
| 1c | GPT-2 small/medium、Qwen3-8B-Base：V1 的 isotropy 明显高于 V3 | small 0.153 vs 0.068（高，但 V1 本身已很低）；**medium 0.212 vs 0.319（V1 反而更低）**；Qwen3-8B-Base 0.340 vs 0.137 | 部分一致（2/3） |
| 1d | GPT-2 large/xl：V1 与 V3 差别不大，或 V3 更高 | 0.297 → 0.708，0.318 → 0.723（V3 高得多，γ 起稀释作用） | 一致 |
| 1e | （条件判读）V1 与 V3 下 12 个模型排序的 Spearman | isotropy ρ = 0.48（p = 0.09），原始 B ρ = 0.59（p = 0.02） | 排序明显改变 |
| 2a | 近似公式与精确值的误差 | −0.035 – +0.018（GPT-2 small/medium 最大，约 −0.035） | （无预设） |
| 2b | 三个各向异性模型的前 8 维贡献 > ‖ū‖² 的一半 | 0.999 / 0.998 / 0.904 | 一致（但另有 4 个非各向异性模型也 > 0.5，见 R3.2） |
| 2c | γ 换成常数后的余弦 = V1 的结果 | 12 个模型差值均为 0 | 一致 |
| 2d | 循环论证检查：前几维在 V1 中是否特殊 | 12 个模型中，V3 前 10 维在 V1 的均值占比 μ²/(μ²+σ²) 的中位百分位都 ≥ 0.978 | 这些维度在 V1 中就已特殊，**不是完全由 γ 造成** |
| 2e | Qwen3 base→instruct：k ≤ 8 弥合差距 > 70% | k=1: 19%，k=2: 32%，k=4: 63%，**k=8: 97%**，k=16: 108% | 一致（k = 8 时） |
| 2f | Llama 对照 | 差距方向相反且小（0.140 → 0.206）；k=16 也只弥合 25% | 作为对照：Llama 的差异不集中在前几维 |
| 3 | bite / bark（不预设阈值） | 12 个模型中，k ≥ 2 时前 k 维的 Δloss 都超过随机对照的最大 \|Δloss\|（2× – 7900×）。绝对值差异很大：Qwen3-8B-Base k=8 为 +4.96 nats/token；gpt2-large/xl、Olmo 在 k ≤ 32 时 ≤ 0.07 | 因模型而异，见 R3.3 |
| 3 | 线性捷径与直接计算一致 | 相对 logit 差 ≤ 3.4e-6 | 一致 |
| 3b | W_U[:,S] m_S 与 log 词频显著正相关 | 全部维度：12 个模型 ρ = 0.57 – 0.66；前 k 维：72 项中 70 项为正，2 项为接近 0 的负值 | 一致，但不限于各向异性模型 |

### R3.1 γ 折算之后还剩多少各向异性（步骤 1）

![r3 fig1](figs/r3_fig1_isotropy_V0-V3.png)

- 把 γ 折进 LM head（`W_U' = W_U·diag(γ)`，`b' = W_U β`）在数值上完全不改变模型输出，因此"最后一层"可以合法地取 V1 而不是 V3。GPT-2 的折算会解除输入/输出 embedding 的共享（只改输出头的副本）。
- 以 V1 计，三个各向异性模型变成：GPT-2 small 0.153（仍很低：残差流本身的公共分量已经很强，γ 只是把它从 0.153 放大到 0.068）；gpt2-medium 0.212（**比 V3 的 0.319 还低**，γ 在 medium 上起的是稀释作用，与第二轮"γ 使 medium 变宽"一致）；Qwen3-8B-Base 0.340（γ 把它压到 0.137）。
- GPT-2 large/xl 在 V1 下只有 0.297 / 0.318，与 GPT-2 small/medium、Qwen3-8B-Base 同属最低的一组；它们在 V3 下的各向同性（0.71 / 0.72）来自 γ。
- 12 个模型在 V1 与 V3 下的 isotropy 排序 Spearman ρ = 0.48。**"哪个模型更各向异性"取决于尺度如何在 γ 与 W_U 之间分配**，而这种分配不影响模型输出。前两轮基于 `hidden_states[-1]` 的跨模型比较都应在这个前提下理解。
- 稳健中心化 B（估计均值时剔除位置 0 和范数 > 10× 中位数的 token）在所有模型、所有 V 上都 ≤ 0.043，即各向异性几乎全部是一个公共均值方向。RankMe/d 除以同 N、d 的高斯参照后，V1 在 0.75 – 0.92 之间；GPT-2 small/medium 的 V2/V3 只有 0.32 / 0.21。

### R3.2 逐维分解（步骤 2）

![r3 fig2](figs/r3_fig2_cum_contrib.png)

- ‖ū‖² 精确分解：GPT-2 small 前 1 维（496）贡献 75%，前 4 维 99%；gpt2-medium 前 1 维 82%；Qwen3-8B-Base 分布更开：前 1 维 33%，前 8 维 90%。
- 前 8 维占比 > 0.5 的还有 Qwen2.5-7B-Instruct（0.84）、Qwen3-8B（0.88）、Qwen3-32B（0.82）、Mistral（0.66），但它们的 ‖ū‖² 本身较小（0.27 – 0.54）。即"集中在少数维度"不是各向异性模型特有的，各向异性的大小来自这些维度上均值的绝对大小。GPT-2 large/xl、Llama 两个版本、Olmo 的前 8 维只占 15% – 41%。
- 近似公式 Σm²/Σ(m²+s²) 与精确值的误差为 −0.035 – +0.018；在 GPT-2 small/medium 上偏低约 0.035（这两个模型的 token 范数差异较大，近似公式假设范数近似相同）。
- **循环论证检查**：在乘 γ 之前的 V1 中，V3 前 10 维的均值占比 μ²/(μ²+σ²) 在所有 12 个模型里都位于第 97.8 – 99.9 百分位（中位数），也就是说这些维度在残差流里本来就是均值主导的维度。γ 的作用因模型而异：GPT-2 small/medium、Qwen 系列、Mistral 中这些维度的 |γ| 也位于第 98.7 – 99.8 百分位（放大）；Llama 两个版本和 gpt2-xl 中只在第 2.7 – 4.4 百分位（压低）。在 GPT-2 small 中，V3 前 10 维只占 V1 的 ‖ū‖² 的 28%（与 V1 自己的前 10 维重合 3 个），γ 把锥集中到了这几维上；在 Qwen3-8B-Base 中，同样的 10 维已占 V1 的 54%（重合 8 个）。
- **base→instruct 反事实**：把 Qwen3-8B-Base 的 V1 在前 k 维的均值平移到 Qwen3-8B 的值（γ 用 base 的），预测余弦从 0.847 降到 0.788（k=1）、0.655（k=4）、0.553（k=8，弥合 97%）、0.519（k=16，超过 instruct 的 0.542）。Llama 对照：差距小且方向相反，k=16 也只弥合 25%。

### R3.3 功能性干预（步骤 3）

![r3 fig3](figs/r3_fig3_dloss.png)

前 500 条序列估计 `hidden_states[-1]` 的各维均值 m（剔除位置 0），在后 500 条（各 tokenizer 9.0 万 – 9.5 万个预测 token）上评估 next-token loss。由于 `logits' = logits − W_U[:,S] m_S`，去掉均值在功能上只等价于一个固定的 logit 偏置，不携带随 token 变化的信息；本节测的是这个偏置有多重要。

| model | 基线 loss | 前 1 维 | 前 8 维 | 前 32 维 | 全部维度 | k=8 相对随机最大值 |
|---|---|---|---|---|---|---|
| gpt2 | 3.775 | +0.0001 | +0.056 | +2.128 | +3.113 | 29× |
| gpt2-medium | 3.460 | +0.004 | +0.118 | +1.687 | +2.973 | 91× |
| gpt2-large | 3.310 | +0.001 | +0.010 | +0.069 | +2.568 | 6× |
| gpt2-xl | 3.220 | +0.001 | +0.005 | +0.047 | +2.512 | 5× |
| Qwen2.5-7B-Instruct | 2.509 | +0.318 | +0.586 | +0.538 | +2.121 | 1280× |
| Llama-3.1-8B | 2.351 | +0.046 | +0.460 | +0.562 | +2.582 | 984× |
| Llama-3.1-8B-Instruct | 2.461 | +0.047 | +0.285 | +0.316 | +2.301 | 255× |
| Qwen3-8B-Base | 2.444 | +0.587 | **+4.956** | +4.480 | +3.131 | 2389× |
| Qwen3-8B | 2.909 | +0.012 | +0.376 | +0.989 | +2.438 | 515× |
| Mistral-Small-24B | 2.257 | +0.075 | +0.701 | +1.068 | +2.093 | 1560× |
| Olmo-3.1-32B | 2.792 | +0.002 | +0.010 | +0.023 | +1.269 | 36× |
| Qwen3-32B | 2.687 | +0.085 | +0.519 | +0.596 | +2.459 | 506× |

（Δloss 单位 nats/token；随机对照 = 从非前 32 维中随机抽 k 维，5 个 seed。"相对随机最大值" = 前 k 维 Δloss ÷ 5 次随机抽取中最大的 |Δloss|，因为随机均值常接近 0，直接相除不稳定。）

- 去掉**全部维度**的均值在所有 12 个模型上都使 loss 增加 1.27 – 3.13 nats/token，包括最后一层各向同性的模型（Olmo +1.27，gpt2-large +2.57）。均值偏置在每个模型里都承担着重要的输出先验，与锥的宽窄无关。
- 去掉**前 k 维**：Qwen3-8B-Base、Qwen2.5、Mistral、Qwen3-32B、Llama、Qwen3-8B 在 k ≤ 8 时就有 0.28 – 4.96 nats 的增加（bite）；GPT-2 large/xl 和 Olmo 在 k ≤ 32 时 ≤ 0.07 nats（在这些模型里，均值偏置分散在很多维度上，前几维不重要）。
- **余弦贡献与功能重要性可以分离**：GPT-2 small 的第 496 维贡献了 75% 的 ‖ū‖²，但去掉它的均值只增加 0.0001 nats。原因是它造成的 logit 偏移几乎是常数：在出现 ≥ 5 次的 token 上，偏移的 |均值| = 41.3，标准差只有 0.15，而 softmax 对常数偏移不敏感。全部 84 个"模型 × 维度集合"组合上，Spearman(偏移在词表上的标准差, Δloss) = 0.95。也就是说，决定功能代价的是偏移在词表上的**差异**，而不是它对余弦的贡献。
- Δloss 随 k 不单调：Qwen2.5 在 k=4 时 +1.74，k=8 反而 +0.59；Qwen3-8B-Base k=8 的 +4.96 大于去掉全部维度的 +3.13。去掉一部分维度的均值可能破坏不同维度偏置之间的相互抵消。

### R3.4 词频（步骤 3b）

在 wikitext-103 train 的前 9000 万字符（各 tokenizer 约 1970 万 – 2100 万 token，6 组 tokenizer）上统计词频，只用出现 ≥ 5 次的 token（约 4 万种）。
- 去掉全部维度均值造成的 logit 偏移 W_U m 与 log 词频的 Spearman ρ 在 12 个模型上为 0.57 – 0.66（n ≈ 4 万，p 值在浮点精度下为 0）。
- 前 k 维的偏移（12 个模型 × 6 个 k = 72 项）：70 项为正（0.105 – 0.623）；Olmo 前 1 维（−0.038）和 Qwen3-32B 前 4 维（−0.017）接近 0。
- 支持"均值方向编码词频先验"的经典解释，但这一相关在各向同性的模型（Olmo、gpt2-large/xl）上同样存在，因此它解释的是均值偏置的**内容**，而不是为什么有的模型在余弦上表现为窄锥。

### R3.5 对前两轮结论的修改

- 第二轮 3.3"最后一层的离群维度正好就是 |γ| 最大的那几维"：仍然成立，但**补充**：这些维度在乘 γ 之前就已是均值主导的维度（R3.2 循环论证检查），离群是"残差流中的强均值 × γ 放大"共同造成的，而不是只由 γ 造成。
- 第二轮 3.5"最大离群维度（2284）的 mean|h| 从 156 降到 61"作为 Qwen3-8B base→post-trained 差异的解释：**修改为**需要约 8 个维度。只平移这一维只能弥合 19% 的差距，前 8 维弥合 97%（R3.2 反事实）。
- 第一、二轮对各模型 `hidden_states[-1]` isotropy 的跨模型比较：数值本身不变，但**补充前提**：把 γ 折进 LM head 后排序会明显改变（ρ = 0.48，R3.1），这些比较依赖于该表示的尺度约定。

## 4. 局限

- 模型集合不是设计出来的对照：只有两对 base/post-trained；参数量集中在 0.1–1.6B（GPT-2）、7–8B 和 24–33B；每个现代家族最多 3 个点。
- 数据是 wikitext-103 test（≥100 词的段落只有 1085 段，1000 段样本几乎是全集），与原论文不同；截断长度 256。
- 所有模型都用 bf16 前向，原论文可能用 fp32。
- B、C 基于每层 10000 个 token 的 bank、20000 对，单 seed，没有置信区间。
- γ 折算后排序会变（第 3 轮 ρ = 0.48），所以对 `hidden_states[-1]` 做跨模型比较依赖于尺度约定。
- 第 3 轮的功能性干预只测了"去掉均值"这一种：它等价于固定的 logit 偏置，测的是输出先验的重要性，不涉及随 token 变化的信息；base→instruct 反事实只平移均值、不改方差，且只有两对模型。

## 5. 可能的下一步（未做）

1. 定位 Qwen2.5 / Qwen3 中去掉位置 0 后仍存在的大范数 token，并给出"去掉所有大范数 token"的 B/C。
2. 补 Qwen3 系列的其他规模（0.6B–14B 的 Base），看 γ 尖峰与最后一层各向异性是否在同一家族内随规模变化。
3. 查 Olmo-3.1 为何没有位置 0 尖峰（架构上 Olmo 系列的 norm 位置与其他模型不同，可作为起点）。
4. 是否纳入 GLM-4.5-Air-FP8（FP8 量化 MoE，因精度与协议不符而排除）。

## 文件

`prepare_data.py`、`measure.py`、`diag_final_norm.py`、`analyze.py`、第 3 轮的 `round3_lib.py`、`extract_round3.py`、`analyze_round3.py`、`count_unigrams.py`、`slurm/run_round3.sbatch`、`slurm/unigrams.sbatch`、`results/round3/`、`cache/banks/`（每个模型 fp32 bank，已加入 `.gitignore`）、`tests/test_metrics.py`、`slurm/run_array.sbatch`（+`slurm/env.sh`；已完成的步骤会自动跳过）、`models.txt`、`models_local.md`、`results/*.json`、`results/diag/*.json`、`results/summary.{md,csv}`、`figs/*.png`、`logs/iso_97{25,43}_*.out`。冒烟测试（各 50 条）的结果在 `results/smoke/`。
