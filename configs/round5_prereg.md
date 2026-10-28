# IsoProbe 第五轮预登记

写于任何第五轮数据分析之前（2026-10-29）。第一部分是任务书第 1–3 项的原文；第二部分是实现这些检验所需、而原文没有写明的操作化细节（由 Claude 根据现有代码与 Godey 等人原文补充）。标注"待确认"的条目在得到确认之前不做判定；确认后的修改会作为单独的 commit 记录在本文件末尾的"修订记录"中，并且仍在任何结果分析之前完成。之后的分析代码只执行本文件定义的检验，其余分析在报告中标注为"探索性"。

---

## 第一部分：任务书原文

### 第 1 项：A2' 的样本外检验（只用第四轮已有的 91 个 checkpoint，不下载新模型）

背景：第四轮在 12 个模型上发现，ΔSTS(V1 − V3) 随 γ 对 top-1 方差占比的改变而变化（ρ = 0.71 – 0.90）。但这个关系是看到数据之后才发现的。本项在训练 checkpoint 上检验它。

- 样本集合（预登记）：第四轮的 91 个 checkpoint 中，max/median |γ| ≥ 1.01 的那些（排除 γ 基本未训练的早期点，因为那时 V1 与 V3 只差 β，结果近乎平凡）。
- 对每个 checkpoint 跑 `sts_round4.py` 的流程：STS-B test 和 SICK-R test；mean 池化与 PromptEOL；V1、V3 各自的 raw、center、zscore 三种变体。不需要层扫描和 abtt。
- 正确性检查：zscore 下 V1 与 V3 的差值必须为 0（同第四轮 A1）。
- 预测量：Δtop1 = top1(V3) − top1(V1)。必须使用与第四轮 A2' 完全相同的字段和定义：查看 analyze_round4.py 实际用的是哪一个，在预登记文件里写明。数值直接取自第四轮 dyn 的 JSON。
- 主要终点（只有这一个）：STS-B、mean 池化、raw 余弦的 ΔSTS(V1 − V3) × 100。
- 检验：
  - 对 5 个模型（4 个 Pythia + OLMo-2）分别计算 Spearman(ΔSTS, Δtop1)。
  - 合并检验用按模型分层的置换检验：只在每个模型内部打乱 ΔSTS，重复 10000 次，统计量为 5 个模型 ρ 的平均。
  - 同一模型的 checkpoint 不独立，所以不要把所有点合并后直接算 p 值。
- 预测：
  - P1-1：平均 ρ > 0，且置换检验 p < 0.05。
  - P1-2：在至少 4 个模型上，|ρ(ΔSTS, Δtop1)| > |ρ(ΔSTS, A_V1 − A_V3)|，即方差集中度比均值方向更能预测 ΔSTS。
  - 次要终点（SICK-R、PromptEOL）只报告，不作判定。

### 第 2 项：与 Godey 等人（2024）"饱和"现象的联系

背景：Godey 等人报告 Pythia ≤ 410M 在训练后期 loss 上升（"饱和"），≤ 160M 同时出现最后一层各向异性陡增，LM head 的奇异值谱坍缩成一个尖峰，并归因于 softmax bottleneck。第四轮发现 70m/160m 的终点坍缩发生在残差流（V1）本身，方向分散在许多维度上，且评估 loss 同时上升。本项用加密的 checkpoint 判断先后顺序和方向关系。

开始前请用 Undermind connector 读 Godey 等人（arXiv 2404.07647）原文，核对他们的奇异熵定义（基于 σ 还是 σ²，以及归一化方式），按原文实现。如果无法确认，就同时报告基于 σ 和 σ² 的两个版本。

加密下载（登录节点；Pythia 每 1000 step 有一个 branch）
- pythia-70m：step 50000 – 143000，每 1000 step 一个
- pythia-160m：step 80000 – 143000，每 1000 step 一个
- pythia-410m：step 80000 – 142000，每 2000 step 一个，另加 143000
- pythia-1.4b（不饱和的对照）：step 80000 – 140000，每 5000 step 一个，另加 143000
- 已经下载过的 checkpoint 跳过。下载前估算总大小并检查配额。

每个 checkpoint 的测量
- 新 checkpoint：跑第四轮 dyn 的全部测量，再加下面的新指标。
- 第四轮已有的 76 个 Pythia checkpoint：只补新指标。能在 CPU 上用 cache 中的 bank 和 safetensors 中的 W_U 算的就在 CPU 上算；需要 logits 的部分做一次只前向评估集的轻量 GPU 计算。
- 新指标：
  1. W_U 谱（只用前 50277 行，fp32 SVD）：σ₁²/Σσ²、σ₁/σ₂、按 Godey 定义的奇异熵（或 KL 到均匀分布）、有效秩。
  2. 方向对齐：V3 的均值方向 ū（L2 归一化 token 的均值，取方向）与 W_U 第一右奇异向量的 |cos|；V1 的 ū 与折算后头部 W_U·diag(γ) 第一右奇异向量的 |cos|（这一对与 γ 折算相容）。另外报告 ū 落在 W_U 前 1 / 5 / 10 个右奇异向量张成的子空间里的平方占比。随机基线：|cos| 约为 √(2/(πd))。
  3. Δloss 的预测量：对第四轮的每个 δ 集合（all、top-1/8/32），在评估位置上计算 ½·E_t Var_{p_t}(δ)，其中 p_t = softmax(L_t)，Var_{p_t}(δ) = Σ_v p_t(v) δ_v² − (Σ_v p_t(v) δ_v)²。与实际 Δloss 并列报告。
  4. 评估 loss：沿用第四轮的 loss_base（后 500 条序列）。

预登记的问题与判定方式
- Q2-1（先后顺序，不预设方向）：对 70m 和 160m 的加密窗口，计算以下四个量的 t50，即"首次达到从窗口起点到最终 checkpoint 总变化量的 50%"的 step：评估 loss（窗口内最小值之后的上升）；W_U 的 σ₁²/Σσ²；V1 的 ‖ū‖²；‖β‖/‖γ‖。比较它们的先后，分辨率为 1000 step。候选假设：H-head（头部谱尖峰先出现）、H-rep（残差流坍缩先出现）、H-sim（在分辨率内同时）。只报告结果，不预设哪个正确。
- Q2-2（方向）：预测 70m/160m 坍缩之后，两种 |cos| 都远高于随机基线（> 0.5），即残差流的公共方向与头部的主奇异方向对齐；410m/1.4b 在同一时期保持低对齐（< 0.2）。
- Q2-3（410m）：Godey 等人报告 410M 也会饱和。预测 410m 在窗口内评估 loss 会上升，但 V1 不发生坍缩（V1 isotropy 不低于 0.4），即饱和可以在没有各向异性坍缩的情况下发生。
- Q2-4（Δloss 预测量）：在所有"checkpoint × δ 集合"（含第三轮的 12 个模型，见下）上，Spearman(½·E_t Var_{p_t}(δ), Δloss) 高于第三轮"词表上的 sd"得到的 0.95；并且第四轮 70m step 143000"corr 为正但子集 Δloss 更大"的矛盾，在新预测量下应该消失（子集的预测量也大于全部维度的预测量）。
- 第三轮 12 个模型的 Δloss 预测量：用 extract_round3.py 的步骤 3 部分补算（只需前向评估集，每个模型几分钟），放在同一个 array 里。

### 第 3 项：早期瞬态是否就是"unigram 阶段"

背景：所有 Pythia 在 step 16 – 64 都有一次强各向异性（A 降到 0.08 – 0.13），而那时 γ 仍全为 1。假设：这是模型主要靠一个公共均值输出词频先验的阶段。

- 补下载四个 Pythia 规模的 step 2、8、32、128（第四轮缺这几个对数间隔点）。
- 词频：用不申请 GPU 的 sbatch 任务，在 wikitext-103 train 的前 9000 万字符上，用 Pythia 的 tokenizer 统计词频（复用 count_unigrams.py，新增一个 tokenizer 组）。同时在后 500 条评估序列上计算 unigram 模型的交叉熵 H_uni（用 add-1 平滑的 train 词频）。
- 对 step 0 – 2000 的每个 checkpoint 测量：Spearman(W_U m, log 词频)（出现 ≥ 5 次的 token）；常数 logit 占比 S_const = ‖L̄ − mean_v(L̄)‖² / E_t ‖L_t − mean_v(L_t)‖²，其中 L̄ 是 logits 在评估位置上的平均（= W_U m + 偏置）。S_const 接近 1 表示各位置的输出几乎相同；评估 loss − H_uni；以及第四轮已有的 A 和 ‖ū‖²。
- 预测：
  - P3-1：在 step 0 – 1000 内，A 的最小值所在的 checkpoint，与 S_const 的最大值、Spearman(W_U m, log 词频) 的最大值落在同一个或相邻的 checkpoint 上。
  - P3-2：在这个 checkpoint 上，评估 loss 与 H_uni 的差距是 step 0 – 1000 内最小的（|loss − H_uni| 最小）。
  - P3-3：到 step 1000，S_const 明显下降，A 回升（第四轮已观察到 A 回升到 0.69 – 0.72）。

---

## 第二部分：操作化细节

### 第 1 项

- **Δtop1 的字段。** 第四轮 A2'（`analyze_round4.py` 第 79 行）用的是 `step1[V]["all"]["top1_var"]`，即 `round3_lib.bank_metrics_r3` 中 `measure.spectrum` 的 `top1_var`：对 bank（10000 个 token，全部位置）减去稳健均值（`round3_lib.robust_mean`：剔除位置 0 与范数 > 10× 中位数的 token 后的均值）后做 SVD，取 σ₁² / Σσ²。第四轮 dyn JSON（`results/round4/dyn/<model>/<rev>.json`）的 `step1[V]["all"]["top1_var"]` 由同一个函数计算。Δtop1 = `step1["V3"]["all"]["top1_var"] − step1["V1"]["all"]["top1_var"]`。
- **A_V1 − A_V3**：`step1["V1"]["all"]["isotropy_full"] − step1["V3"]["all"]["isotropy_full"]`（与第四轮 A2 相同）。
- **样本集合**：`gamma_beta.gamma_max_over_median ≥ 1.01`，按第四轮 dyn JSON 计算，共 59 个 checkpoint：pythia-70m 12 个（step 1000 – 143000）、pythia-160m 12 个（step 1000 – 143000）、pythia-410m 11 个（step 2000 – 143000）、pythia-1.4b 11 个（step 2000 – 143000）、OLMo-2-0425-1B 13 个（stage1 step 10000 起至 main）。
- **STS 流程**：`sts_round4.py` 原实现（V1 per-token、fp64 的 V3 与池化、同一拟合集与模板）；只加两个开关：只算 raw / center / zscore，不做层扫描。bootstrap 照常（1000 次），只报告不判定。
- **ρ**：Spearman（平均秩处理并列），每个模型内对其样本内的 checkpoint 计算。
- **置换检验**：在每个模型内部独立打乱 ΔSTS（Δtop1 不动），10000 次，统计量为 5 个模型 ρ 的平均，随机种子 0。p 值 = (1 + #{置换统计量 ≥ 观测值}) / (1 + 10000)（单侧，**待确认**：方向性预测采用单侧；另报告双侧 p = 用 |平均 ρ| 比较）。
- **P1-2**：对每个模型比较 |ρ(ΔSTS, Δtop1)| 与 |ρ(ΔSTS, A_V1 − A_V3)|，前者严格更大的模型数 ≥ 4 即为成立。

### 第 2 项

- **奇异熵**：Godey 等人（Section 4.2）原文只写"computed as the Kullback-Leibler divergence between the normalized singular value distribution and the uniform distribution"，没有公式，没有说明 σ 还是 σ²，也没有说明是否中心化。因此按任务书同时报告两个版本：KL(p‖u) = log r − H(p)，r = 奇异值个数（= d），p_i = σ_i / Σσ（σ 版）或 σ_i² / Σσ²（σ² 版）；W_U 不中心化。
- **有效秩**：沿用 `measure.spectrum` 的 RankMe 定义 exp(−Σ p log p)，p = σ / Σσ，与前几轮一致。
- **W_U**：safetensors 中的头部权重，前 50277 行，按模型加载时的 bf16 精度转为 fp32 后做 SVD（与第三、四轮 W_U 的数值一致）。
- **ū 与对齐**：ū 用第四轮 bank（`cache/round4_banks/`，新 checkpoint 由新跑的 dyn 生成）的全部 10000 个 token：V3 ū = mean(normalize(V3))，V1 ū = mean(normalize(V1))，取方向（单位化）。|cos(ū_V3, v₁(W_U))|；|cos(ū_V1, v₁(W_U·diag(γ)))|；ū 在前 1 / 5 / 10 个右奇异向量子空间中的平方占比，两对都报告。随机基线 √(2/(π d))。
- **Δloss 预测量**：评估位置与第四轮相同（序列 500 – 999，每条预测位置 1 … T−1），p_t = softmax(L_t)，L_t 用 bf16 的 `hidden_states[-1]` 转 fp32 后乘 fp32 W_U（与第四轮 Δloss 的计算完全相同）。δ 取第四轮 bank 中保存的 δ 向量（整个词表行数，不截断）。冒烟测试中对 1 – 2 个 δ 集合用逐位置的直接求和核对。
- **t50**：
  - 窗口：70m 为 step 50000 – 143000，160m 为 step 80000 – 143000（加密网格上的所有 checkpoint）。
  - 评估 loss：基线 = 窗口内最小值（及其 step）；总变化 = loss(143000) − 最小值；t50 = 最小值之后第一个满足 loss ≥ 最小值 + 0.5 × 总变化的 step。
  - 其他三个量：基线 = 窗口起点的值；总变化 = 终点值 − 起点值；t50 = 第一个越过 起点 + 0.5 × 总变化（沿变化方向）的 step。
  - 先后比较：t50 相同或相差 1000 step 以内（≤ 1000）视为"在分辨率内同时"。
- **Q2-2 的"坍缩之后"与"同一时期"**（**待确认**）：70m / 160m 取各自 V1 ‖ū‖² 的 t50 到 step 143000 之间的全部 checkpoint，要求每一个上两种 |cos| 都 > 0.5；410m / 1.4b 取 step ≥ min(70m 的 t50, 160m 的 t50) 的全部 checkpoint，要求每一个上两种 |cos| 都 < 0.2。另报告 step 143000 单点的结果。
- **Q2-3**：410m 窗口为 step 80000 – 143000；"loss 上升" = loss(143000) > 窗口内最小值，并报告上升幅度；"V1 不坍缩" = 窗口内每个 checkpoint 的 V1 isotropy（`isotropy_full`）≥ 0.4。两者同时满足为成立。
- **Q2-4 的样本**（**待确认**）：所有带新指标的 Pythia checkpoint（第四轮 76 个 + 加密 + 早期补点）× δ ∈ {all, top1, top8, top32}，加上第三轮 12 个模型 × 同样 4 个 δ 集合；判定 = Spearman(½·E_t Var, Δloss) > 0.95。另在同一样本上报告"词表 sd"的 Spearman 作对照（只报告）。矛盾检验：70m step 143000 上子集 Δloss > 全部维度 Δloss 的预登记 δ 集合只有 top32（7.92 > 3.16），判定 = 预测量(top32) > 预测量(all)。
- **第三轮 12 个模型的预测量**：评估集、W_U、m 与 δ 与 `extract_round3.py` 步骤 3 相同（序列 500 – 999），δ 取 `cache/banks/<slug>.pt` 中保存的向量。

### 第 3 项

- **词频组**：Pythia 四个规模共用同一个 tokenizer（按 `count_unigrams.tok_group` 判定，若不同则分组）；与第三轮相同的前 9000 万字符。
- **H_uni**：评估目标与第四轮 loss_base 相同（序列 500 – 999 的位置 1 … T−1），q(v) = (count(v) + 1) / (N + V)，V = 50277（tokenizer 词表），H_uni = −mean log q(y_t)，单位 nats。
- **Spearman(W_U m, log 词频)**：W_U m = 第四轮 bank 中的 δ_all（前 50277 行），只用出现 ≥ 5 次的 token。
- **S_const**：L_t 为评估位置的 logits（前 50277 列），L̄ = mean_t L_t；mean_v 为对这 50277 个词表项求平均；S_const = ‖L̄ − mean_v(L̄)‖² / mean_t ‖L_t − mean_v(L_t)‖²（∈ [0, 1]）。
- **checkpoint 序列**：step 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000（判定窗口 0 – 1000），另画 2000。"相邻" = 在这个有序序列中相邻。
- **P3-1 / P3-2 / P3-3 的判定**（**待确认**）：每个预测在每个 Pythia 规模上分别判定，4 个规模都满足才记为"一致"，否则报告满足的个数。P3-3 的"明显下降"定义为 S_const(step 1000) ≤ 0.5 × max_{0–1000} S_const；"A 回升"定义为 A(step 1000) − min_{0–1000} A ≥ 0.2。

---

## 修订记录

（空）
