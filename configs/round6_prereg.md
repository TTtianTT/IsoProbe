# IsoProbe 第六轮（阶段一）预登记

写于任何第六轮 S1–S3 数据分析之前（2026-11-17 15:30 前后，Asia/Singapore）。S0 只做了基线复核（gpt2 V1 / V3 isotropy
0.1533 / 0.0679，与第三轮差 0）和评估集构建，没有计算任何第六轮指标。S4–S6 的预测在各自开跑前追加到本文件末尾。
本文件之外的分析在报告中标为"探索性"。

## 共同设定（S1–S3）

- 数据：`cache/banks/<model>.pt` 中 final norm 的输入 x（10000 个 token，fp32），全程转成 fp64。V1 = normalize(x)，V3 = γ⊙V1 + β。
  W_U 从 safetensors 读（GPT-2 为共享的 `wte.weight`），先舍入到 bf16（与 GPU 任务加载的模型一致）再转 fp64。
- 指标（都在 bank 上算）：
  - **A**：bank 版 isotropy = 1 − 各序列（≤ 10 个 bank token）内两两余弦均值的平均（`round3_lib.bank_self_sim`）。
    注意它与第三轮 `isotropy_full`（全序列）不同，第三轮两者相差 < 0.01。
  - **B_raw**：全部跨序列 token 对的平均余弦（精确值，`cross_cos_exact`，不抽样）；**B_c**：先减去 bank 均值再算同一量。
  - **RankMe/d**：中心化 bank 的奇异值熵（`measure.spectrum`）除以 d；另报告除以同 N、d 高斯参照后的值。
  - **top1**：中心化 bank 第一主成分的方差占比。
- 基点：(V1, W_U·diag(γ), b' = W_U β)。G 中的 A 都作用在 V1 上（h → A h，头 → W A⁻¹，偏置不变）。
- 三族（每个模型、每族、每个 κ 各 20 个样本，seed 0–19，族与 κ 用不同的随机流）：
  - G_diag：A = diag(s)，log s 独立均匀分布在 [−½ ln κ, ½ ln κ]，κ ∈ {10, 100}；
  - G_orth：Haar 正交 Q（高斯矩阵 QR 后按 R 对角符号修正）；
  - G_lin：A = Q₁ diag(s) Q₂，Q₁、Q₂ 独立 Haar，s 同 G_diag。
- 最坏情形（只对 A）：参数化 t = ln s ∈ [0, ln κ]^d。因为 A 对整体缩放不变，"条件数 ≤ κ"与这个盒约束等价。用投影 Adam
  （fp64，400 步）分别最大化、最小化 A；初值为常数、ln|γ| 线性映射进盒子、3 个随机点，取最好的一个。
  局部优化得到的是真实可达区间的**内部估计**，所以下面 P6-2 的相交比例是下界。
- γ 的条件数记为 max|γ| / min|γ|；V2 = diag(γ) V1 属于 G_diag(κ) 当且仅当该值 ≤ κ。
- **规范度量的实现（对任务书的补充）。** 有偏置时 logit 是仿射的：z = W h + b。为了让 (V1, 折算头, b' = W_U β) 与
  (V3, 原头, 0) 给出同一个对象，h* 用增广形式：h̃ = [h; 1]，W̃ = [W, b]，M̃ = W̃ᵀ P W̃，h* = M̃^½ h̃（对称平方根，fp64 eigh，
  负特征值截为 0）。h* 的内积等于中心化 logit（含偏置）的内积，因此对 G 中任何可逆 A 以及平移 / β 折算都不变（只差一个正交变换，
  而上面 5 个指标都对正交变换不变）。没有偏置时（8 个 RMSNorm 模型）它就是任务书的 M = W_Uᵀ P W_U。
  - 均匀版 P = I − 11ᵀ/V，V = 头的行数（含 padding 行，它们也进 softmax）。
  - 词频加权版 P_π = diag(π) − ππᵀ，π = 第三轮 `cache/unigrams/<group>.npy` 的计数归一化（不平滑；tokenizer 之外的行 π = 0）。
  - h* 的 RankMe 仍除以 d。

## S1 预测

| 编号 | 预测 | 判定 |
|---|---|---|
| P6-1 | G_orth 下 A、B_raw、B_c、RankMe/d、top1 不变 | 12 个模型 × 20 个样本 × 5 个指标，相对 V1 的最大绝对差 < 1e-5 |
| P6-2 | G_diag、κ ≤ 100 时模型间 A 的排序不可识别 | 主判定用 κ = 100 的最坏情形区间 [A_min, A_max]：66 个模型对中区间相交（max(下界) ≤ min(上界)）的比例 ≥ 80%。κ = 10 只报告 |
| P6-3 | A*、B* 等是规范不变量 | 两个 P 版本、5 个指标：(V1, 折算头, b') 与 (V3, 原头, 0) 的相对差 < 1e-4；20 个 G_lin（κ = 100）样本（头 W A⁻¹ 显式计算后重新求 M̃）同样 < 1e-4。相对差 = \|a − b\| / max(\|b\|, 1e-3) |
| 无方向 | A*（均匀版）下 12 个模型的排序与 V1、V3 排序的 Spearman | 只报告（V1、V3 用同一 bank 版 A） |

## S2 预测（STS-B test，mean 池化）

- 设定同第四轮：每句单独前向，mean 池化去掉位置 0，zscore 的统计量只在 `data/fit_sentences.jsonl` 上估计，所有计算 fp64。
- 表示都由 fp64 池化后的 V1 线性（h* 为仿射）得到，因为线性映射与平均池化可交换：V1、V3 = γ⊙V1 + β、S1 的两个最坏情形
  对角变换（κ = 100 的 A_max、A_min）、S1 的 G_diag κ = 100 前 3 个随机样本（seed 0–2）、h*（均匀版、词频版）。
- 核对：V1、V3 的 raw 与 zscore 分数与第四轮 `results/round4/sts/<model>.json` 的差应为 0。

| 编号 | 预测 | 判定 |
|---|---|---|
| P6-4 | 任意对角变换下 zscore 的 STS 不变 | 12 个模型 × {V3、两个最坏情形、3 个随机}：zscore 分数与 V1 zscore 的差全部为 0（fp64） |
| 无方向 | h* 上 raw 余弦相对 V3 raw、V3 zscore 的高低 | 只报告，配对 bootstrap（1000 次）95% CI |

## S3 预测（outlier 维度）

- OD 定义按 Macocco 等人的代码（见 `related_round6.md`）：维度 j 是 OD 当且仅当 median_i \|h_ij\| > 全体 \|h_ij\| 的 0.99 分位数。
  用 bank 中去掉位置 0 的 token（原文的位置都在第 101 个词之后）。
- 表示：V3、V1、S1 的两个最坏情形对角变换（κ = 100）。报告两两 Jaccard；两个空集的 Jaccard 记为 1。
- 重合：OD(V3) 与第三轮 V3 前 10 个 ‖ū‖² 贡献维（`results/round3/analysis.json` 的 `top10_dims`）的交集大小；
  与"可加功能占比"前 10 维的交集大小，可加占比 = Cov(W[:,k] m_k, δ_all) / Var(δ_all)，在出现 ≥ 5 次的 token 上计算
  （第三轮口径），对每一维都算后取前 10。

| 编号 | 预测 | 判定 |
|---|---|---|
| P6-5 | max/median \|γ\| ≥ 5 的四个模型（gpt2、gpt2-medium、Qwen3-8B-Base、Qwen3-8B）上 OD 集合依赖规范 | 4 个模型都满足 Jaccard(OD(V1), OD(V3)) < 0.5 才判为成立；报告满足的个数 |

---

## S4、S5 预登记（追加于 2026-11-17 15:45，任何 S4/S5 计算之前）

共同设定：
- checkpoint：pythia-70m step 50000、80000、110000、130000、143000；pythia-160m step 80000、110000、130000、143000；
  pythia-410m step 80000、143000（本地原 seed）。
- loss 一律在 `data/eval_loss_round6.jsonl`（3351 段，截到 512 token，50.7 万个预测 token）上算，按 token 平均。
  另外单独报告去掉 `in_samples` 段落后的值（用于检查与前几轮估计数据的重叠）。若前向太慢则截到前 20 万个 token 并记录。
- 所有干预都在 final norm 处做：V1 = normalize(x)（fp32，由 bf16 的 x 计算），logit = W_f V1 + b'，W_f = W_U·diag(γ)，b' = W_U β。
  基线 loss 用同一路径计算，并与模型自身 logit 在前 5 条序列上核对。
- 训练数据（S4 的偏置拟合、S5）：wikitext-103 train 非空行依次用 Pythia tokenizer 编码、拼接、切成 512 token 的块，
  按 seed 0 打乱。S5 用前 16000 块，S4 的偏置拟合用之后的块。评估集不参与任何拟合。

### S4

- **4b**：ū = V1 bank（`cache/round{4,5}_banks/`，第三轮的 1000 条序列）的 L2 归一化 token 的均值，再单位化；
  c = bank 上 ūᵀV1 的均值。三种干预（u 为单位方向，p_t = uᵀV1_t）：
  1. V1 − c·u（只去掉沿 u 的均值，等价于固定 logit 偏置）；
  2. V1 − (p_t − c)·u（沿 u 的分量固定为 c）；
  3. V1 − p_t·u（完全投影掉）。
  对照：5 个随机高斯单位方向（seed 0–4），各自的 c 同样在 bank 上估计。Δloss = loss(干预) − loss(基线)。
- **4c**：W_f 的 SVD，σ₁ 截到 σ₂：W_t = W_f − (σ₁ − σ₂) u₁v₁ᵀ（用头的全部行，含 padding 行）。然后只拟合一个 logit 偏置 b：
  初值为补偿均值的 b₀ = (σ₁ − σ₂) E_train[v₁ᵀV1] u₁，再用 Adam（lr 1e-2）在训练块上走 200 步（每步 8 × 512 token，seed 0）。
  报告截断后、b₀ 后、200 步后的 Δloss。

| 编号 | 预测 | 判定 |
|---|---|---|
| P6-8 | 坍缩后，干预 2 的代价远小于干预 1 | 70m、160m 的 step 143000 都满足 Δloss(2) < 0.2·Δloss(1)（ū 方向） |
| P6-9 | 谱尖峰只承载常数先验 | 70m、160m 的 step 143000 都满足：截断加偏置拟合 200 步后 Δloss < 0.02 nats |

其余 checkpoint、随机方向、干预 3 只报告。

### S5（缩减版，"初步"）

- 冻结 final norm 之前的全部参数；训练 final norm 的 γ、β 和 W_U（fp32 副本，从该 checkpoint 自己的值初始化）。
  logit = (γ ⊙ normalize(x) + β) W_Uᵀ，x 由冻结的 bf16 主体前向得到。
- AdamW，lr 1e-4（恒定，无 warmup），weight_decay 0，betas (0.9, 0.999)，1000 步，batch 16 × 512 token，seed 0。
  每 250 步存一次头的权重与优化器状态（`cache/round6/refit/`，不进 git）并在评估集上算 loss，中断后从最近的存档续跑。
- 顺序：70m 的 5 个 → 410m 的 2 个 → 160m 的 4 个。
- 读法（不预设方向）：对每个规模，D_raw = loss_base(step 143000) − min_所选 checkpoint loss_base，
  D_refit = loss_refit(step 143000) − min_所选 checkpoint loss_refit（refit 取 1000 步时的值）。
  D_refit ≤ 0.25·D_raw → 读出侧；≥ 0.75·D_raw → 身体侧；其间 → 混合。D_raw ≤ 0 时不作判读。
  单 seed、单学习率，报告中标为"初步"。
