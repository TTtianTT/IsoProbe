# 第六轮（阶段一）进度

更新于 2026-11-17 17:45（Asia/Singapore）。全部阶段已完成，所有结果已 commit 并 push；队列中没有残留作业。
按计划第 8 节：服务器重启后不自动续跑，等待用户确认下一阶段的计划。

| 阶段 | 状态 | 作业号 | 预登记判定 / 备注 |
|---|---|---|---|
| S0 | 完成 | srun 71342 | 基线 gpt2 V1/V3 = 0.1533 / 0.0679，与第三轮差 0；评估集 50.7 万 token（保留并标记了 in_samples 段落，否则只有 30.9 万） |
| S1 | 完成 | CPU array 71346（12 个任务，4RTXq，无 GPU，25 – 112 分钟） | P6-1、P6-2、P6-3 成立 |
| S2 | 完成 | GPU array 71417、71425、71443（依次运行，不重叠） | P6-4 成立；V1/V3 与第四轮差 0 |
| S3 | 完成 | CPU 71418、71426、71444 | P6-5 不成立（3/4，Qwen3-8B-Base J = 0.73） |
| S4 | 完成 | GPU array 71360；探索性 srun 71375、71381、71384、71385 | P6-8、P6-9 不成立；坍缩后代价集中在约 0.1% 的 token |
| S5 | 完成（初步） | GPU array 71386；71398（重跑 70m step 50000） | 按规则判为"身体侧"，但 1000 步重拟合没有改变谱尖峰，判读不可信 |
| S6 | 完成 | GPU array 71403 | 70m 8/9、160m 1/9 个新 seed 坍缩 |

**资源**：GPU 约 1.5 GPU 小时（全部为 1 张 B300 的任务，任何时刻 ≤ 2 张）；CPU 约 620 核小时（S1 为主）。

**续跑命令**（都会跳过已有输出；目前没有未完成项）：
- S1：`sbatch --array=0-11%6 slurm/round6_gauge.sbatch`
- S2：`sbatch --array=0-11%2 slurm/round6_sts.sbatch`（需要 `cache/round6/gauge/`）
- S3：`sbatch slurm/round6_od.sbatch`
- S4 / S5：`sbatch --array=0-10%2 slurm/round6_causal.sbatch ablate|refit`
- S6：`sbatch --array=0-1%2 slurm/round6_poly.sbatch`
- 汇总：`$PY round6_summary.py s1|s2|s3|s4|s5|s6`

**遇到的问题**
1. S0 的两次 push 卡在 SSH 握手（端口可达、握手无响应）；S4 提交时 push 恢复，之后每次都成功。
2. S1 的最坏情形优化（fp64、5 个初值 × 400 步 × 4 组）比预估慢：d = 5120 的模型约 1 小时 50 分钟。为此把 CPU array 的并发从 6 提到 12（不占 GPU），
   并把 S2 / S3 拆成按 S1 完成顺序的分批提交。
3. S5 的 70m step 50000 第一次从冒烟测试的存档续跑（存档目录没有随 `--out` 改变）：step 0/250/500 的评估是在 2 万 token 子集上算的。
   已删除并重跑（71398），代码加了 `--refit_cache`。
4. 三条 commit message 中的数字后来在核对时更正（S1 区间的描述、S2 两个计数、S3 的 8/12 → 7/12），更正记录在各自的后续 commit 与 REPORT 中。
5. 预登记修订一次（4c 增加未截断头的偏置对照），在任何 S4 计算之前提交。

**留给下一阶段（未做）**：S5 完整版（3 seed、学习率选择、随机初始化 / 截断后头的对照，并检查尖峰是否被去掉）；S4 中承载代价的约 0.1% token 的具体识别；
PolyPythias 的中间 checkpoint（尤其 160m）。

---

# 阶段二进度（2026-11-17 起）

更新于 2026-11-18 04:55（Asia/Singapore）。T0 – T6 与第 9 节补充 1 – 4 全部完成，所有结果已 commit 并 push；队列中没有残留作业（`squeue` 为空）。
按计划：第 9 节补充项已全部做完，不再开新实验；服务器重启后不自动续跑，等待用户确认。

| 任务 | 状态 | 作业号 | 判定 / 备注 |
|---|---|---|---|
| T0 | 完成 | — | 停机条件未触发（Cain 2026 只在机制上部分重合）；h* = Park et al. 因果内积几何 |
| T1 | 完成 | CPU 71457、71458 | P7-1 不成立（70m +0.115）；坍缩在 h* 中表现为公共均值方向（‖ū‖² 0.998） |
| T2 | 完成 | GPU 71470；CPU 71471、71480、分析重跑 | 2a 弥散（补集 ≥ 0.983）；P7-2 不成立（70m +0.133）；H 主要是 UTF-8 字节片段 |
| T3 | 完成 | GPU 71482；核查 71528 | 数值核查：SVD + 数值秩容差下 4/4 极差 < 1e-6，停止条件未触发；P7-3 按预注册不成立（极差 0.0196，只有 Qwen3-8B/-Base 的 V3，1e-12 阈值丢掉 1 个特征值）；P7-4 不成立（8/12） |
| T4 | 完成 | GPU 71505、71507（取消）、71510；fp32 评估 loss CPU 71542（超时于 884/1300）、71623（续完 1300/1300） | P7-5 成立（8/8）；fp32 loss：70m seed 2 – 9 的 t50 loss − t50 ‖ū‖² 中位数 −7000（2/8 > 0），loss 后期上升中位数 0.0643 nat；bf16 版作废 |
| T5 | 完成 | GPU 71536（array 0-14%2）；smoke 71535（目录已删） | P7-6 成立（0.181 / 0.181 / 0.182）；身体侧（bf16 主体 D_refit/D_raw 70m 1.45、160m 1.24；fp32 主体 3.29、1.38） |
| T6 | 完成 | GPU 71568（array 0-4%2）；CPU smoke 71539（目录已删） | 探索性：R-pen 把 ‖ū‖² 压到 0.019 / 0.009；从 143000 出发 loss 4.574 对普通续训 3.6345 |
| 补充 1 | 完成 | GPU 71576（任务 0 – 7） | T5 随机头 3 seed：160m D_refit +0.1478 ×3，410m 平均 −0.0119 |
| 补充 2 | 完成 | GPU 71576（任务 8 – 9） | 70m seed 2/3：补集占比 0.991 / 0.997，ρ 0.116 / 0.100 |
| 精度核查 | 完成 | GPU 71593（array 0-8%2）；CPU 71579 | fp32 主体下 70m 50000 → 143000 loss 上升 0.0490（bf16 主体 0.1674）；T2 / T5 的分类不变 |
| 补充 3 | 完成 | GPU 71617（array 0-11%2）；smoke 71615（gpt2、小拟合集，独立目录，已删） | SICK-R mean / STS-B PromptEOL：秩容差路径白化极差 12/12 为 0；zscore G_lin 极差 > 0.5 为 8/12、6/12；白化 − zscore CI > 0 为 12/12、4/12 |
| 补充 4 | 完成 | GPU 71656（array 0-1%2） | R-pen λ = 0.1 / 10：‖ū‖² 0.862 / 0.017，loss 3.6345 / 5.1958（普通续训 3.6345） |

**资源（阶段二，sacct）**：GPU 约 21.0 GPU 小时（全部为 1 张 B300 的任务，任何时刻 ≤ 2 张，GPU array 都是 `%2` 且同一时间只有一个）；
CPU 约 2100 核小时，其中 fp32 评估 loss（71542 + 71623）约 1910。下载 PolyPythias 1142 个 checkpoint 共 182 GB（< 500 GB）。

**续跑命令**（都跳过已有输出；目前没有未完成项）。先 `source slurm/env.sh; unset PYTHONPATH`：
- T1：`sbatch --array=0-11%12 slurm/round6_p2_traj.sbatch`；T2：`sbatch --array=0-5%2 slurm/round6_p2_tokens.sbatch`
- T3：`sbatch --array=0-11%2 slurm/round6_p2_whiten.sbatch`；核查 `sbatch --array=0-3%2 slurm/round6_p2_whiten_check.sbatch`
- T4：`sbatch --array=0-1%2 slurm/round6_p2_poly.sbatch`；fp32 loss `sbatch --array=0-11%12 slurm/round6_p2_evalfp32.sbatch`
- T5：`sbatch --array=0-14%2 slurm/round6_p2_refit.sbatch`；T6：`sbatch --array=0-4%2 slurm/round6_p2_ft.sbatch`
- 补充 1 – 2：`sbatch --array=0-9%2 slurm/round6_p2_fallback.sbatch`；精度核查：`sbatch --array=0-8%2 slurm/round6_p2_fp32check.sbatch`
- 补充 3：`sbatch --array=0-11%2 slurm/round6_p2_fb3.sbatch`；补充 4：`sbatch --array=0-1%2 slurm/round6_p2_ft.sbatch --list configs/round6_p2_ft_extra.tsv`
- 所有 sbatch 都需要加 `--deadline=<截止时间>`。汇总：`$PY round6_p2_summary.py t1|t2|t3|t3check|t4|t5|t6|fb3`

问题记录（阶段二）：
- 19:2x 发现最初的下载进程已全部退出（4 个停在下载中途），886/1142 行完成；用 `--shard k/6` 重启 6 个进程（已完成的行按 `.complete` 跳过）。
- T4 第一次提交（71505）后改为每个 GPU 4 个进程（71507），因 B300 是 exclusive-process 模式失败（cudaErrorDevicesUnavailable），取消后回到每 GPU 1 个进程（71510）。
  取消前把 `round6_poly.py` 的 JSON 写入改为先写临时文件再改名；已有的 108 个 JSON 全部可解析。
- 22:40 发现 T4 的评估 loss（bf16 模型 logit）在坍缩 checkpoint 上严重偏高，撤回 REPORT 中的 loss t50 结论，预注册追加修订后用 fp32 重算（CPU 71542）。
- 00:15 精度问题：bf16 主体路径（阶段一 S4、T2、T5）把 70m 50000 → 143000 的 loss 上升从 0.049（fp32）放大到 0.167；160m 0.0875 对 0.1188。
  预注册追加了精度核查（fp32 主体重做 T2 导出与 T5 seed 0），排在补充 1 – 2 之后。前几轮基于 bf16 主体的 loss 数值（dyn loss_base 等）可能同样受影响，未改动。
- 01:30 fp32 评估 loss 的 CPU array 71542 在 3 小时上限处超时（884/1300，没有残留临时文件），续交 71623 完成剩余部分（03:48）。
- fp32 下 T4 的 loss 后期上升只有百分之几 nat，loss t50 对噪声敏感；REPORT 中 loss 与 ‖ū‖² 的先后只作描述。
- 补充 3 中 eigh 1e-12 路径在 SICK-R 上另有 GPT-2 large / xl 极差 6.3e-6 / 1.2e-6（> 1e-6）；秩容差路径全部为 0。
- commit 中的一处预注册时间戳写错（01:25 → 01:05），已在随后的 commit 中更正。
- 所有 push 均一次成功。

---

# 阶段三进度（2026-11-18 23:20 起）

更新于 2026-11-18 23:54（Asia/Singapore；原文误写为 11-19 00:50，已按 commit 13676ea 更正）。**按任务书第 9 节停下**：文献核对命中停机条件"白化余弦作为规范不变的句向量相似度"（部分命中，见下），等待用户决定。
队列为空；停下前已在跑的两个作业（U0 dyn GPU array、U1 CPU array）跑完并提交，没有提交新作业（U0 的 S4 部分未提交）。

| 任务 | 状态 | 作业号 | 判定 / 备注 |
|---|---|---|---|
| U0（dyn / logit 部分） | 完成 | GPU 71745（array 0-9%2）；smoke srun 71743、71744（独立目录，已删） | **P8-1 成立**：fp32 Spearman 0.9914（同一批点 bf16 0.9919，340 点）；Δloss、可加占比无符号变化，"子集 Δloss > 全部"无变化；corr(δ_S, δ_R) 3 处变号（2 处 \|corr\| ≤ 0.005，410m step 128000 top8 −0.107 → +0.057，因 fp32 下前 8 维集合变了一维） |
| U0（S4 部分） | 完成 | GPU 71762（array 0-10%2） | **P8-2 成立**（0.9986 / 0.9545）；11 个 checkpoint 的干预 Δloss 与原值差 ≤ 0.013；没有结论反转，停机条件未触发 |
| U1 | 完成 | CPU 71752（array 0-2，4RTXq）；smoke 71749、71751（独立目录，已删） | **P8-3、P8-4 成立**（8 个群内情形，logits 最大相对误差 ≤ 3.7e-12，各层指标差 ≤ 6.3e-12）；对照：gpt2 非保 1 的 Haar 0.69、OLMo-2 Haar 1.52。理论 `docs/theory_round6.md`（23:59 写完） |
| U2 – U4、claims.md | 未开始 | — | — |
| 文献（第 8 节） | 完成 1 – 3 | — | 见 `related_round6.md`"阶段三" |

**停机原因（需要用户决定）**：Svendsen, Jakobsen, Mager & Nassar 2026（arXiv 2605.30596，*Improving Relative Representations with Learned Anchors and Whitened Inner Products*）
逐字写有 "Exact affine invariance holds only for the non-shrunk form (Appendix B)"（§3.3）与 "an affine-invariant function"（§8），对象是白化内积 (x − μ)ᵀΣ⁻¹(a − μ)，
用在 BERT 系列文本编码器与 GPT-2 / TinyLlama / Gemma 的隐藏状态上。与本项目的差别：它针对不同编码器之间的仿射差异（跨模型对齐），不是同一模型的保持函数的重参数化；
用内积而非余弦；没有 STS、没有与 zscore 比较、没有数值秩问题。附录 B 的证明未能逐字读到（本机无 PDF 文本抽取工具，HTML 截断）。
若继续，建议：引用它作为"白化相似度的仿射不变性"的已有陈述，把第 2 句"办法"的新意限定为"用于 LLM 的规范问题（final norm 折算、跨模型排序）与 STS 上的系统比较"，然后按原顺序做 U0 的 S4 部分与 U2 – U4。

**另外两条文献结果（不触发停机）**：SliceGPT（Ashkboos et al. 2024，Theorem 1）已证明 RMSNorm 模型对正交 Q 的不变性（充分性）；U1 命题 2 的 RMSNorm 充分性部分改为引用。
本项目补的是必要性、LN 下沿 1 的剪切自由度（数值上 gpt2 剪切后输出不变、n(x) 指标不变、未归一化 x 的 A 变化 0.286），以及 OLMo-2 型子层输出后 norm 只剩带符号置换（Haar Q 破坏输出）。

**资源（阶段三，sacct，`results/round6/budget.json`）**：GPU 0.67 / 60 小时，CPU 29.0 / 3000 核小时（17 个作业，均 ≤ 2 张卡）。

**23:56 用户决定继续**（按上面的建议）：引用 Svendsen et al. 2026，第 2 句的新意限定为"用于 LLM 的规范问题与 STS 上的系统比较"；按原顺序继续 U0 的 S4 部分与 U2 – U4。
U0 S4：GPU array 71762（`--array=0-10%2`，23:57 提交）。
另：本阶段 3 处时间戳曾写错（U0 / U1 预登记标题、本节"更新于"），已按 commit 时间更正；预登记仍都早于对应计算（第一个冒烟 srun 在 36ea614 之后，U1 作业在 dac92db 之后）。
U1 文档完成（23:59）。U2：GPU array 71774（00:09 提交，12 个模型 `%2`）。U3：pythia-14m / 2.8b 下载中（登录节点），之后 CPU 作业（不占 GPU，与 U2 并行；U3 的其余部分只用已有数据）。
**U2 实现问题（00:24 发现并修正）**：gpt2 的 mean 池化在 STS12 上白化 7 表示极差 4.5e-3。原因不是数值秩：STS12 的句子 "end" 在不加 BOS 的 tokenizer 下只有 1 个 token，
"去掉位置 0 的平均"为空，得到 NaN 向量，NaN 余弦在各表示下打乱了秩。其余句对的余弦差 ≤ 1e-8。该切片来自阶段二 `round6_whiten.py`（STS-B / SICK-R 中没有单 token 句子，阶段二不受影响）；
第四轮 `sts_round4.py` 对 T = 1 取位置 0，是正确的。修正为与第四轮相同（commit b7f9836），已有缓存中的 NaN 行重新前向。受影响：不加 BOS 的 9 个模型、STS12、mean 池化、1 个句子。
任务 0 – 2（gpt2、gpt2-medium、gpt2-large）用旧代码跑，结果作废，待 71774 结束后重跑（缓存保留，只补这一行并重做分析）。

**U3 完成**（00:44）：CPU 71779。P8-7 成立（pythia-2.8b A* 0.500，A*_π 0.297）；第 2、3 行在 A* 下成立，第 1、4、6 行不成立，第 5 行两对在 A* 下同向。pythia-14m 终点没有坍缩（只报告）。
U4 的纯分析部分已做：P8-8 成立（1300 个 checkpoint，Spearman 0.953）。U4 的 GPU 部分（机制检查、12 个模型）等 U2 array 结束后提交。
