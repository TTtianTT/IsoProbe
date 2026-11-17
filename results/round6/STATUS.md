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
