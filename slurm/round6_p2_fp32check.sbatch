#!/bin/bash
#SBATCH --job-name=iso-r6p2-f32chk
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-8%2 slurm/round6_p2_fp32check.sbatch
# Round 6 phase 2 precision check (prereg amendment 00:20): fp32 body instead of bf16.
# Tasks 0-5: T5 random-init refit, seed 0, rows 0,6,9,10,11,12 of configs/round6_p2_refit.tsv; tasks 6-8: T2 token dumps (rows 0-1, 2-3, 4-5).
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
export ISO_BODY_DTYPE=fp32
k=$SLURM_ARRAY_TASK_ID
R=(0 6 9 10 11 12)
if [ $k -lt 6 ]; then
  $PY round6_refit2.py run --rows ${R[$k]} --out results/round6/p2/refit_fp32/ --cache cache/round6/p2/refit_fp32/
else
  j=$(( (k-6)*2 ))
  $PY round6_tokens.py dump --rows "$j-$((j+1))" --cache cache/round6/p2/tokens_fp32
fi
