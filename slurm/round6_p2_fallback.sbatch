#!/bin/bash
#SBATCH --job-name=iso-r6p2-fb
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-9%2 slurm/round6_p2_fallback.sbatch
# Round 6 phase 2 section-9 fallback items 1-2: tasks 0-7 = rows of configs/round6_p2_refit_extra.tsv (T5 extra seeds),
# tasks 8-9 = token dumps of configs/round6_p2_tokens_seeds.tsv (rows 0-1 and 2-3).
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
k=$SLURM_ARRAY_TASK_ID
if [ $k -lt 8 ]; then
  $PY round6_refit2.py run --list configs/round6_p2_refit_extra.tsv --rows $k
else
  j=$(( (k-8)*2 ))
  $PY round6_tokens.py dump --list configs/round6_p2_tokens_seeds.tsv --rows "$j-$((j+1))"
fi
