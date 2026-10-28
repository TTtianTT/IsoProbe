#!/bin/bash
#SBATCH --job-name=iso-r4b
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-$(( $(wc -l < configs/round4_groups.txt) - 1 ))%2 [--time=..] slurm/run_dyn.sbatch
# Round 4 B: line SLURM_ARRAY_TASK_ID+1 of configs/round4_groups.txt = rows of configs/round4_ckpts.tsv to run
# in sequence (existing results are skipped).
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
ROWS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" configs/round4_groups.txt)
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): rows $ROWS"
$PY dyn_round4.py --rows "$ROWS"
