#!/bin/bash
#SBATCH --job-name=iso-r5b
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-$(( $(wc -l < configs/round5_gpu_groups.txt) - 1 ))%2 [--time=..] slurm/run_r5gpu.sbatch
# Round 5 items 2-3: rows of configs/round5_gpu.tsv (see round5_gpu.py); existing outputs are skipped.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
ROWS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" configs/round5_gpu_groups.txt)
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): rows $ROWS"
$PY round5_gpu.py --rows "$ROWS"
