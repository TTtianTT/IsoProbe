#!/bin/bash
#SBATCH --job-name=iso-r3
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-$(( $(wc -l < models.txt) - 1 ))%2 slurm/run_round3.sbatch
# Round 3 GPU part for line SLURM_ARRAY_TASK_ID+1 of models.txt; skipped if its result already exists.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
MODEL=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" models.txt)
SLUG=$(basename "$MODEL")
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $MODEL"
[[ -f results/round3/$SLUG.json ]] && { echo "results/round3/$SLUG.json exists, skipping"; exit 0; }
$PY extract_round3.py --model "$MODEL"
