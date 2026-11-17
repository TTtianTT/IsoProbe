#!/bin/bash
#SBATCH --job-name=iso-r6-causal
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-10%2 slurm/round6_causal.sbatch ablate|refit   (rows of configs/round6_causal.tsv)
# Round 6 S4 (ablate) / S5 (refit); finished outputs are skipped, refit resumes from the last 250-step save.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $1 row $SLURM_ARRAY_TASK_ID"
$PY round6_causal.py "$1" --rows "$SLURM_ARRAY_TASK_ID"
