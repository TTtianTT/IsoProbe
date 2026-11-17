#!/bin/bash
#SBATCH --job-name=iso-r6p2-ft
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-4%2 slurm/round6_p2_ft.sbatch   (rows of configs/round6_p2_ft.tsv)
# Round 6 phase 2 T6 (exploratory): full fine-tuning of pythia-70m, 3000 steps; extra arguments are passed through.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
echo "task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname): $(sed -n "$((${SLURM_ARRAY_TASK_ID:-0}+1))p" configs/round6_p2_ft.tsv)"
$PY round6_ft.py --rows "${ROWS:-${SLURM_ARRAY_TASK_ID:-0}}" "$@"
