#!/bin/bash
#SBATCH --job-name=iso-r6p2-refit
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-14%2 slurm/round6_p2_refit.sbatch   (rows of configs/round6_p2_refit.tsv)
# Round 6 phase 2 T5: reduced-head refit (gamma, beta, W_U) on frozen bodies; resumable from the latest 1000-step save.
# Extra arguments (smoke tests: --total 200 --max_tokens 20000 --out <scratch> --cache <scratch>) are passed through.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
echo "task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname): $(sed -n "$((${SLURM_ARRAY_TASK_ID:-0}+1))p" configs/round6_p2_refit.tsv)"
$PY round6_refit2.py run --rows "${ROWS:-${SLURM_ARRAY_TASK_ID:-0}}" "$@"
