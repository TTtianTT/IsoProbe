#!/bin/bash
#SBATCH --job-name=iso-r6p2-fb3
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-11%2 slurm/round6_p2_fb3.sbatch   (rows of configs/round6_models.txt)
# Round 6 phase 2 section-9 fallback 3: T3 on SICK-R (mean pooling) and STS-B with PromptEOL; extra args passed through.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
MODEL=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-${ROW:-0}}+1))p" configs/round6_models.txt)
echo "task ${SLURM_ARRAY_TASK_ID:-${ROW:-0}} on $(hostname): $MODEL"
$PY round6_whiten.py --model "$MODEL" --configs sickr:mean,stsb:eol --out results/round6/p2/whiten_fb3/ "$@"
