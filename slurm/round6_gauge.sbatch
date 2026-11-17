#!/bin/bash
#SBATCH --job-name=iso-r6-gauge
#SBATCH --partition=4RTXq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# CPU only (no --gres). Usage: sbatch --array=0-11%6 slurm/round6_gauge.sbatch   (rows of configs/round6_models.txt)
# Round 6 S1: gauge audit + gauge metric, one model per task; existing outputs are skipped.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
MODEL=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" configs/round6_models.txt)
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $MODEL"
$PY round6_gauge.py --model "$MODEL"
