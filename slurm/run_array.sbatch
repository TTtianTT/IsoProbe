#!/bin/bash
#SBATCH --job-name=iso
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-$(( $(wc -l < models.txt) - 1 ))%2 slurm/run_array.sbatch
# One model (line SLURM_ARRAY_TASK_ID+1 of models.txt) per task, one GPU per task:
# measure.py (A-D metrics), then diag_final_norm.py (final-norm decomposition).
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
MODEL=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" models.txt)
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SLUG=$(basename "$MODEL")
# each step is skipped if its output already exists, so the array can be re-submitted after adding models
[[ -f results/$SLUG.json ]] || $PY measure.py --model "$MODEL" --data data/samples.jsonl --max_len 256 --out results/
[[ -f results/diag/$SLUG.json ]] || $PY diag_final_norm.py --model "$MODEL" --data data/samples.jsonl --max_len 256 --out results/diag/
