#!/bin/bash
#SBATCH --job-name=iso-r6p2-whtchk
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-3%2 slurm/round6_p2_whiten_check.sbatch
# Round 6 phase 2 T3 numerical check (configs/round6_prereg.md, T3 amendment): eigh vs SVD vs SVD with rank tolerance.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
MODELS=(Qwen3-8B-Base Qwen3-8B Llama-3.1-8B gpt2)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
MODEL=$(grep -m1 -- "$M\$" configs/round6_models.txt)
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $MODEL"
$PY round6_whiten.py --model "$MODEL" --check --out results/round6/p2/whiten_check/
