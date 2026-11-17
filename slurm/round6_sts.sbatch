#!/bin/bash
#SBATCH --job-name=iso-r6-sts
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-11%2 slurm/round6_sts.sbatch   (rows of configs/round6_models.txt; needs S1 caches)
# Round 6 S2: STS-B test, mean pooling, V1/V3 + gauge-transformed representations; existing outputs are skipped.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
MODEL=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" configs/round6_models.txt)
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $MODEL"
SLUG=$(basename "$MODEL")
if [ -f "results/round6/sts/$SLUG.json" ]; then echo "exists, skipping"; exit 0; fi
$PY sts_round4.py --model "$MODEL" --variants raw,zscore --no_layers --datasets stsb --pools mean \
    --gauge_cache cache/round6/gauge/ --out results/round6/sts/
