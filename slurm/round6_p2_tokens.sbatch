#!/bin/bash
#SBATCH --job-name=iso-r6p2-tok
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-5%2 slurm/round6_p2_tokens.sbatch   (rows of configs/round6_p2_tokens.tsv)
# Round 6 phase 2 T2 dump: per-token npz in cache/round6/p2/tokens/ (skipped if present).
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
$PY round6_tokens.py dump --rows "$SLURM_ARRAY_TASK_ID"
