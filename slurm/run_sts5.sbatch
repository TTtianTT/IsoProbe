#!/bin/bash
#SBATCH --job-name=iso-r5a
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%A_%a.out
# Round 5 item 1: STS (raw / center / zscore, no layer sweep) on the checkpoints of configs/round5_sts.tsv.
# Line SLURM_ARRAY_TASK_ID+1 of configs/round5_sts_groups.txt = 0-based row range to run in sequence.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
RANGE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" configs/round5_sts_groups.txt)
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): rows $RANGE"
for i in $(seq ${RANGE%-*} ${RANGE#*-}); do
  IFS=$'\t' read -r REPO REV < <(sed -n "$((i+1))p" configs/round5_sts.tsv)
  NAME=$(basename "$REPO")__$REV
  [[ -f results/round5/sts/$NAME.json ]] && { echo "$NAME exists, skipping"; continue; }
  $PY sts_round4.py --model "/dataset1/zailong/models/ckpts/$(basename "$REPO")/$REV" --variants raw,center,zscore \
      --no_layers --out results/round5/sts/ --out_name "$NAME"
done
