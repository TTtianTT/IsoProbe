#!/bin/bash
#SBATCH --job-name=iso-r6p2-traj
#SBATCH --partition=4RTXq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# CPU only (no --gres). Usage: sbatch --array=0-11%12 slurm/round6_p2_traj.sbatch  (shard k/12 of configs/round6_p2_traj.tsv)
# Round 6 phase 2 T1; finished checkpoints are skipped.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
$PY round6_traj.py --shard "$SLURM_ARRAY_TASK_ID/12"
