#!/bin/bash
#SBATCH --job-name=iso-r6-poly
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-1%2 slurm/round6_poly.sbatch   (task 0: rows 0-8 = 70m seeds, task 1: rows 9-17 = 160m seeds)
# Round 6 S6: PolyPythias step143000; finished rows are skipped.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
ROWS=$([ "$SLURM_ARRAY_TASK_ID" = 0 ] && echo 0-8 || echo 9-17)
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): rows $ROWS"
$PY round6_poly.py --rows "$ROWS"
