#!/bin/bash
#SBATCH --job-name=iso-r6p3-u2diag
#SBATCH --partition=4RTXq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
# CPU only (no --gres). Usage: sbatch --array=0-11%12 slurm/round6_p3_whiten_diag.sbatch  (after U2: caches must be NaN-free)
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PY round6_p3_whiten_diag.py "$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" models.txt)"
