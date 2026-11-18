#!/bin/bash
#SBATCH --job-name=iso-r6p3-u3
#SBATCH --partition=4RTXq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# CPU only (no --gres). Round 6 phase 3 U3: pythia-14m / 2.8b step143000 (fp32 dyn bank + T1 metrics) and the 2.8b main check.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
export ISO_DEV=cpu ISO_MODEL_DTYPE=fp32 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PY round6_p3_reanalysis.py measure
