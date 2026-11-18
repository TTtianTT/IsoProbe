#!/bin/bash
#SBATCH --job-name=iso-r6p3-u1
#SBATCH --partition=4RTXq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
# CPU only (no --gres). Usage: sbatch --array=0-2%12 slurm/round6_p3_gauge_resid.sbatch
# Round 6 phase 3 U1: fp64 residual-stream gauge check; task 0 gpt2, 1 olmo2-1b, 2 qwen2.5-7b.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
M=(gpt2 olmo2-1b qwen2.5-7b)
$PY round6_gauge_resid.py --model ${M[$SLURM_ARRAY_TASK_ID]}
