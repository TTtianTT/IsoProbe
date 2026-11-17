#!/bin/bash
#SBATCH --job-name=iso-r6p2-evf32
#SBATCH --partition=4RTXq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# CPU only (no --gres). Usage: sbatch --array=0-11%12 slurm/round6_p2_evalfp32.sbatch
# Round 6 phase 2 T4 amendment: fp32 eval loss for every T4 checkpoint; task k takes rows k, k+12, ... of both lists.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
export ISO_DEV=cpu OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
k=$SLURM_ARRAY_TASK_ID
for L in configs/round6_poly_ckpts.tsv configs/round6_p2_origseed_eval.tsv; do
  N=$(wc -l < $L)
  $PY round6_poly.py --list $L --rows "$(seq $k 12 $((N-1)) | paste -sd,)" --eval_only --eval_dtype fp32 --eval_out results/round6/p2/poly_eval_fp32/
done
