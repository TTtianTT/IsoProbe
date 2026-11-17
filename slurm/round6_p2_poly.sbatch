#!/bin/bash
#SBATCH --job-name=iso-r6p2-poly
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-1%2 slurm/round6_p2_poly.sbatch
# Round 6 phase 2 T4: array task k runs P worker processes on its one GPU (the 70m/160m checkpoints leave the GPU mostly idle);
# worker w measures rows k + 2w, k + 2w + 2P, ... of configs/round6_poly_ckpts.tsv (dyn + spectrum + eval loss),
# then the eval loss of the original-seed windows (configs/round6_p2_origseed_eval.tsv, same striding).
# Rows not yet downloaded or already measured are skipped; re-submit to fill in.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
k=$SLURM_ARRAY_TASK_ID
P=${P:-4}
N=$(wc -l < configs/round6_poly_ckpts.tsv)
M=$(wc -l < configs/round6_p2_origseed_eval.tsv)
for w in $(seq 0 $((P-1))); do
  s=$((k + 2*w))
  ( $PY round6_poly.py --list configs/round6_poly_ckpts.tsv --rows "$(seq $s $((2*P)) $((N-1)) | paste -sd,)" --eval
    $PY round6_poly.py --list configs/round6_p2_origseed_eval.tsv --rows "$(seq $s $((2*P)) $((M-1)) | paste -sd,)" --eval_only
  ) > logs/iso-r6p2-poly_${SLURM_ARRAY_JOB_ID}_${k}_w${w}.out 2>&1 &
done
wait
