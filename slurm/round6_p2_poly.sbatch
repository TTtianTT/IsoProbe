#!/bin/bash
#SBATCH --job-name=iso-r6p2-poly
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-1%2 slurm/round6_p2_poly.sbatch
# Round 6 phase 2 T4: task k measures rows k, k+2, ... of configs/round6_poly_ckpts.tsv (dyn + spectrum + eval loss),
# then the eval loss of the original-seed windows (configs/round6_p2_origseed_eval.tsv, same striding).
# Rows not yet downloaded or already measured are skipped; re-submit to fill in.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
k=$SLURM_ARRAY_TASK_ID
N=$(wc -l < configs/round6_poly_ckpts.tsv)
$PY round6_poly.py --list configs/round6_poly_ckpts.tsv --rows "$(seq $k 2 $((N-1)) | paste -sd,)" --eval
M=$(wc -l < configs/round6_p2_origseed_eval.tsv)
$PY round6_poly.py --list configs/round6_p2_origseed_eval.tsv --rows "$(seq $k 2 $((M-1)) | paste -sd,)" --eval_only
