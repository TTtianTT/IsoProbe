#!/bin/bash
#SBATCH --job-name=iso-r6p3-u0
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-9%2 slurm/round6_p3_audit.sbatch dyn   (task k: rows k, k+10, ... of configs/round6_p3_audit_ckpts.tsv)
#        sbatch --array=0-10%2 slurm/round6_p3_audit.sbatch s4  (task k: row k of configs/round6_causal.tsv)
# Round 6 phase 3 U0: whole-model fp32 audit. Smoke tests: set ISO_P3_AUDIT_OUT / ISO_BANK_DIRS to a separate directory.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
export ISO_MODEL_DTYPE=fp32 ISO_BODY_DTYPE=fp32
export ISO_BANK_DIRS=${ISO_BANK_DIRS:-cache/round6/p3/fp32_banks}
k=$SLURM_ARRAY_TASK_ID
if [ "$1" = dyn ]; then
  N=$(wc -l < configs/round6_p3_audit_ckpts.tsv)
  $PY round6_p3_audit.py dyn --rows "$(seq $k ${NT:-10} $((N-1)) | paste -sd,)"
else
  $PY round6_p3_audit.py s4 --rows $k
fi
