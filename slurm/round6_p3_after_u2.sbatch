#!/bin/bash
#SBATCH --job-name=iso-r6p3-u2u4
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-15%2 slurm/round6_p3_after_u2.sbatch
#   tasks 0-2 : U2 rerun for gpt2 / gpt2-medium / gpt2-large (single-token pooling fix; caches kept, NaN rows re-embedded)
#   task 3    : U4 mechanism check (pythia-70m step 143000 / 50000, per-segment precision)
#   tasks 4-15: U4 bf16 vs fp32 on models.txt line (k-3)
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
k=$SLURM_ARRAY_TASK_ID
if [ $k -le 2 ]; then
  $PY round6_p3_whiten.py --model "$(sed -n "$((k+1))p" models.txt)"
elif [ $k -eq 3 ]; then
  $PY round6_p3_precision.py mech
else
  $PY round6_p3_precision.py big --model "$(sed -n "$((k-3))p" models.txt)"
fi
