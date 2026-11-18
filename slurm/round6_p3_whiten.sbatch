#!/bin/bash
#SBATCH --job-name=iso-r6p3-u2
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-11%2 slurm/round6_p3_whiten.sbatch   (task k = line k of models.txt)
# Round 6 phase 3 U2: whitening on STS12-16 / STS-B / SICK-R x mean / last / PromptEOL (round6_p3_whiten.py).
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
M=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" models.txt)
$PY round6_p3_whiten.py --model "$M"
