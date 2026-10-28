#!/bin/bash
#SBATCH --job-name=iso-unigram
#SBATCH --partition=RTXq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# CPU only (no --gres): round-3 step 3b unigram counts.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
export TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PY count_unigrams.py
