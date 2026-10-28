#!/bin/bash
#SBATCH --job-name=iso-unigram5
#SBATCH --partition=RTXq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# CPU only (no --gres): round-5 item 3: Pythia tokenizer unigram counts.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
export TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PY count_unigrams.py --models configs/round5_unigram_models.txt --out results/round5/unigram_groups.json
