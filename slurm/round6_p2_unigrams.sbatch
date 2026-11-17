#!/bin/bash
#SBATCH --job-name=iso-r6p2-uni
#SBATCH --partition=4RTXq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%j.out
# CPU only. Round 6 phase 2 T1: unigram counts for the OLMo-2 tokenizer (same text prefix as round 3).
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
$PY count_unigrams.py --models configs/round6_p2_unigram_models.txt --out results/round6/p2/unigram_groups_olmo2.json
