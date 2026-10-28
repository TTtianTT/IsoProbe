#!/bin/bash
#SBATCH --job-name=iso-r5cpu
#SBATCH --partition=RTXq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# CPU only (no --gres): round-5 W_U spectra, direction alignment, early-phase word-frequency Spearman, H_uni.
# Processes every Pythia bank present at start; already finished checkpoints are skipped.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
$PY round5_cpu.py
