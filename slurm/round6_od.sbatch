#!/bin/bash
#SBATCH --job-name=iso-r6-od
#SBATCH --partition=4RTXq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%j.out
# CPU only (no --gres). Round 6 S3: outlier dimensions for all models that have an S1 gauge cache; existing outputs skipped.
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
$PY round6_od.py
