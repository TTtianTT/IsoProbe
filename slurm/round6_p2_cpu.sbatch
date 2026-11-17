#!/bin/bash
#SBATCH --job-name=iso-r6p2-cpu
#SBATCH --partition=4RTXq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --deadline=2026-11-18T08:30:00
#SBATCH --output=logs/%x_%j.out
# CPU only (no --gres). Usage: sbatch slurm/round6_p2_cpu.sbatch "<python args>"  e.g. "round6_tokens_cpu.py"
set -euo pipefail
cd /dataset1/zailong/workspace/IsoProbe
unset PYTHONPATH
source slurm/env.sh
$PY $1
