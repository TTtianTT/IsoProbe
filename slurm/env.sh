# Shared environment for IsoProbe Slurm steps (python/torch env reused from TurnPrint).
export PY=/dataset1/zailong/envs/peft-sft-lab/bin/python
export HF_HOME=/dataset1/zailong/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# node-local /tmp is small; keep temp files on /dataset1
export TMPDIR=/dataset1/zailong/tmp/iso-${SLURM_JOB_ID:-local}
export TMP=$TMPDIR TEMP=$TMPDIR MPLCONFIGDIR=$TMPDIR/mpl
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
