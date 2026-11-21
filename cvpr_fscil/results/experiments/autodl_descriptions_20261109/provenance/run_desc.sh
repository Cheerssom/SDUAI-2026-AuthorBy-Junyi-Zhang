#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=${PYTHON:-/root/autodl-tmp/desc-env/bin/python}
exec "$PYTHON" -u desc_pipeline.py --max-active "${MAX_ACTIVE:-2}"
