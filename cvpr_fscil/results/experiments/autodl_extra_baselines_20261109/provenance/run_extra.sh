#!/usr/bin/env bash
# GDA + KLDA extra baselines on every archived standard-FSCIL task (resumable; skips completed tasks).
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=${PYTHON:-/root/autodl-tmp/fscil-env/bin/python}
"$PYTHON" extra_baselines.py --selftest 2>&1 | tee -a work/extra_baselines.log
"$PYTHON" extra_baselines.py --all 2>&1 | tee -a work/extra_baselines.log
"$PYTHON" fscil_summary.py 2>&1 | tee work/summary_with_extra.txt
