#!/usr/bin/env bash
# AnchorCE + base-frozen LDA evidence source on all 54 archived tasks (resumable), then summary.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=${PYTHON:-/root/autodl-tmp/fscil-env/bin/python}
"$PYTHON" anchor_gda.py --all 2>&1 | tee -a work/anchorg.log
"$PYTHON" fscil_summary.py 2>&1 | tee work/summary_with_anchorg.txt
echo ANCHORG_DONE | tee -a work/anchorg.log
