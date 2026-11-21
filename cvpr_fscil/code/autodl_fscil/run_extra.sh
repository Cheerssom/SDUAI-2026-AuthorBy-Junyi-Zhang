#!/usr/bin/env bash
# Round 3 (2026-11-09): (A) GDA + KLDA baselines, (B) retention-ratio / calibration-cap sweep (B/16),
# (C) efficiency. Resumable: completed tasks are skipped. Run stages separately with STAGES="A B C".
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=${PYTHON:-/root/autodl-tmp/fscil-env/bin/python}
STAGES=${STAGES:-"A B C"}
LOG=work/round3.log
if [[ $STAGES == *A* ]]; then
  "$PYTHON" extra_baselines.py --selftest 2>&1 | tee -a $LOG
  "$PYTHON" extra_baselines.py --all 2>&1 | tee -a $LOG
  "$PYTHON" fscil_summary.py 2>&1 | tee work/summary_with_extra.txt
fi
if [[ $STAGES == *B* ]]; then
  for ds in cifar100 mini cub200; do for seed in 0 1 2; do
    "$PYTHON" fscil_readouts.py --dataset $ds --encoder b16 --track certified --seed $seed --tag _sweep --sweep 2>&1 | tee -a $LOG
  done; done
  "$PYTHON" collect_sweep.py 2>&1 | tee work/summary_sweep.txt
fi
if [[ $STAGES == *C* ]]; then
  "$PYTHON" efficiency.py --datasets mini cub200 --encoder b16 2>&1 | tee work/efficiency.txt
fi
echo ROUND3_DONE "$STAGES" | tee -a $LOG
