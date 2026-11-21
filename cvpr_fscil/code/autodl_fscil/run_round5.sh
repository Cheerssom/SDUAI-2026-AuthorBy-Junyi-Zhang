#!/usr/bin/env bash
# Round 5 (2026-11-10). Resumable: completed tasks are skipped. Select stages with STAGES="A B C D".
#  A  plug-in base anchoring: HyCal / GDA original vs anchored, B/16, both tracks       -> work/readouts_plugin
#  B  four-source AnchorCE certified against its own closed head (common reference)    -> work/readouts_anchorg_ref4
#  C  GDA / KLDA certified against the four-source AnchorCE closed head                -> work/readouts_extra_ref4
#  D  the eleven original readouts certified against the four-source closed head       -> work/readouts_ref4 (FeCAM on CPU, slowest)
# Every certified run also stores the r x calibration-cap sweep. Requires work/readouts_anchorg (round of 2026-11-09).
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=${PYTHON:-/root/autodl-tmp/fscil-env/bin/python}
STAGES=${STAGES:-"A B C D"}
LOG=work/round5.log
test -d work/readouts_anchorg || { echo "missing work/readouts_anchorg" | tee -a $LOG; exit 1; }
if [[ $STAGES == *A* ]]; then
  "$PYTHON" plugin_anchor.py --all --encoders b16 2>&1 | tee -a $LOG
  "$PYTHON" collect_plugin.py 2>&1 | tee work/summary_plugin.txt
fi
if [[ $STAGES == *B* ]]; then
  "$PYTHON" anchor_gda.py --all --ref4 --tracks certified 2>&1 | tee -a $LOG
fi
if [[ $STAGES == *C* ]]; then
  "$PYTHON" extra_baselines.py --all --ref4 --tracks certified 2>&1 | tee -a $LOG
fi
if [[ $STAGES == *D* ]]; then
  for ds in cifar100 mini cub200; do for enc in b16 l14 b32; do for seed in 0 1 2; do
    "$PYTHON" fscil_readouts.py --dataset $ds --encoder $enc --track certified --seed $seed --tag _ref4 --ref4 2>&1 | tee -a $LOG
  done; done; done
fi
"$PYTHON" fscil_summary.py _ref4 2>&1 | tee work/summary_ref4.txt
"$PYTHON" collect_sweep.py _ref4 2>&1 | tee work/summary_sweep_ref4.txt
echo ROUND5_DONE "$STAGES" | tee -a $LOG
