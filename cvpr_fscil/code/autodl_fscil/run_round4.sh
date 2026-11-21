#!/usr/bin/env bash
# Round 4 (2026-11-09). Resumable: completed tasks are skipped. Select stages with STAGES="A B C D".
#  A  four-source AnchorCE ablations (none / fixed / rawg), all encoders     -> work/readouts_anchorg_{none,fixed,rawg}
#  B  four-source AnchorCE with BiMC descriptions, all encoders             -> work/readouts_anchorg_desc
#  C  r x calibration-cap sweep, B/16 certified: 11 original readouts, GDA/KLDA, four-source AnchorCE
#  D  efficiency (miniImageNet, CUB-200, B/16); keep the GPU otherwise idle
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=${PYTHON:-/root/autodl-tmp/fscil-env/bin/python}
STAGES=${STAGES:-"A B C D"}
LOG=work/round4.log
if [[ $STAGES == *A* ]]; then
  for v in none fixed rawg; do "$PYTHON" anchor_gda.py --all --variant $v 2>&1 | tee -a $LOG; done
fi
if [[ $STAGES == *B* ]]; then
  "$PYTHON" anchor_gda.py --all --text desc 2>&1 | tee -a $LOG
fi
if [[ $STAGES == *C* ]]; then
  for ds in cifar100 mini cub200; do for seed in 0 1 2; do
    "$PYTHON" fscil_readouts.py --dataset $ds --encoder b16 --track certified --seed $seed --tag _sweep --sweep 2>&1 | tee -a $LOG
  done; done
  "$PYTHON" extra_baselines.py --all --sweep --encoders b16 --tracks certified 2>&1 | tee -a $LOG
  "$PYTHON" anchor_gda.py --all --sweep --encoders b16 --tracks certified 2>&1 | tee -a $LOG
  "$PYTHON" collect_sweep.py 2>&1 | tee work/summary_sweep.txt
fi
if [[ $STAGES == *D* ]]; then
  "$PYTHON" efficiency.py --datasets mini cub200 --encoder b16 2>&1 | tee work/efficiency.txt
fi
"$PYTHON" fscil_summary.py 2>&1 | tee work/summary_round4.txt
echo ROUND4_DONE "$STAGES" | tee -a $LOG
