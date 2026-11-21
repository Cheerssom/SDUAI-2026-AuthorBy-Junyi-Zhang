#!/usr/bin/env bash
# Usage: bash run_all.sh   (resumable: finished features/readouts are skipped)
set -e
cd "$(dirname "$0")"
mkdir -p work
DATA=${DATA:-/root/autodl-tmp/data}
ENCODERS=${ENCODERS:-"b16 l14 b32"}
declare -A ROOT=( [cifar100]=$DATA/cifar-100-python [mini]=$DATA/miniimagenet [cub200]=$DATA/cub )
declare -A IDX=( [cifar100]=$DATA/index_list/cifar100 [mini]=$DATA/index_list/mini_imagenet [cub200]=$DATA/index_list/cub200 )
for enc in $ENCODERS; do
  for ds in cifar100 mini cub200; do
    python fscil_features.py --dataset $ds --encoder $enc --root ${ROOT[$ds]} --index_dir ${IDX[$ds]} 2>&1 | tee -a work/log_features.txt
  done
done
for enc in $ENCODERS; do
  for ds in cifar100 mini cub200; do
    for track in standard certified; do
      for seed in 0 1 2; do
        python fscil_readouts.py --dataset $ds --encoder $enc --track $track --seed $seed 2>&1 | tee -a work/log_readouts.txt
      done
    done
  done
done
python fscil_summary.py | tee work/summary.txt
