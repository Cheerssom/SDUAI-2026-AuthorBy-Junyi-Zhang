set -euo pipefail
source /etc/network_turbo
cd /root/autodl-tmp/autodl_fscil
export PATH=/root/autodl-tmp/fscil-env/bin:/root/miniconda3/bin:$PATH
export HF_HOME=/root/autodl-tmp/hf-cache
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
nohup python -u prepare_data.py >work/prepare_data.log 2>&1 </dev/null &
echo PREPARE_PID=$!
