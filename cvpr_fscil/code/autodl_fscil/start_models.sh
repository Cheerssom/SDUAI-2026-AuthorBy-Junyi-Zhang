set -euo pipefail
source /etc/network_turbo
cd /root/autodl-tmp/autodl_fscil
export HF_HOME=/root/autodl-tmp/hf-cache
export OMP_NUM_THREADS=8
nohup /root/autodl-tmp/fscil-env/bin/python -u download_models.py >work/models.log 2>&1 </dev/null &
echo MODELS_PID=$!
