set -euo pipefail
kill 2737
source /etc/network_turbo
cd /root/autodl-tmp/autodl_fscil
export FSCIL_IMAGE_WORKERS=6
export HF_HOME=/root/autodl-tmp/hf-cache
nohup /root/autodl-tmp/fscil-env/bin/python -u pipeline.py >>work/pipeline.log 2>&1 </dev/null &
echo NEW_PIPELINE_PID=$!
