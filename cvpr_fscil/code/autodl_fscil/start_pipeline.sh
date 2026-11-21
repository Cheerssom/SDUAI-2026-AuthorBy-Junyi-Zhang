set -euo pipefail
source /etc/network_turbo
cd /root/autodl-tmp/autodl_fscil
nohup /root/autodl-tmp/fscil-env/bin/python -u pipeline.py >work/pipeline.log 2>&1 </dev/null &
echo PIPELINE_PID=$!
