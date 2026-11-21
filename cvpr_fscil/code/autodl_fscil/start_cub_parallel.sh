source /etc/network_turbo
cd /root/autodl-tmp/autodl_fscil
nohup /root/autodl-tmp/fscil-env/bin/python -u cub_parallel.py >work/cub_parallel.log 2>&1 </dev/null &
echo CUB_PID=$!
