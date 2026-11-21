source /etc/network_turbo
cd /root/autodl-tmp/autodl_fscil
nohup /root/autodl-tmp/fscil-env/bin/python -u alternate_data.py >work/alternate_data.log 2>&1 </dev/null &
echo ALT_DATA_PID=$!
