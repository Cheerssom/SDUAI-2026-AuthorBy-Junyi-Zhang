set -euo pipefail
source /etc/network_turbo
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/autodl_fscil
python -m venv --system-site-packages /root/autodl-tmp/fscil-env
/root/autodl-tmp/fscil-env/bin/python -m pip install --no-cache-dir transformers==4.44.2 huggingface_hub==0.25.2 safetensors==0.4.5 gdown==5.2.0
/root/autodl-tmp/fscil-env/bin/python -c "import torch,transformers,numpy;torch.manual_seed(2026);x=torch.randn(96,96,device='cuda');print('ENV_READY',torch.__version__,transformers.__version__,torch.isfinite(x@x).all().item())"
