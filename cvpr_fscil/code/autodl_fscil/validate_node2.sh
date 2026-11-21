set -euo pipefail
cd /root/autodl-tmp/autodl_fscil
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/root/autodl-tmp/fscil-env/bin/python - <<'PY'
import torch,numpy,transformers,tokenizers,requests,PIL
from transformers import CLIPModel,CLIPTokenizer,CLIPImageProcessor
torch.manual_seed(2026);x=torch.randn(96,96,device='cuda');assert torch.isfinite(x@x).all()
print('KERNEL_WITNESS',torch.__version__,transformers.__version__,torch.cuda.get_device_name())
model=CLIPModel.from_pretrained('openai/clip-vit-base-patch32',local_files_only=True)
CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32',local_files_only=True)
CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32',local_files_only=True)
print('MODEL_CACHE_WITNESS',model.config.projection_dim)
PY
/root/autodl-tmp/fscil-env/bin/python fscil_readouts.py --help >/dev/null
/root/autodl-tmp/fscil-env/bin/python fscil_features.py --help >/dev/null
echo DOC_VALIDATION_PASSED
