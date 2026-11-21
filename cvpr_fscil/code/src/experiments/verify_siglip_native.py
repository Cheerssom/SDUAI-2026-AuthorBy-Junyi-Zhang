"""Native preprocessing/logit witness for the pinned cross-encoder cache."""
from pathlib import Path
import json,sys
import torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import CFG
from transformers import SiglipModel,SiglipTokenizer,SiglipImageProcessor
from experiments.kbs_runtime import configure,atomic_json,digest_file
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_cross_encoder'

@torch.inference_mode()
def main():
    device=configure();modeldir=OUT/'model';h=OUT/'features'
    model=SiglipModel.from_pretrained(str(modeldir),local_files_only=True,use_safetensors=True).to(device).eval()
    tokenizer=SiglipTokenizer.from_pretrained(str(modeldir),local_files_only=True)
    processor=SiglipImageProcessor.from_pretrained(str(modeldir),local_files_only=True)
    source=json.loads((ROOT/'experiments/kbs_imagenetr/data_manifest.json').read_text())
    images=[]
    for r in source['pools']['test'][:4]:
        with Image.open(CFG.data_root/'imagenet-r'/r['path']) as im:images.append(im.convert('RGB'))
    px=processor(images=images,return_tensors='pt').to(device)
    text=tokenizer(['a photo of a '+n+'.' for n in source['class_names'][:4]],padding='max_length',max_length=64,return_tensors='pt').to(device)
    assert text['input_ids'].shape==(4,64)
    image=F.normalize(model.get_image_features(**px).float(),dim=1)
    words=F.normalize(model.get_text_features(**text).float(),dim=1)
    manual=(image@words.T)*model.logit_scale.exp()+model.logit_bias
    native=model(**px,**text).logits_per_image.float()
    error=float((manual-native).abs().max());assert error<1e-3,error
    cache=torch.load(h/'frozen.pt',map_location='cpu',weights_only=False)
    cosine=F.cosine_similarity(image.cpu(),cache['E_test'][:4]);assert float(cosine.min())>.999
    result=dict(complete=True,cache_sha256=digest_file(h/'frozen.pt'),native_logit_max_absolute_error=error,
                native_float32_vs_cached_autocast_min_cosine=float(cosine.min()),text_padding=64,
                native_tokenizer=True,preprocessing='model-native SiglipImageProcessor',
                common_cosine_interface='Native scale/bias omitted consistently for all readouts; ranking witness verified.')
    atomic_json(result,h/'native_witness.json');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
