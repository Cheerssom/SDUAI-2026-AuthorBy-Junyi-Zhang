"""Encode class descriptions into CLIP text features for the description-anchor experiment.

Input : descriptions/<dataset>.json  -- {"<class name>": ["sentence", ...], ...}
        Keys must match the class names stored in work/features/<dataset>/<encoder>/frozen.pt
        (matching ignores case, underscores and surrounding spaces); every class needs >= 1 sentence.
Output: work/description_features/<dataset>/<encoder>.pt  -- float tensor [num_classes, D], unit rows,
        each row the normalised mean of that class's normalised sentence embeddings.
Usage : python encode_descriptions.py --dataset cifar100 --encoder b16
"""
from pathlib import Path
import argparse, json, hashlib
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parent
D = R / 'work'
MODELS = {'b16': 'openai/clip-vit-base-patch16', 'b32': 'openai/clip-vit-base-patch32',
          'l14': 'openai/clip-vit-large-patch14'}


def key(name):
    return ' '.join(str(name).replace('_', ' ').lower().split())


@torch.inference_mode()
def main():
    a = argparse.ArgumentParser()
    a.add_argument('--dataset', required=True, choices=['cifar100', 'mini', 'cub200'])
    a.add_argument('--encoder', required=True, choices=list(MODELS))
    a = a.parse_args()
    z = torch.load(D / 'features' / a.dataset / a.encoder / 'frozen.pt', map_location='cpu', weights_only=False)
    names = [key(n) for n in z['names']]
    description_file=R/'descriptions'/f'{a.dataset}.json'
    raw=json.loads(description_file.read_text(encoding='utf-8'))
    assert all(isinstance(v,list) and v and all(isinstance(x,str) and x.strip() for x in v) for v in raw.values())
    table = {key(k): v for k, v in raw.items()}
    assert len(table)==len(raw), 'Normalized description keys collide'
    missing = [n for n in names if not table.get(n)]
    assert not missing, f'{len(missing)} classes lack descriptions, e.g. {missing[:5]}'
    from transformers import CLIPModel, CLIPTokenizer
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    model = CLIPModel.from_pretrained(MODELS[a.encoder],local_files_only=True).cuda().eval()
    tok = CLIPTokenizer.from_pretrained(MODELS[a.encoder],local_files_only=True)
    rows = []
    for n in names:
        sents = [s.strip() for s in table[n] if s.strip()]
        b = tok(sents, padding=True, truncation=True, max_length=77, return_tensors='pt').to('cuda')
        e = F.normalize(model.get_text_features(**b).float(), dim=1)
        rows.append(F.normalize(e.mean(0), dim=0).cpu())
    out = D / 'description_features' / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    feats = torch.stack(rows)
    assert feats.shape[0] == z['T'].shape[0] and feats.shape[1] == z['T'].shape[1], (feats.shape, z['T'].shape)
    temporary=out/f'{a.encoder}.tmp.pt';torch.save(feats,temporary);temporary.replace(out/f'{a.encoder}.pt')
    meta=dict(complete=True,encoder=a.encoder,model=MODELS[a.encoder],classes=len(names),names_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest(),descriptions_sha256=hashlib.sha256(description_file.read_bytes()).hexdigest(),cache_sha256=hashlib.sha256((out/f'{a.encoder}.pt').read_bytes()).hexdigest(),sentences_per_class=[len(table[n]) for n in names],source='BiMC official, commit b4ef2ea01b268098631f75ace4e716daff75db6b')
    (out/f'{a.encoder}.json').write_text(json.dumps(meta,indent=2))
    print('DESCRIPTIONS_ENCODED', a.dataset, a.encoder, tuple(feats.shape),
          'mean sentences/class %.1f' % (sum(len(table[n]) for n in names) / len(names)), flush=True)


if __name__ == '__main__':
    main()
