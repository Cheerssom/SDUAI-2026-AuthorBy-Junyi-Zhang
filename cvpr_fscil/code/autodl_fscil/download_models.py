from transformers import CLIPModel,CLIPTokenizer,CLIPImageProcessor
for en in ['base-patch16','large-patch14','base-patch32']:
 name='openai/clip-vit-'+en
 print('MODEL_START',name,flush=True)
 model=CLIPModel.from_pretrained(name);CLIPTokenizer.from_pretrained(name);CLIPImageProcessor.from_pretrained(name)
 del model
 print('MODEL_READY',name,flush=True)
