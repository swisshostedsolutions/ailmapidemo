from transformers.pipelines import SUPPORTED_TASKS
def_models = {k: v["default"]["model"]["pt"][0] for k, v in SUPPORTED_TASKS.items() if "model" in v["default"].keys()}
print(def_models)
# {'audio-classification': 'superb/wav2vec2-base-superb-ks', 'automatic-speech-recognition': 'facebook/wav2vec2-base-960h', 'text-to-audio': 'suno/bark-small',
#  'feature-extraction': 'distilbert/distilbert-base-cased', 'text-classification': 'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
#  'token-classification': 'dbmdz/bert-large-cased-finetuned-conll03-english', 'question-answering': 'distilbert/distilbert-base-cased-distilled-squad',
#  'table-question-answering': 'google/tapas-base-finetuned-wtq', 'visual-question-answering': 'dandelin/vilt-b32-finetuned-vqa',
#  'document-question-answering': 'impira/layoutlm-document-qa', 'fill-mask': 'distilbert/distilroberta-base', 'summarization': 'sshleifer/distilbart-cnn-12-6',
#  'text2text-generation': 'google-t5/t5-base', 'text-generation': 'openai-community/gpt2', 'zero-shot-classification': 'facebook/bart-large-mnli',
#  'zero-shot-image-classification': 'openai/clip-vit-base-patch32', 'zero-shot-audio-classification': 'laion/clap-htsat-fused',
#  'image-classification': 'google/vit-base-patch16-224', 'image-feature-extraction': 'google/vit-base-patch16-224',
#  'image-segmentation': 'facebook/detr-resnet-50-panoptic', 'image-to-text': 'ydshieh/vit-gpt2-coco-en',
#  'object-detection': 'facebook/detr-resnet-50', 'zero-shot-object-detection': 'google/owlvit-base-patch32',
#  'depth-estimation': 'Intel/dpt-large', 'video-classification': 'MCG-NJU/videomae-base-finetuned-kinetics', 'mask-generation': 'facebook/sam-vit-huge', 'image-to-image': 'caidas/swin2SR-classical-sr-x2-64'} 

