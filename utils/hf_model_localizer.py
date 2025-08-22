import os
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

classifier = pipeline("sentiment-analysis")
print(classifier.model.name_or_path)
print(classifier.tokenizer.name_or_path)

classifier = pipeline("zero-shot-classification")
print(classifier.model.name_or_path)
print(classifier.tokenizer.name_or_path)


# Saving the huggingface pipeline model for sentiment
# analysis locally for offline use if not already present

def local_load_hf_sentiment(save_path = "./local_model/sentiment"):

    
    if not os.path.exists(save_path):
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("Model saved.")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(save_path)
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        print("Load model locally.")
    
    return model, tokenizer

def local_load_hf_zeroshotclassification(save_path = "./local_model/zeroshotclassification"):

    
    if not os.path.exists(save_path):
        model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("Model saved.")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(save_path)
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        print("Load model locally.")
    
    return model, tokenizer