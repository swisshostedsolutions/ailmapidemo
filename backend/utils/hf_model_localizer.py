import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def get_local_model(model_name, task_name, base_save_path="./local_model"):
    """
    Lädt ein Hugging Face Modell und Tokenizer.

    Prüft, ob das Modell bereits lokal unter dem `task_name` gespeichert ist.
    Wenn nicht, wird es von Hugging Face heruntergeladen und gespeichert.
    Ansonsten wird es vom lokalen Pfad geladen.

    Args:
        model_name (str): Der Name des Modells auf dem Hugging Face Hub 
                          (z.B. "distilbert-base-uncased-finetuned-sst-2-english").
        task_name (str): Ein kurzer Name für den Task, wird als Ordnername verwendet
                         (z.B. "sentiment-analysis").
        base_save_path (str, optional): Das Hauptverzeichnis, in dem die Modelle 
                                        gespeichert werden. Standard ist "./local_model".

    Returns:
        tuple: Ein Tupel mit dem geladenen Modell und Tokenizer.
    """
    save_path = os.path.join(base_save_path, task_name)
    
    print("\n" + "--- Loading model for task: " + task_name + "---")
    
    if not os.path.exists(save_path):
        print(f"Local model not found. Downloading '{model_name}' from Hugging Face Hub...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Erstellt die Ordner, falls sie nicht existieren
        os.makedirs(save_path, exist_ok=True)
        
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to '{save_path}'.")
    else:
        print(f"Loading model from local path: '{save_path}'...")
        model = AutoModelForSequenceClassification.from_pretrained(save_path)
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        print("Model loaded successfully.")
    
    return model, tokenizer

# Saving the huggingface pipeline model for sentiment
# analysis locally for offline use if not already present

# def local_load_hf_sentiment(save_path = "./local_model/sentiment"):

    
#     if not os.path.exists(save_path):
#         model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
#         tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
#         model.save_pretrained(save_path)
#         tokenizer.save_pretrained(save_path)
#         print("Model saved.")
#     else:
#         model = AutoModelForSequenceClassification.from_pretrained(save_path)
#         tokenizer = AutoTokenizer.from_pretrained(save_path)
#         print("Load model locally.")
    
#     return model, tokenizer

# def local_load_hf_zeroshotclassification(save_path = "./local_model/zeroshotclassification"):

    
#     if not os.path.exists(save_path):
#         model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
#         tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
#         model.save_pretrained(save_path)
#         tokenizer.save_pretrained(save_path)
#         print("Model saved.")
#     else:
#         model = AutoModelForSequenceClassification.from_pretrained(save_path)
#         tokenizer = AutoTokenizer.from_pretrained(save_path)
#         print("Load model locally.")
    
#     return model, tokenizer