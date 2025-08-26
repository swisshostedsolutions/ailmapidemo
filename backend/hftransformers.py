import torch
import logging
from utils.hf_model_localizer import get_local_model
from transformers import pipeline

# 🤫 SCHRITT 1: Unerwünschte INFO-Meldungen von transformers unterdrücken.
# Dies muss ganz am Anfang deines Skripts stehen.
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- 1. Gerät (Device) bestimmen ---
# Prüfe, ob eine CUDA-fähige GPU verfügbar ist, ansonsten nutze die CPU.
# Dies macht deinen Code portabel und zukunftssicher.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}") # Eigene, kontrollierte Ausgabe!

# --- Modell-Definitionen ---
# Hier definieren wir, welche Modelle wir für welche Tasks verwenden wollen.
# Das macht es einfach, später neue Modelle hinzuzufügen.
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
ZEROSHOT_MODEL = "facebook/bart-large-mnli"

# --- Modelle laden ---
# Wir verwenden unsere neue, universelle Funktion
sentiment_model, sentiment_tokenizer = get_local_model(
    model_name=SENTIMENT_MODEL, 
    task_name="sentiment-analysis"
)

zeroshot_model, zeroshot_tokenizer = get_local_model(
    model_name=ZEROSHOT_MODEL, 
    task_name="zero-shot-classification"
)
print("\n" + "="*30 + "\n")

# --- Test 1: Sentiment Analysis ---
print("--- Running Sentiment Analysis ---")
classifier_sentiment = pipeline(
    'sentiment-analysis', 
    model=sentiment_model, 
    tokenizer=sentiment_tokenizer,
    device=device
)
message_sentiment = classifier_sentiment(
    ["I just hate that stuff.",
     "This is so great!",
     "I am feeling neutral about this."]
)
print(message_sentiment)
print("\n" + "="*30 + "\n")


# --- Test 2: Zero-Shot Classification ---
print("--- Running Zero-Shot Classification ---")
classifier_zeroshot = pipeline(
    "zero-shot-classification", 
    model=zeroshot_model, 
    tokenizer=zeroshot_tokenizer,
    device=device
)
message_zeroshot = classifier_zeroshot(
    "This is a course about inflation",
    candidate_labels=["education", "politics", "economy"],
)
print(message_zeroshot)