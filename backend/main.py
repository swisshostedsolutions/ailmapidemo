# main.py

import logging
from fastapi import FastAPI
from transformers.pipelines import SUPPORTED_TASKS

# Konfiguriere das Logging, um INFO-Meldungen von transformers zu unterdrücken
logging.getLogger("transformers").setLevel(logging.WARNING)

# Erstelle die FastAPI-Instanz
app = FastAPI(
    title="Hugging Face Tasks API",
    description="Eine API, um unterstützte Tasks und Modelle von Hugging Face abzurufen.",
    version="0.1.0",
)

# Lade die Modelldaten einmal beim Start
SUPPORTED_MODELS = {
    task: details["default"]["model"]["pt"][0]
    for task, details in SUPPORTED_TASKS.items()
    if "model" in details.get("default", {})
}

@app.get("/")
def read_root():
    """Gibt eine Willkommensnachricht zurück."""
    return {"message": "Willkommen bei der Hugging Face API!"}


@app.get("/tasks")
def get_supported_tasks():
    """
    Gibt eine Liste aller von der Pipeline unterstützten Tasks
    und deren Standard-Modellnamen zurück.
    """
    return SUPPORTED_MODELS