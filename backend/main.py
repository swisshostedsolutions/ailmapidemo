# main.py

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from utils.hf_model_localizer import get_local_model
from utils.hf_get_pipeline_models import get_supported_models


class PipelineRequest(BaseModel):
    task_name: str
    text_input: str


# Konfiguriere das Logging, um INFO-Meldungen von transformers zu unterdrücken
logging.getLogger("transformers").setLevel(logging.WARNING)

# Erstelle die FastAPI-Instanz
app = FastAPI(
    title="Hugging Face Tasks API",
    description="Eine API, um unterstützte Tasks und Modelle von Hugging Face abzurufen.",
    version="0.1.0",
)

# Lade die Modelldaten einmal beim Start
SUPPORTED_MODELS = get_supported_models()


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


@app.post("/run-pipeline/")
def run_pipeline(request: PipelineRequest):
    """
    Führt eine Hugging Face Pipeline für einen gegebenen Task und Input aus.
    """
    if request.task_name not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=404, detail=f"Task '{request.task_name}' not found or not supported.")

    # Modellnamen aus unserer Liste holen
    model_name = SUPPORTED_MODELS[request.task_name]

    # Modell laden (lokal oder von HF) mit deiner existierenden Funktion
    model, tokenizer = get_local_model(
        model_name=model_name,
        task_name=request.task_name
    )

    # Pipeline erstellen und ausführen
    try:
        classifier = pipeline(
            request.task_name, model=model, tokenizer=tokenizer)
        result = classifier(request.text_input)
        return {"task": request.task_name, "result": result}
    except Exception as e:
        # Fängt generelle Fehler bei der Pipeline-Ausführung ab
        raise HTTPException(status_code=500, detail=str(e))
