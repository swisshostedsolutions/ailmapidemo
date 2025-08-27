# main.py

import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from .utils.hf_model_localizer import get_local_model
# from utils.hf_get_pipeline_models import get_supported_models
from .utils.hf_task_analyzer import get_full_task_details


class PipelineRequest(BaseModel):
    task_name: str
    # text_input: str
    # This flexible dictionary can hold any arguments
    inputs: Dict[str, Any]


# Konfiguriere das Logging, um INFO-Meldungen von transformers zu unterdrücken
logging.getLogger("transformers").setLevel(logging.WARNING)

# Erstelle die FastAPI-Instanz
app = FastAPI(
    title="Hugging Face Tasks API",
    description="Eine API, um unterstützte Tasks und Modelle von Hugging Face abzurufen.",
    version="0.2.0",
)

# Lade die Modelldaten einmal beim Start
# SUPPORTED_MODELS = get_supported_models()

# --- Simplified Startup Logic ---
# Load all task details (models and signatures) once at startup
TASK_DETAILS = get_full_task_details()



@app.get("/")
def read_root():
    """Gibt eine Willkommensnachricht zurück."""
    return {"message": "Willkommen bei der Hugging Face API!"}


@app.get("/tasks")
def get_all_tasks():
    """
    Gibt eine Liste aller von der Pipeline unterstützten Tasks
    und deren Standard-Modellnamen zurück.
    """
    return TASK_DETAILS


@app.post("/run-pipeline/")
def run_pipeline(request: PipelineRequest):
    """
    Führt eine Hugging Face Pipeline für einen gegebenen Task und Input aus.
    """
    if request.task_name not in TASK_DETAILS:
        raise HTTPException(
            status_code=404, detail=f"Task '{request.task_name}' not found or not supported.")

    # Modellnamen aus unserer Liste holen
    model_name = TASK_DETAILS[request.task_name]["default_model"]

    # Modell laden (lokal oder von HF) mit deiner existierenden Funktion
    model, tokenizer = get_local_model(
        model_name=model_name,
        task_name=request.task_name
    )

    # Pipeline erstellen und ausführen
    try:
        classifier = pipeline(
            request.task_name, model=model, tokenizer=tokenizer)
        # This is the key change!
        # The ** operator unpacks the dictionary into keyword arguments.
        # e.g., {"sequences": "...", "candidate_labels": [...]} becomes
        # classifier(sequences="...", candidate_labels=[...])
        result = classifier(**request.inputs)
        # result = classifier(request.text_input)
        return {"task": request.task_name, "result": result}
    except Exception as e:
        # Fängt generelle Fehler bei der Pipeline-Ausführung ab
        raise HTTPException(status_code=500, detail=str(e))