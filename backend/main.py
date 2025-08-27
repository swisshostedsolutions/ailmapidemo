# main.py

import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from .utils.hf_model_localizer import get_local_model
from .utils.hf_task_analyzer import get_full_task_details


class PipelineRequest(BaseModel):
    task_name: str
    # text_input: str
    # This flexible dictionary can hold any arguments
    inputs: Dict[str, Any]


# Configuring the loudness of the logging messages
logging.getLogger("transformers").setLevel(logging.WARNING)

# Create the FastAPI application instance
app = FastAPI(
    title="Hugging Face Pipeline Toolkit",
    description="An API to support the use of Hugging Face's Transformers Pipelines",
    version="0.2.0",
)

# --- Main application startup logic ---
# Load all task details (models and signatures) into memory once at startup
TASK_DETAILS = get_full_task_details()



@app.get("/")
def read_root():
    """Returns a welcome message."""
    return {"message": "Welcome to the Hugging Face Pipeline Toolkit!"}


@app.get("/tasks")
def get_all_tasks():
    """
    Returns a full list of all supported tasks, including their
    default models and required input parameters.
    """
    return TASK_DETAILS


@app.post("/run-pipeline/")
def run_pipeline(request: PipelineRequest):
    """
    Executes a Hugging Face pipeline for a given task and input.
    """
    if request.task_name not in TASK_DETAILS:
        raise HTTPException(
            status_code=404, detail=f"Task '{request.task_name}' not found or not supported.")

    # Get the model name from the pre-loaded details
    model_name = TASK_DETAILS[request.task_name]["default_model"]

    # Delegate model loading to the utility function hf_model_localizer.py
    model, tokenizer = get_local_model(
        model_name=model_name,
        task_name=request.task_name
    )

    # Execute the pipeline in a try...except block for safety
    try:
        classifier = pipeline(request.task_name, model=model, tokenizer=tokenizer)
        # The ** operator unpacks the dictionary into keyword arguments.
        # e.g., {"sequences": "...", "candidate_labels": [...]} becomes
        # classifier(sequences="...", candidate_labels=[...])
        result = classifier(**request.inputs)
        return {"task": request.task_name, "result": result}
    except Exception as e:
        # If anything goes wrong, return a clean 500 error
        raise HTTPException(status_code=500, detail=str(e))