# in backend/main.py
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from transformers import pipeline
from fastapi.encoders import jsonable_encoder

from .utils.hf_task_analyzer import get_full_task_details
from .utils.hf_model_localizer import get_pipeline_components

class PipelineRequest(BaseModel):
    task_name: str
    inputs: Dict[str, Any]

logging.getLogger("transformers").setLevel(logging.WARNING)
app = FastAPI(title="Hugging Face Pipeline Toolkit", version="2.0.0")
TASK_DETAILS = get_full_task_details()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Hugging Face Pipeline Toolkit!"}

@app.get("/tasks")
def get_all_tasks():
    return TASK_DETAILS

@app.post("/run-pipeline/")
def run_pipeline(request: PipelineRequest):
    if request.task_name not in TASK_DETAILS:
        raise HTTPException(status_code=404, detail=f"Task '{request.task_name}' not found.")

    model_name = TASK_DETAILS[request.task_name]["default_model"]
    
    try:
        pipeline_components = get_pipeline_components(
            model_name=model_name,
            task_name=request.task_name
        )
        
        classifier = pipeline(request.task_name, **pipeline_components)
        
        # --- THIS IS THE HYBRID CALL LOGIC ---
        # Most pipelines work best with keyword arguments.
        # We'll create a set of special cases that require positional arguments.
        POSITIONAL_ARG_TASKS = {"summarization", "text2text-generation", "text-generation"}

        if request.task_name in POSITIONAL_ARG_TASKS:
            # For these specific tasks, pass inputs as a list of values
            result = classifier(list(request.inputs.values()))
        else:
            # For all other tasks, unpack the inputs as keyword arguments
            result = classifier(**request.inputs)
        
        encoded_result = jsonable_encoder(result)
        return {"task": request.task_name, "result": encoded_result}

    except Exception as e:
        logging.error(f"Pipeline execution failed for task '{request.task_name}'", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))