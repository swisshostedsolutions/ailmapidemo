# in backend/main.py
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from transformers import pipeline

from .utils.hf_model_localizer import get_local_model_and_processor
from .utils.hf_task_analyzer import get_full_task_details

# --- Pydantic model is correct ---
class PipelineRequest(BaseModel):
    task_name: str
    inputs: Dict[str, Any]

# --- Logging and App setup are correct ---
logging.getLogger("transformers").setLevel(logging.WARNING)
app = FastAPI(
    title="Hugging Face Pipeline Toolkit",
    description="A dynamic API to discover and run Hugging Face pipelines.",
    version="1.5.0",
)

# --- Startup logic is correct ---
TASK_DETAILS = get_full_task_details()


# --- THIS IS THE NEW, ROBUST HELPER FUNCTION ---
def create_pipeline_components(processor):
    """
    Inspects the processor and builds the correct component dictionary
    to pass to the pipeline factory.
    """
    components = {}
    # Safely extract the specific components if they exist on the processor object
    if hasattr(processor, 'image_processor'):
        components['image_processor'] = processor.image_processor
    if hasattr(processor, 'feature_extractor'):
        components['feature_extractor'] = processor.feature_extractor
    if hasattr(processor, 'tokenizer'):
        components['tokenizer'] = processor.tokenizer
    
    # Fallback for simple processors that are tokenizer-like
    if not components:
        components['tokenizer'] = processor
            
    return components


@app.get("/")
def read_root():
    return {"message": "Welcome to the Hugging Face Pipeline Toolkit!"}


@app.get("/tasks")
def get_all_tasks():
    return TASK_DETAILS


@app.post("/run-pipeline/")
def run_pipeline(request: PipelineRequest):
    """Executes a Hugging Face pipeline for a given task and input."""
    if request.task_name not in TASK_DETAILS:
        raise HTTPException(status_code=404, detail=f"Task '{request.task_name}' not found.")

    model_name = TASK_DETAILS[request.task_name]["default_model"]
    
    model, processor = get_local_model_and_processor(
        model_name=model_name,
        task_name=request.task_name
    )

    try:
        # --- THIS IS THE FINAL, CORRECTED PIPELINE CREATION ---
        # 1. Use the helper to get the precise components
        pipeline_components = create_pipeline_components(processor)
        
        # 2. Unpack the components into the pipeline call
        classifier = pipeline(request.task_name, model=model, **pipeline_components)
        
        result = classifier(**request.inputs)
        return {"task": request.task_name, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))