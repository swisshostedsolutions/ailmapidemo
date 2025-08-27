import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from transformers import pipeline
from fastapi.encoders import jsonable_encoder

from .utils.hf_model_localizer import get_pipeline_components
from .utils.hf_task_analyzer import get_full_task_details

class PipelineRequest(BaseModel):
    task_name: str
    inputs: Dict[str, Any]

logging.getLogger("transformers").setLevel(logging.WARNING)
app = FastAPI(title="Hugging Face Pipeline Toolkit", version="2.0.0")
TASK_DETAILS = get_full_task_details()

def create_pipeline_components(loaded_components):
    components = {"model": loaded_components.get("model")}
    if "tokenizer" in loaded_components: components['tokenizer'] = loaded_components['tokenizer']
    if "image_processor" in loaded_components: components['image_processor'] = loaded_components['image_processor']
    if "feature_extractor" in loaded_components: components['feature_extractor'] = loaded_components['feature_extractor']
    return components

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
    
    loaded_components = get_pipeline_components(model_name=model_name, task_name=request.task_name)
    
    try:
        pipeline_components = create_pipeline_components(loaded_components)
        classifier = pipeline(request.task_name, **pipeline_components)
        result = classifier(**request.inputs)
        
        # Use FastAPI's jsonable_encoder to ensure the result is serializable
        encoded_result = jsonable_encoder(result)
        
        return {"task": request.task_name, "result": encoded_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))