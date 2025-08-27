# in scripts/discover_param_names.py
import inspect
import re
import json
from transformers import pipeline
from transformers.pipelines import SUPPORTED_TASKS
from backend.utils.hf_model_localizer import get_local_model_and_processor
import pandas as pd

# Test asset paths
IMAGE_PATH = "test_assets/sample.jpg"
AUDIO_PATH = "test_assets/sample.wav"
VIDEO_PATH = "test_assets/sample.mp4"

def create_dummy_data(task_name, parameters):
    """Creates a dictionary of plausible dummy data based on the task and parameter names."""
    if task_name == "question-answering":
        return {"question": "Who?", "context": "He did it."}
    if task_name == "table-question-answering":
        table = pd.DataFrame.from_dict({"actors": ["brad pitt"], "movies": ["once upon a time in hollywood"]})
        return {"query": "how many movies?", "table": table}
    if task_name == "fill-mask":
        return {"inputs": "The capital of France is <mask>."}

    dummy_data = {}
    for param in parameters:
        name = param["name"]
        if "image" in name: dummy_data[name] = IMAGE_PATH
        elif "video" in name: dummy_data[name] = VIDEO_PATH
        elif "audio" in name: dummy_data[name] = AUDIO_PATH
        elif "candidate_labels" in name: dummy_data[name] = ["first", "second"]
        else: dummy_data.setdefault(name, "This is a test sentence.")
    return dummy_data

def create_pipeline_components(model, processor):
    """Inspects the processor and builds the correct component dictionary for the pipeline."""
    components = {"model": model}
    if hasattr(processor, 'tokenizer'): components['tokenizer'] = processor
    if hasattr(processor, 'image_processor'): components['image_processor'] = processor
    if hasattr(processor, 'feature_extractor'): components['feature_extractor'] = processor
    if len(components) == 1: components['tokenizer'] = processor
    return components

def discover_inconsistencies():
    print("Starting final discovery and pre-caching for ALL pipelines...")
    final_parameter_map = {}
    base_tasks = {
        task: details for task, details in SUPPORTED_TASKS.items()
        if "impl" in details and "default" in details and "model" in details["default"]
    }

    for task_name, details in base_tasks.items():
        print(f"--- Analyzing and Caching task: {task_name} ---")
        try:
            model_name = details["default"]["model"]["pt"][0]
            model, processor = get_local_model_and_processor(model_name=model_name, task_name=task_name)
            
            pipeline_components = create_pipeline_components(model, processor)
            classifier = pipeline(task_name, **pipeline_components)

            pipeline_class = details["impl"]
            signature = inspect.signature(pipeline_class.preprocess)
            introspected_params = [
                {"name": param.name}
                for param in signature.parameters.values()
                if param.name not in ["self", "args", "kwargs"] and param.default is inspect.Parameter.empty
            ]
            dummy_input = create_dummy_data(task_name, introspected_params)

            classifier(**dummy_input)
            print(f"✅ '{task_name}' parameters seem consistent.")

        except TypeError as e:
            error_str = str(e)
            match = re.search(r"missing \d+ required positional argument[s]*: '(\w+)'", error_str)
            if match and introspected_params:
                real_name = match.group(1)
                introspected_name = introspected_params[0]["name"]
                if introspected_name != real_name:
                    print(f"❗️ Found inconsistency for '{task_name}': Introspection found '{introspected_name}', but runtime needs '{real_name}'.")
                    if task_name not in final_parameter_map:
                        final_parameter_map[task_name] = {}
                    final_parameter_map[task_name][introspected_name] = real_name
            else:
                print(f"⚠️  Could not parse TypeError for '{task_name}': {e}")
        except Exception as e:
            print(f"❌ Failed to initialize or run pipeline for '{task_name}': {e}")
        
    print("\n\n--- Discovery Complete! ---")
    print("Copy the following dictionary into `backend/utils/task_analyzer.py`:\n")
    print(f"PARAMETER_NAME_MAP = {json.dumps(final_parameter_map, indent=4)}")

if __name__ == "__main__":
    discover_inconsistencies()