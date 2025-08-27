# in scripts/discover_param_names.py
import inspect
import re
import json
from transformers import pipeline
from transformers.pipelines import SUPPORTED_TASKS
from backend.utils.hf_model_localizer import get_local_model

# NEW: A set of non-text tasks to exclude from our V1.0
EXCLUDED_TASKS = {
    "audio-classification", "automatic-speech-recognition", "text-to-audio",
    "image-classification", "image-feature-extraction", "image-segmentation",
    "image-to-text", "object-detection", "video-classification", "depth-estimation",
    "visual-question-answering", "document-question-answering", "zero-shot-image-classification",
    "zero-shot-audio-classification", "zero-shot-object-detection", "mask-generation", "image-to-image",
    "image-text-to-text"
}

# The data factory is now much simpler
def create_dummy_data(parameters):
    dummy_data = {}
    for param in parameters:
        name = param["name"]
        if "candidate_labels" in name:
            dummy_data[name] = ["first label", "second label"]
        elif "example" in name:
            dummy_data[name] = {"question": "Who?", "context": "He did it."}
        else:
            # A generic fallback for all other text inputs
            dummy_data[name] = "This is a test sentence with a <mask> token."
    return dummy_data


def discover_inconsistencies():
    print("Starting discovery for TEXT-BASED pipelines...")
    print("This will download necessary models to backend/local_model/.\n")
    
    final_parameter_map = {}

    base_tasks = {
        task: details for task, details in SUPPORTED_TASKS.items()
        if "impl" in details and "default" in details and "model" in details["default"]
    }

    for task_name, details in base_tasks.items():
        # NEW: Skip any tasks in our exclusion list
        if task_name in EXCLUDED_TASKS:
            continue

        print(f"--- Analyzing and Caching task: {task_name} ---")
        try:
            pipeline_class = details["impl"]
            signature = inspect.signature(pipeline_class.preprocess)
            
            introspected_params = []
            for param in signature.parameters.values():
                if param.name in ["self", "args", "kwargs"] or param.default is not inspect.Parameter.empty:
                    continue
                introspected_params.append({"name": param.name})

            dummy_input = create_dummy_data(introspected_params)
            
            model_name = details["default"]["model"]["pt"][0]
            model, tokenizer = get_local_model(
                model_name=model_name, 
                task_name=task_name, 
                base_save_path="backend/local_model"
            )

            classifier = pipeline(task_name, model=model, tokenizer=tokenizer)
            classifier(**dummy_input)
            print(f"✅ '{task_name}' parameters seem consistent.")

        except TypeError as e:
            error_str = str(e)
            match = re.search(r"missing \d+ required positional argument[s]*: '(\w+)'", error_str)
            if match:
                real_name = match.group(1)
                if introspected_params:
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