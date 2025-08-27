# in scripts/discover_param_names.py
import inspect
import re
from transformers import pipeline
from transformers.pipelines import SUPPORTED_TASKS
# --- NEW IMPORTS ---
# Import our own model localizer
from backend.utils.hf_model_localizer import get_local_model
# We no longer need these for dummy data, but the pipeline might need them to load the assets
from PIL import Image
import numpy as np


# The file paths and dummy data factory remain the same
IMAGE_PATH = "test_assets/sample.jpg"
AUDIO_PATH = "test_assets/sample.wav"
VIDEO_PATH = "test_assets/sample.mp4"


def create_dummy_data(parameters):
    # ... (this function does not need to be changed)
    dummy_data = {}
    for param in parameters:
        name = param["name"]
        if "image" in name:
            dummy_data[name] = IMAGE_PATH
        elif "video" in name:
            dummy_data[name] = VIDEO_PATH
        elif "audio" in name:
            dummy_data[name] = AUDIO_PATH
        elif "inputs" in name:
            if "fill-mask" in parameters[0].get("type", ""):
                dummy_data[name] = "Hello, I'm a <mask> model."
            else:
                dummy_data[name] = "This is a test sentence."
        elif "candidate_labels" in name:
            dummy_data[name] = ["first label", "second label"]
        elif "example" in name:
            dummy_data[name] = {"question": "Who?", "context": "He did it."}
        elif name == 'sequences':
            dummy_data[name] = "This is a test sentence."
    return dummy_data


def discover_inconsistencies():
    print("Starting discovery and pre-caching of all models...")
    print("This will take a long time and download many models to backend/local_model/.\n")

    final_parameter_map = {}

    base_tasks = {}
    for task, details in SUPPORTED_TASKS.items():
        if "impl" in details and "default" in details and "model" in details["default"]:
            base_tasks[task] = details

    for task_name, details in base_tasks.items():
        print(f"--- Analyzing and Caching task: {task_name} ---")
        try:
            # --- KEY CHANGES START HERE ---

            # 1. Get the default model name
            model_name = details["default"]["model"]["pt"][0]

            # 2. Use our own function to download the model to the correct location
            #    We specify the path relative to the project root where we run the script.
            model, tokenizer = get_local_model(
                model_name=model_name,
                task_name=task_name,
                base_save_path="backend/local_model"
            )

            # 3. Initialize the pipeline using the pre-loaded model
            classifier = pipeline(task_name, model=model, tokenizer=tokenizer)

            # --- KEY CHANGES END HERE ---

            pipeline_class = details["impl"]
            signature = inspect.signature(pipeline_class.preprocess)

            introspected_params = []
            for param in signature.parameters.values():
                if param.name in ["self", "args", "kwargs"] or param.default is not inspect.Parameter.empty:
                    continue
                introspected_params.append({"name": param.name})

            dummy_input = create_dummy_data(introspected_params)

            # Try to call the pipeline
            classifier(**dummy_input)
            print(f"✅ '{task_name}' OK. Model cached in backend/local_model/.")

        except TypeError as e:
            # ... (error handling for inconsistency detection remains the same)
            error_str = str(e)
            match = re.search(
                r"missing \d+ required positional argument[s]*: '(\w+)'", error_str)
            if match:
                real_name = match.group(1)
                if introspected_params:
                    introspected_name = introspected_params[0]["name"]
                    if introspected_name != real_name:
                        print(
                            f"❗️ Found inconsistency for '{task_name}': Introspection found '{introspected_name}', but runtime needs '{real_name}'.")
                        if task_name not in final_parameter_map:
                            final_parameter_map[task_name] = {}
                        final_parameter_map[task_name][introspected_name] = real_name
            else:
                print(f"⚠️  Could not parse TypeError for '{task_name}': {e}")
        except Exception as e:
            print(
                f"❌ Failed to initialize or run pipeline for '{task_name}': {e}")

    print("\n\n--- Discovery Complete! ---")
    print("Copy the following dictionary into `backend/utils/task_analyzer.py`:\n")
    print("PARAMETER_NAME_MAP = {")
    for task, overrides in final_parameter_map.items():
        print(f'    "{task}": {overrides},')
    print("}")


if __name__ == "__main__":
    discover_inconsistencies()
