import os
from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoImageProcessor,
)
from transformers.pipelines import SUPPORTED_TASKS

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(UTILS_DIR, '..')
DEFAULT_SAVE_PATH = os.path.join(BACKEND_DIR, 'local_model')

def get_pipeline_components(model_name, task_name, base_save_path=DEFAULT_SAVE_PATH):
    task_dir = os.path.join(base_save_path, task_name)
    print(f"\n--- Loading components for task: {task_name} ---")

    try:
        model_class = SUPPORTED_TASKS[task_name]["pt"][0]
    except KeyError:
        raise ValueError(f"Task '{task_name}' not supported.")

    components = {}
    component_types = {
        "model": model_class,
        "tokenizer": AutoTokenizer,
        "image_processor": AutoImageProcessor,
        "feature_extractor": AutoFeatureExtractor,
    }

    for name, component_class in component_types.items():
        component_path = os.path.join(task_dir, name)
        try:
            if os.path.exists(component_path):
                print(f"Loading {name} from local path...")
                components[name] = component_class.from_pretrained(component_path)
            else:
                print(f"Local {name} not found. Downloading from '{model_name}'...")
                component = component_class.from_pretrained(model_name)
                component.save_pretrained(component_path)
                components[name] = component
                print(f"{name} saved to '{component_path}'.")
        except Exception:
            print(f"--> Component '{name}' not applicable or found for this model.")
            pass
    
    if "model" not in components:
        raise RuntimeError(f"Failed to load the essential 'model' component for {model_name}.")

    print("Components loaded successfully.")
    return components