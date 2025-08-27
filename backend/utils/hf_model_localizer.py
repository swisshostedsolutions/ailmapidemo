# in backend/utils/hf_model_localizer.py
import os
from transformers import AutoTokenizer
# Correctly import SUPPORTED_TASKS from the 'pipelines' submodule
from transformers.pipelines import SUPPORTED_TASKS

# --- Build a robust, absolute path to the model directory ---
# This logic should be at the module level (outside the function).
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(UTILS_DIR, '..')
DEFAULT_SAVE_PATH = os.path.join(BACKEND_DIR, 'local_model')


def get_local_model(model_name, task_name, base_save_path=DEFAULT_SAVE_PATH):
    """
    Loads a Hugging Face model and tokenizer with the correct
    architecture for the given task.
    """
    full_save_path = os.path.join(base_save_path, task_name)
    
    print(f"\n--- Loading model for task: {task_name}---")
    
    try:
        # Look up the correct AutoModel class for the task
        model_class = SUPPORTED_TASKS[task_name]["pt"][0]
    except KeyError:
        raise ValueError(f"Task '{task_name}' or its required model class is not supported.")

    if not os.path.exists(full_save_path):
        print(f"Local model not found. Downloading '{model_name}' from Hugging Face Hub...")
        
        # Use the specific model_class we found to load the model
        model = model_class.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        os.makedirs(full_save_path, exist_ok=True)
        
        model.save_pretrained(full_save_path)
        tokenizer.save_pretrained(full_save_path)
        print(f"Model saved to '{full_save_path}'.")
    else:
        print(f"Loading model from local path: '{full_save_path}'...")
        
        # Also use the specific model_class here for loading from a local path
        model = model_class.from_pretrained(full_save_path)
        tokenizer = AutoTokenizer.from_pretrained(full_save_path)
        print("Model loaded successfully.")
    
    return model, tokenizer