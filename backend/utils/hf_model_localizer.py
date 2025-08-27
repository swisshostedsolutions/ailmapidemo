
# in backend/utils/hf_model_localizer.py
import os
# We now import AutoProcessor, our all-in-one tool
from transformers import AutoProcessor
from transformers.pipelines import SUPPORTED_TASKS

# The path logic remains the same
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(UTILS_DIR, '..')
DEFAULT_SAVE_PATH = os.path.join(BACKEND_DIR, 'local_model')


def get_local_model_and_processor(model_name, task_name, base_save_path=DEFAULT_SAVE_PATH):
    """
    Loads a Hugging Face model and its appropriate processor (tokenizer,
    image processor, feature extractor, or a combination) for the given task.
    """
    full_save_path = os.path.join(base_save_path, task_name)
    
    print(f"\n--- Loading components for task: {task_name}---")
    
    try:
        model_class = SUPPORTED_TASKS[task_name]["pt"][0]
    except KeyError:
        raise ValueError(f"Task '{task_name}' or its required model class is not supported.")

    if not os.path.exists(full_save_path):
        print(f"Local components not found. Downloading '{model_name}' from Hugging Face Hub...")
        
        # Load the specific model class and the generic processor
        model = model_class.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        
        os.makedirs(full_save_path, exist_ok=True)
        
        # Save both the model and the processor
        model.save_pretrained(full_save_path)
        processor.save_pretrained(full_save_path)
        print(f"Components saved to '{full_save_path}'.")
    else:
        print(f"Loading components from local path: '{full_save_path}'...")
        
        # Load both from the local path
        model = model_class.from_pretrained(full_save_path)
        processor = AutoProcessor.from_pretrained(full_save_path)
        print("Components loaded successfully.")
    
    # The function now returns a processor instead of just a tokenizer
    return model, processor