import inspect
from transformers.pipelines import SUPPORTED_TASKS

# Map for known parameter name inconsistencies 
# from /scripts/discover_param_names.py
PARAMETER_NAME_MAP = {
    "text-to-audio": {'text': 'text_inputs'},
    "token-classification": {'sentence': 'inputs'},
    "document-question-answering": {'input': 'image'},
    "zero-shot-classification": {'inputs': 'sequences'},
}


def get_full_task_details():
    """
    Analyzes all supported tasks to extract their default model and
    the signature of their input parameters.

    Returns:
        dict: A dictionary where each key is a task name and the value
              contains the default model and a list of parameters.
    """
    task_details = {}

    # First, get the basic list of tasks and default models
    base_tasks = {
        task: details["default"]["model"]["pt"][0]
        for task, details in SUPPORTED_TASKS.items()
        if "model" in details.get("default", {})
    }

    # Now, iterate through the tasks and inspect each one
    for task_name, model_name in base_tasks.items():
        try:
            # Get the pipeline class for the task
            pipeline_class = SUPPORTED_TASKS[task_name]["impl"]
            # 4. Use 'inspect' to get the signature of the preprocess method
            signature = inspect.signature(pipeline_class.preprocess)

            parameters = []
            for param in signature.parameters.values():
                # Ignore internal parameters
                if param.name in ["self", "args", "kwargs"]:
                    continue

                # Apply our manual corrections from PARAMETER_NAME_MAP
                introspected_name = param.name
                correct_name = PARAMETER_NAME_MAP.get(task_name, {}).get(introspected_name, introspected_name)

                parameters.append({
                    "name": correct_name,
                    "type": str(param.annotation)if param.annotation is not inspect.Parameter.empty else "any",
                    "default_value": param.default if param.default is not inspect.Parameter.empty else "REQUIRED"
                })

            # Assemble the final data for this task
            task_details[task_name] = {
                "default_model": model_name,
                "parameters": parameters
            }
        except Exception as e:
            # Skip any tasks that cause an error during inspection
            print(f"--> Could not inspect task '{task_name}'. Error: {e}")
            continue
            
    return task_details