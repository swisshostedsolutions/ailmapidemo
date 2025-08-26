from transformers.pipelines import SUPPORTED_TASKS

def get_supported_models():
    """
    Extrahiert alle von der Pipeline unterstützten Tasks und deren
    Standard-Modellnamen aus der transformers-Bibliothek.
    """
    return {
        task: details["default"]["model"]["pt"][0]
        for task, details in SUPPORTED_TASKS.items()
        if "model" in details.get("default", {})
    }