import joblib
import os
from typing import Any
from app.core.config import settings

def load_artifact(file_path: str) -> Any:
    """Safely loads a joblib artifact."""
    if not os.path.exists(file_path):
        return None
    return joblib.load(file_path)

def get_model_metadata() -> dict:
    """Retrieves metadata from the saved model artifact."""
    artifact = load_artifact(settings.MODEL_PATH)
    if not artifact:
        return {}
        
    return {
        "auprc": artifact.get("auprc"),
        "features": artifact.get("feature_names"),
        "threshold": artifact.get("optimal_threshold")
    }
