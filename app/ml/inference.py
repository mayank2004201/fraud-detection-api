import joblib
import os
import pandas as pd
from typing import Dict, Any
from app.core.config import settings
from app.ml.preprocessing import preprocess_transaction

class FraudPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.threshold = None
        self.is_loaded = False
        
    def load_model(self):
        """Loads the serialized model artifact from disk."""
        if not os.path.exists(settings.MODEL_PATH):
            raise FileNotFoundError(f"Model artifact not found at {settings.MODEL_PATH}")
            
        artifact = joblib.load(settings.MODEL_PATH)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.feature_names = artifact["feature_names"]
        self.threshold = artifact.get("optimal_threshold", settings.CLASSIFICATION_THRESHOLD)
        self.is_loaded = True
        
    def predict(self, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Processes raw transaction data and returns fraud probability and classification.
        """
        if not self.is_loaded:
            self.load_model()
            
        # 1. Preprocess
        processed_data = preprocess_transaction(
            raw_data, 
            self.scaler, 
            self.feature_names
        )
        
        # 2. Inference
        probability = float(self.model.predict_proba(processed_data)[:, 1][0])
        is_fraud = bool(probability > self.threshold)
        
        return {
            "fraud_probability": round(probability, 4),
            "is_fraud": is_fraud,
            "threshold_used": self.threshold
        }

# Global instance for app-wide use
predictor = FraudPredictor()
