import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler
from app.core.config import settings

def preprocess_transaction(
    raw_data: pd.DataFrame, 
    scaler: StandardScaler, 
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Normalizes Time and Amount and ensures feature order matches training.
    
    Args:
        raw_data: DataFrame with raw transaction features.
        scaler: The fitted StandardScaler from the model artifact.
        feature_names: The exact list of features the model expects.
        
    Returns:
        Preprocessed DataFrame ready for inference.
    """
    df = raw_data.copy()
    
    # Scale features (consistent with trainer.py)
    if all(col in df.columns for col in settings.COLS_TO_SCALE):
        df[settings.COLS_TO_SCALE] = scaler.transform(df[settings.COLS_TO_SCALE])
    
    # Ensure all required features are present and in the correct order
    # Default to 0.0 for missing features (though API should validate)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
            
    return df[feature_names]
