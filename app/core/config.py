import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Project Root and Paths
    PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
    MODEL_DIR: str = os.path.join(PROJECT_ROOT, "model")
    MODEL_PATH: str = os.path.join(MODEL_DIR, "model.pkl")
    # Model Hyperparameters (Research Optimiums)
    SEED: int = 42
    TEST_SIZE: float = 0.2
    MODEL_PARAMS: dict = {
        "max_depth": 5,
        "learning_rate": 0.2,
        "n_estimators": 250
    }
    
    # Feature Engineering
    COLS_TO_SCALE: list = ["Time", "Amount"]
    CLASSIFICATION_THRESHOLD: float = 0.3

settings = Settings()
