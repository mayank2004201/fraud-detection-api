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
    
    # Anomaly Detection (Isolation Forest)
    ISOLATION_FOREST_PARAMS: dict = {
        "n_estimators": 100,
        "contamination": "auto",
        "random_state": 42
    }

    # LLM Settings (Groq API)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "default-secret-change-me")
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    
    # Risk Override Settings (Role 2)
    DATABASE_PATH: str = os.path.join(PROJECT_ROOT, "llm_decisions.db")
    UNCERTAIN_ZONE_LOW: float = 0.4
    UNCERTAIN_ZONE_HIGH: float = 0.7

settings = Settings()
