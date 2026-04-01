"""
Constants for the Fraud Detection API.
"""

# API Metadata
APP_NAME = "Fraud Detection API"
API_VERSION = "1.0.0"

# Risk Levels
RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"

# Models
XGB_MODEL_NAME = "XGBoost Classifier"
ISO_FOREST_NAME = "Isolation Forest Anomaly Detector"

# LLM Roles
LLM_ROLE_INVESTIGATOR = "Fraud Investigator"
LLM_ROLE_OVERRIDE = "Risk Override Specialist"
LLM_ROLE_QUERY_ENGINE = "Natural Language Query Engine"

# Dataset Constants
# The dataset has V1-V28 PCA features + Time and Amount
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
TARGET_COLUMN = "Class"
