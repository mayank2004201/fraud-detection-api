import os
import joblib
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from app.core.config import settings

def load_and_preprocess() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Internal: Loads and preprocesses transaction data."""
    if not os.path.exists(settings.DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {settings.DATA_PATH}")

    df = pd.read_csv(settings.DATA_PATH)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=settings.TEST_SIZE, 
        stratify=y, 
        random_state=settings.SEED
    )

    # Scaling features using centralized config
    scaler = StandardScaler()
    X_train[settings.COLS_TO_SCALE] = scaler.fit_transform(X_train[settings.COLS_TO_SCALE])
    X_test[settings.COLS_TO_SCALE] = scaler.transform(X_test[settings.COLS_TO_SCALE])

    return X_train, X_test, y_train, y_test, scaler

def run_training_pipeline():
    """High-level entry point to run the full training and persistence flow."""
    print("🚀 Starting Training Pipeline...")
    
    # 1. Prepare Data
    print("  -> Preparing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    # 2. Model Training
    print(f"  -> Training XGBoost (Params: {settings.MODEL_PARAMS})...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        **settings.MODEL_PARAMS,
        scale_pos_weight=scale_pos_weight,
        random_state=settings.SEED,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # 3. Anomaly Detection (Isolation Forest)
    print(f"  -> Training Isolation Forest (Params: {settings.ISOLATION_FOREST_PARAMS})...")
    iso_forest = IsolationForest(**settings.ISOLATION_FOREST_PARAMS)
    iso_forest.fit(X_train)

    # 4. Prediction & Evaluation
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > settings.CLASSIFICATION_THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "pr_auc": average_precision_score(y_test, probs),
        "roc_auc": roc_auc_score(y_test, probs)
    }

    print("\n  -> Model Metrics:")
    for name, value in metrics.items():
        print(f"     {name.capitalize():<12}: {value:.4f}")

    # 4. Drift Baseline
    feature_baseline = {
        col: {"mean": float(X_train[col].mean()), "std": float(X_train[col].std())}
        for col in X_train.columns
    }

    # 5. Persistence
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    artifact = {
        "model": model,
        "isolation_forest": iso_forest,
        "scaler": scaler,
        "feature_baseline": feature_baseline,
        "feature_names": X_train.columns.tolist(),
        "auprc": metrics["pr_auc"],
        "optimal_threshold": settings.CLASSIFICATION_THRESHOLD
    }
    
    joblib.dump(artifact, settings.MODEL_PATH)
    print(f"\n✅ Pipeline complete. Model saved to: {settings.MODEL_PATH}")
