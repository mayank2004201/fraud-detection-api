import os
import sys

# Ensure 'app' is discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ml.trainer import run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline()
