import os

# Folder + file structure
structure = {
    "app": [
        "__init__.py",
        "main.py",
    ],
    "app/core": [
        "__init__.py",
        "config.py",
        "constants.py",
        "logger.py",
        "logging_config.py",
        "security.py",
    ],
    "app/api": [
        "__init__.py",
        "routes.py",
        "dependencies.py",
        "middleware.py",
    ],
    "app/schemas": [
        "__init__.py",
        "schemas.py",
    ],
    "app/services": [
        "__init__.py",
        "risk_engine.py",
        "analysis_service.py",
        "llm_query_service.py",
    ],
    "app/ml": [
        "__init__.py",
        "inference.py",
        "preprocessing.py",
        "utils.py",
    ],
    "app/llm": [
        "__init__.py",
        "client.py",
        "llm_investigator.py",
        "llm_override.py",
        "llm_query.py",
        "prompts.py",
    ],
    "app/storage": [
        "__init__.py",
        "storage_handler.py",
    ],
    "monitoring": [
        "__init__.py",
        "drift.py",
        "logging_monitor.py",
    ],
    "logs": [
        "app.log",
        "error.log",
    ],
    "model": [
        ".gitkeep",
    ],
    "training": [
        "train.ipynb",
        "pipeline.py",
    ],
    "tests": [
        "__init__.py",
        "test_api.py",
        "test_ml.py",
        "test_llm.py",
        "test_storage.py",
    ],
    "scripts": [
        "setup_structure.py",
    ],
    "": [  # root-level files
        "Dockerfile",
        "render.yaml",
        "requirements.txt",
        ".env.example",
        # intentionally excluding README.md and .gitignore
    ],
}


def create_structure(base_path="."):
    for folder, files in structure.items():
        folder_path = os.path.join(base_path, folder)

        # Create directory if not root
        if folder:
            os.makedirs(folder_path, exist_ok=True)

        for file in files:
            file_path = os.path.join(folder_path, file) if folder else os.path.join(base_path, file)

            # Skip if file already exists
            if os.path.exists(file_path):
                continue

            # Create empty file
            with open(file_path, "w", encoding="utf-8") as f:
                if file.endswith(".log"):
                    pass  # keep log files empty
                elif file.endswith(".py"):
                    f.write(f"# {file}\n")
                elif file.endswith(".ipynb"):
                    f.write("")  # blank notebook placeholder
                else:
                    f.write("")

    print("✅ Project structure created successfully.")


if __name__ == "__main__":
    create_structure()