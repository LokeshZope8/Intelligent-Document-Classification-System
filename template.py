import os
from pathlib import Path

list_of_files = [

    "requirements.txt",
    "README.md",

    # Config
    "configs/config.yaml",

    # Artifacts
    "artifacts/.gitkeep",

    # Logs
    "logs/.gitkeep",

    # Data folders
    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",

    # Notebooks
    "notebooks/.gitkeep",

    # Components
    "src/components/data_ingestion.py",
    "src/components/data_validation.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",

    # Pipeline
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",

    # Entity
    "src/entity/config_entity.py",
    "src/entity/artifact_entity.py",

    # Utils
    "src/utils/common.py",

    # Core
    "src/logger.py",
    "src/exception.py",

    # API
    "api/app.py",
]

for filepath in list_of_files:

    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass