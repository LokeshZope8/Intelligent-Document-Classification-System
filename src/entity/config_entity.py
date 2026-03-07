from dataclasses import dataclass


@dataclass
class DataIngestionConfig:

    dataset_name: str
    raw_data_dir: str
    processed_data_dir: str


@dataclass
class DataTransformationConfig:

    max_features: int
    ngram_range: tuple


@dataclass
class ModelTrainerConfig:

    models: list


@dataclass
class ModelEvaluationConfig:

    metric: str
    target_score: float


@dataclass
class PredictionConfig:

    confidence_threshold: float