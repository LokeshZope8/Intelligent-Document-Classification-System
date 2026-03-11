import os
import joblib
import json

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.utils.common import read_yaml_file
from src.logger import logging


class ModelEvaluation:

    def __init__(self, config_path="configs/config.yaml"):

        config = read_yaml_file(config_path)

        self.metrics_dir = config["model_trainer"]["metrics_dir"]

    def evaluate(self):

        logging.info("Starting model evaluation")

        model = joblib.load("artifacts/models/best_model.pkl")

        X_test = joblib.load("data/processed/X_test.pkl")
        y_test = joblib.load("data/processed/y_test.pkl")

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        report = classification_report(y_test, predictions, output_dict=True)

        cm = confusion_matrix(y_test, predictions)

        os.makedirs(self.metrics_dir, exist_ok=True)

        # Save metrics
        metrics_path = os.path.join(self.metrics_dir, "metrics.json")

        with open(metrics_path, "w") as f:
            json.dump({
                "accuracy": accuracy,
                "classification_report": report
            }, f, indent=4)

        # Plot confusion matrix
        plt.figure(figsize=(8,6))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

        plt.title("Confusion Matrix")

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        cm_path = os.path.join(self.metrics_dir, "confusion_matrix.png")

        plt.savefig(cm_path)

        logging.info(f"Evaluation metrics saved at {metrics_path}")

        logging.info(f"Confusion matrix saved at {cm_path}")

        return accuracy