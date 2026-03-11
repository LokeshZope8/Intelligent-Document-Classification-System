import os
import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from src.utils.common import read_yaml_file
from src.logger import logging


class ModelTrainer:

    def __init__(self, config_path="configs/config.yaml"):

        config = read_yaml_file(config_path)

        self.model_dir = config["model_trainer"]["model_dir"]

    def train(self):

        logging.info("Starting model training")

        X_train = joblib.load("data/processed/X_train.pkl")
        X_test = joblib.load("data/processed/X_test.pkl")
        y_train = joblib.load("data/processed/y_train.pkl")
        y_test = joblib.load("data/processed/y_test.pkl")

        models = {
            "NaiveBayes": MultinomialNB(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "SVM": LinearSVC()
        }

        best_score = 0
        best_model = None
        best_model_name = ""

        for name, model in models.items():

            logging.info(f"Training model: {name}")

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            score = accuracy_score(y_test, predictions)

            logging.info(f"{name} Accuracy: {score}")

            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name

        os.makedirs(self.model_dir, exist_ok=True)

        model_path = os.path.join(self.model_dir, "best_model.pkl")

        joblib.dump(best_model, model_path)

        logging.info(f"Best Model: {best_model_name}")
        logging.info(f"Best Accuracy: {best_score}")
        logging.info(f"Model saved at: {model_path}")

        return model_path