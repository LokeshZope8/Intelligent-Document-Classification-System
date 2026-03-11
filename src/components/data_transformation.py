import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

from src.utils.common import read_yaml_file
from src.logger import logging


class DataTransformation:

    def __init__(self, config_path="configs/config.yaml"):

        config = read_yaml_file(config_path)

        self.max_features = config["feature_engineering"]["max_features"]
        self.ngram_range = tuple(config["feature_engineering"]["ngram_range"])
        self.processed_dir = config["data_transformation"]["processed_data_dir"]

    def transform(self, dataset_path):

        logging.info("Starting feature engineering")

        df = pd.read_csv(dataset_path)

        X = df["text"].astype(str)
        y = df["label"]

        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english"
        )

        X_vectorized = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized,
            y,
            test_size=0.2,
            random_state=42
        )

        os.makedirs(self.processed_dir, exist_ok=True)

        joblib.dump(vectorizer, os.path.join(self.processed_dir, "tfidf_vectorizer.pkl"))
        joblib.dump(X_train, os.path.join(self.processed_dir, "X_train.pkl"))
        joblib.dump(X_test, os.path.join(self.processed_dir, "X_test.pkl"))
        joblib.dump(y_train, os.path.join(self.processed_dir, "y_train.pkl"))
        joblib.dump(y_test, os.path.join(self.processed_dir, "y_test.pkl"))

        logging.info("Feature engineering completed")

        return os.path.join(self.processed_dir, "X_train.pkl")