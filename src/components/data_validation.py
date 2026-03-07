import pandas as pd

from src.utils.common import read_yaml_file
from src.logger import logging


class DataValidation:

    def __init__(self, config_path="configs/config.yaml"):

        config = read_yaml_file(config_path)

        self.required_columns = config["data_validation"]["required_columns"]

    def validate(self, dataset_path):

        logging.info("Starting data validation")

        df = pd.read_csv(dataset_path)

        # Check required columns
        for col in self.required_columns:
            if col not in df.columns:
                raise Exception(f"Missing required column: {col}")

        # Check empty dataset
        if df.shape[0] == 0:
            raise Exception("Dataset is empty")

        # Check missing values
        missing_values = df.isnull().sum().sum()

        if missing_values > 0:
            logging.warning(f"Dataset contains {missing_values} missing values")

        logging.info("Data validation completed successfully")

        return True