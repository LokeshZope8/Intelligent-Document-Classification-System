import os
import sys
import pandas as pd

from datasets import load_dataset

from src.exception import CustomException
from src.logger import logging
from src.utils.common import read_yaml_file


class DataIngestion:

    def __init__(self, config_path="configs/config.yaml"):

        try:
            config = read_yaml_file(config_path)

            self.dataset_name = config["data_ingestion"]["dataset_name"]
            self.raw_data_dir = config["data_ingestion"]["raw_data_dir"]

        except Exception as e:
            raise CustomException(e, sys)

    def download_dataset(self):

        try:

            logging.info("Starting dataset download")

            dataset = load_dataset(self.dataset_name)

            df = dataset["train"].to_pandas()

            os.makedirs(self.raw_data_dir, exist_ok=True)

            output_path = os.path.join(self.raw_data_dir, "invoice_dataset.csv")

            df.to_csv(output_path, index=False)

            logging.info("Dataset successfully downloaded")

            return output_path

        except Exception as e:
            raise CustomException(e, sys)

