from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.logger import logging


class TrainingPipeline:

    def start(self):

        logging.info("Training pipeline started")

        ingestion = DataIngestion()

        dataset_path = ingestion.download_dataset()

        validation = DataValidation()

        validation.validate(dataset_path)

        logging.info("Data validation stage completed")


if __name__ == "__main__":

    pipeline = TrainingPipeline()

    pipeline.start()