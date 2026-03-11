import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.ocr_extractor import OCRExtractor
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.logger import logging



class TrainingPipeline:

    def start(self):

        logging.info("Training pipeline started")

        #Data Ingestion
        ingestion = DataIngestion()

        dataset_path = ingestion.download_dataset()
    
        #Data Validation
        validation = DataValidation()

        validation.validate(dataset_path)

        #if data is already in text format ocr will be skipped
        df = pd.read_csv(dataset_path)

        if "filepath" in df.columns or "image" in df.columns:

            logging.info("Image dataset detected. Running OCR stage.")

            ocr = OCRExtractor()

            dataset_path = ocr.extract_text(dataset_path)

        else:

            logging.info("Text dataset detected. Skipping OCR stage.")

        final_dataset_path = dataset_path

        logging.info(f"Final dataset used for transformation: {final_dataset_path}")

        #Data Transformation
        transformation = DataTransformation()
        transformation.transform(final_dataset_path)
        
        logging.info("Feature engineering stage completed")

        #Model Trainer
        trainer = ModelTrainer()
        trainer.train()

        logging.info("Model training stage completed")

        #Model Evaluation
        evaluation = ModelEvaluation()
        evaluation.evaluate()

        logging.info("Model evaluation stage completed")


if __name__ == "__main__":

    pipeline = TrainingPipeline()

    pipeline.start()