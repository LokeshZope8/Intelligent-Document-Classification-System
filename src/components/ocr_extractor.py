import os
import pandas as pd
import pytesseract
from PIL import Image

from src.utils.common import read_yaml_file
from src.logger import logging


class OCRExtractor:

    def __init__(self, config_path="configs/config.yaml"):

        config = read_yaml_file(config_path)

        self.tesseract_path = config["ocr"]["tesseract_path"]
        self.output_dir = config["ocr"]["extracted_text_dir"]

        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

    def extract_text(self, dataset_path):

        logging.info("Starting OCR extraction")

        df = pd.read_csv(dataset_path)

        # 🔎 Debugging
        print("Columns in dataset:", df.columns.tolist())
        print(df.head())

        texts = []

        for img_path in df["filepath"]:

            try:
                image = Image.open(img_path)

                text = pytesseract.image_to_string(image)

                texts.append(text)

            except Exception as e:

                logging.warning(f"OCR failed for {img_path}")

                texts.append("")