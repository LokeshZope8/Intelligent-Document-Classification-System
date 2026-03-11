import joblib
import pytesseract

from PIL import Image

from src.utils.common import read_yaml_file


class PredictionPipeline:

    def __init__(self, config_path="configs/config.yaml"):

        config = read_yaml_file(config_path)

        self.tesseract_path = config["ocr"]["tesseract_path"]

        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

        # Load artifacts
        self.model = joblib.load("artifacts/models/best_model.pkl")

        self.vectorizer = joblib.load("data/processed/tfidf_vectorizer.pkl")

    def predict(self, image_path):

        # OCR extraction
        image = Image.open(image_path)

        text = pytesseract.image_to_string(image)

        # Vectorization
        text_vector = self.vectorizer.transform([text])

        prediction = self.model.predict(text_vector)[0]

        # Confidence score (approx)
        if hasattr(self.model, "predict_proba"):
            confidence = self.model.predict_proba(text_vector).max()
        else:
            confidence = 0.0

        return prediction, confidence