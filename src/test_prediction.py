from src.pipeline.prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline()

prediction, confidence = pipeline.predict(
    r"D:\projects\Intelligent Document Classification System\Intelligent-Document-Classification-System\sample_images\sample_invoice.png"
)

print("Predicted Class:", prediction)
print("Confidence:", confidence)


