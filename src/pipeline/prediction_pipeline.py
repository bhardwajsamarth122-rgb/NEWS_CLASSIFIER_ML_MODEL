from pathlib import Path

from src.features.label_encoder import CategoryEncoder
from src.features.vectorizer import TextVectorizer
from src.models.train import NewsClassifier
from src.utils.exceptions import NewsClassifierException
from src.utils.logger import get_logger

logger = get_logger()

class PredictionPipeline:
    
    def __init__(self):
        try:
            
            logger.info("prediction Pipeline Initializing")
            
            # defining label Encoder
            self.encoder =CategoryEncoder()
            self.encoder.load(Path("Artifacts/label_encoder.pkl"))
            
            # defining Vectorizer
            self.vectorizer = TextVectorizer()
            self.vectorizer.load(Path("Artifacts/vectorizer.pkl"))
            
            
            
            # defining model
            self.model = NewsClassifier()
            self.model.load(Path("Artifacts/models/news_classifier.pkl"))
            
            logger.info("Prediction Pipeline Initialized Succesfully")
    
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise NewsClassifierException(e)
        
    def predict(self, text: str):
        try:

            logger.info("Starting Prediction")

            text_vector = self.vectorizer.transform([text])
            pred_label = self.model.model.predict(text_vector)[0]
            proba = self.model.model.predict_proba(text_vector)[0]
            labels = self.encoder.encoder.classes_
            category = self.encoder.encoder.inverse_transform([pred_label])[0]
            
            prob_dict = {
                            label: round(float(prob) * 100, 2)
                            for label, prob in zip(labels, proba)
                        }
            confidence = round(float(max(proba)) * 100, 2)

            logger.info("Prediction Completed")
            
            return {
            "label": category,
            "confidence": confidence,
            "probabilities": prob_dict
            }

        except Exception as e:
            logger.error(f"Prediction Failed : {e}")
            raise NewsClassifierException(e)