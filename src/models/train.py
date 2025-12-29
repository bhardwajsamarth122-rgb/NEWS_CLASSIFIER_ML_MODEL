from src.utils.logger import get_logger
from src.utils.exceptions import NewsClassifierException

logger = get_logger()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

class NewsClassifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        
    def train(self, X,y):
        try:
            logger.info("Training has started")
            self.model.fit(X,y)
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Error During Training is {e}")
            raise NewsClassifierException("Training Failed") from e

    def evaluate(self,X,y):
        y_preds=self.model.predict(X)
        return classification_report(y,y_preds,output_dict=True)
        
    def save(self, path):
        joblib.dump(self.model,path)
        
    def load(self,path):
        self.model = joblib.load(path)