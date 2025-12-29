import pandas as ps
from pathlib import Path

from src.features.label_encoder import CategoryEncoder
from src.features.vectorizer import TextVectorizer
from src.models.train import NewsClassifier
from src.utils.exceptions import NewsClassifierException
from src.utils.logger import get_logger

logger = get_logger()

def Run_Training():
    try:
        logger.info("Training Pipeline is been Started.")
        ## now taking Train Nd validation data Path

        train_path = Path("Data/Processed/train.csv")
        Validation_Path = Path("Data/Processed/Validation.csv")
    
        train_df = ps.read_csv(train_path)
        validation_df = ps.read_csv(Validation_Path)
    
        ## encoding category feature of the 
        encoder = CategoryEncoder()
        encoder.fit(train_df['category'])
        y_train = encoder.transform(train_df['category'])
        y_val = encoder.transform(validation_df['category'])
        Path("Artifacts").mkdir(parents=True, exist_ok=True)
        encoder.save(Path("Artifacts/label_encoder.pkl"))

        ## now Vectorizing
        vectorizer = TextVectorizer()
        X_train = vectorizer.fit_transform(train_df['text'])
        X_val = vectorizer.transform(validation_df['text'])
        vectorizer.save(Path("Artifacts/vectorizer.pkl"))
    
        ## Model Training
        model = NewsClassifier()
        model.train(X_train,y_train)
    
        ## Validation 
        metrics = model.evaluate(X_val,y_val)
    
        ## Save Path 
        Path("Artifacts/models").mkdir(parents=True, exist_ok=True)
        model.save("Artifacts/models/news_classifier.pkl")
    
        logger.info("Training Pipeline Executed Successfully")
        return metrics
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise NewsClassifierException("exception : ") from e

if __name__ == "__main__":
    Run_Training()
    