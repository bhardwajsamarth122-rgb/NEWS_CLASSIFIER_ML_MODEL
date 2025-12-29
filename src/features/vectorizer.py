from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path

class TextVectorizer:
    def __init__(self, max_features = 20000, ngram_range = (1,2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        
    # Fit + Transform (TRAIN ONLY)
    
    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X)
        
    # Transform is only for the test and validation data
    
    def transform(self, X):
        return self.vectorizer.transform(X)
                
    def save(self, Path : Path):
        joblib.dump(self.vectorizer, Path)
        
    def load(self, path : Path):
        self.vectorizer=joblib.load(path)