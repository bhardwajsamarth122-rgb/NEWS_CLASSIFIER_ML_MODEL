from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# Why these imports?

# LabelEncoder → actual encoding logic

# joblib → save/load encoder

# Path → OS-safe paths (industry standard)

class CategoryEncoder:
    def __init__(self):
        self.encoder = LabelEncoder()
        
        # ⚠️ IMPORTANT RULE
        # 👉 fit() is called ONLY ON TRAIN DATA
        
    def fit(self,y):
        self.encoder.fit(y)
        
    def transform(self,y):
        return self.encoder.transform(y)
        
    def save(self, path : Path):
        joblib.dump(self.encoder, path)
        
    def load(self, path: Path):
        self.encoder = joblib.load(path)
        # 🧠 What this does:

        # Converts labels → numbers

        # Uses existing mapping

        # Does NOT create new classes