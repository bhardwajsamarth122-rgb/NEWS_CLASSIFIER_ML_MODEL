import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger
from src.utils.exceptions import NewsClassifierException
from src.config.config_loader import load_config
import sys

logging = get_logger()

class DataIngestion :
    def __init__(self):
        self.config = load_config()["data_ingestion"]
        
    def initiate_data_ingestion(self):
        try:
            print("check")
            logging.info("Starting Data Ingestion")
            raw_path = self.config["raw_data_path"]
            processed_dir = self.config["processed_dir"]
            
            os.makedirs(processed_dir,exist_ok=True)
            
            df = pd.read_json(raw_path,lines=True)

            ## keep only required columns
            df = df[["headline", "short_description", "category"]]
            df["text"] = df["headline"] + " " + df["short_description"]
            df = df[["text", "category"]]
            
            train_df, temp_df = train_test_split(
                df,
                test_size=self.config["test_size"] + self.config["validation_size"],
                random_state=self.config["random_state"],
                stratify=df["category"]
            )
            
            validation_df, test_df = train_test_split(
                temp_df,
                test_size=self.config["test_size"] / (self.config["test_size"] + self.config["validation_size"]),
                random_state=self.config["random_state"],
                stratify=temp_df["category"]
            )
            
            train_df.to_csv(self.config["train_path"], index=False)
            validation_df.to_csv(self.config["validation_path"], index=False)
            test_df.to_csv(self.config["test_path"], index=False)
            
            logging.info("Data ingestion completed successfully")

            return (
                self.config["train_path"],
                self.config["validation_path"],
                self.config["test_path"]
            )


        except Exception as e:
            logging.error("Error in data ingestion")
            raise NewsClassifierException(e,sys)