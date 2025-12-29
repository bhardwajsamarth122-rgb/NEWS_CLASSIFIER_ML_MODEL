import sys
import os
import yaml
from src.utils.exceptions import NewsClassifierException

def load_config():
    try:
        config_path = os.path.join("src", "config", "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise NewsClassifierException(e, sys)