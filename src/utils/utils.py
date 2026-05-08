import pickle
import json
import os
from src.utils.exceptions import CustomException
from src.utils.logger import logging
import sys


def save_object(file_path, obj):
    """Save any Python object to disk."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a pickle object from disk."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def save_json(file_path, obj):
    """Save a JSON-serializable object."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(obj, f, indent=2)
        logging.info(f"JSON saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)