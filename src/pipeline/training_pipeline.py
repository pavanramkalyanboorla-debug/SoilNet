import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tensorflow as tf
from src.components.data_loader import DataLoader, DataLoaderConfig
from src.components.preprocessing import DataPreprocessor, PreprocessingConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.model_evaluator import ModelEvaluator
from src.utils.exceptions import CustomException
from src.utils.logger import logging

tf.random.set_seed(42)

if __name__ == "__main__":
    try:
        # 1. Data Loading
        loader = DataLoader()
        train_ds, val_ds = loader.get_train_val_split()
        class_names = loader.class_names
        logging.info(f"Data loaded. Classes: {class_names}")

        # 2. Preprocessing (cache, prefetch)
        preprocessor = DataPreprocessor()
        train_ds, val_ds = preprocessor.optimize(train_ds, val_ds)
        logging.info("Data preprocessing complete.")

        # 3. Model Training
        trainer = ModelTrainer()
        trainer.train(train_ds, val_ds, class_names)
        trainer.save_model()
        logging.info("Model training and saving complete.")

        # 4. Evaluation
        evaluator = ModelEvaluator()
        report, cm = evaluator.evaluate(val_ds)
        evaluator.save_artifacts(report, cm)
        logging.info("Evaluation complete.")

    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        raise CustomException(e, sys)