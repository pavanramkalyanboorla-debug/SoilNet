import os
import sys
import json
import time
import numpy as np                
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from src.utils.exceptions import CustomException
from src.utils.logger import logging
from src.constants import (
    MODEL_PATH, CLASS_NAMES_PATH, EVALUATION_DIR,
    CLASS_DISPLAY, IMG_SIZE,
)

class ModelEvaluator:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        import pickle
        with open(CLASS_NAMES_PATH, "rb") as f:
            self.class_names = pickle.load(f)
        self.display_names = [CLASS_DISPLAY.get(c, c) for c in self.class_names]

    def evaluate(self, val_ds):
        """Compute classification report and confusion matrix on the given unshuffled dataset."""
        try:
            y_true = np.concatenate([y.numpy() for _, y in val_ds])
            y_pred_probs = self.model.predict(val_ds, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)

            report = classification_report(
                y_true, y_pred,
                target_names=self.display_names,
                zero_division=0,
                output_dict=True,
            )
            cm = confusion_matrix(y_true, y_pred)
            logging.info(f"Evaluation accuracy: {report['accuracy']:.4f}")
            return report, cm
        except Exception as e:
            raise CustomException(e, sys)

    def benchmark(self):
        """Measure inference latency and model file size."""
        try:
            sample = tf.random.normal((1, IMG_SIZE[0], IMG_SIZE[1], 3))
            start = time.time()
            _ = self.model.predict(sample, verbose=0)
            latency = (time.time() - start) * 1000
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            return latency, size_mb
        except Exception as e:
            raise CustomException(e, sys)

    def save_artifacts(self, report, cm):
        """Save metrics, confusion matrix, and benchmark to disk."""
        try:
            os.makedirs(EVALUATION_DIR, exist_ok=True)

            if report:
                with open(os.path.join(EVALUATION_DIR, "metrics.json"), "w") as f:
                    json.dump(report, f, indent=2)

            if cm is not None:
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.display_names,
                    yticklabels=self.display_names,
                )
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.tight_layout()
                plt.savefig(os.path.join(EVALUATION_DIR, "confusion_matrix.png"), dpi=150)
                plt.close()

            latency, size_mb = self.benchmark()
            with open(os.path.join(EVALUATION_DIR, "benchmark.json"), "w") as f:
                json.dump({"latency_ms": round(latency, 2), "size_mb": round(size_mb, 2)}, f, indent=2)

            logging.info("Evaluation artifacts saved.")
        except Exception as e:
            raise CustomException(e, sys)