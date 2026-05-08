# src/pipeline/predict_pipeline.py
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dataclasses import dataclass
from src.utils.exceptions import CustomException
from src.utils.logger import logging
from src.constants import (
    IMG_SIZE,            # (224, 224)
    DENSE_UNITS,
    DROPOUT_RATE,
    CLASS_NAMES_PATH,
    ARTIFACTS_DIR,
    CLASS_DISPLAY,
)
import os

MODEL_WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "model_weights.weights.h5")

# Add the channel dimension so input_shape is always a 3‑tuple
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)


@dataclass
class PredictPipelineConfig:
    input_shape: tuple = INPUT_SHAPE       # (224, 224, 3)
    dense_units: int = DENSE_UNITS
    dropout_rate: float = DROPOUT_RATE
    weights_path: str = MODEL_WEIGHTS_PATH
    class_names_path: str = CLASS_NAMES_PATH


class PredictPipeline:
    """Builds the inference model from scratch, loads trained weights, and serves predictions."""

    def __init__(self, config: PredictPipelineConfig = None):
        self.config = config or PredictPipelineConfig()
        self.model = None
        self.class_names = None
        self.display_names = None
        self._build_and_load()

    def _build_and_load(self):
        try:
            # 1. Load class names
            with open(self.config.class_names_path, "rb") as f:
                self.class_names = pickle.load(f)
            self.display_names = [CLASS_DISPLAY.get(c, c) for c in self.class_names]
            num_classes = len(self.class_names)

            # 2. Build inference architecture (NO augmentation)
            #    input_shape is a 3‑tuple (height, width, channels)
            inputs = keras.Input(shape=self.config.input_shape, name="input_layer")

            # EfficientNetV2‑specific preprocessing
            x = keras.applications.efficientnet_v2.preprocess_input(inputs)

            # Load the base model without weights (we'll load from H5)
            base = keras.applications.EfficientNetV2B0(
                include_top=False,
                weights=None,
                input_shape=self.config.input_shape,      # now (224, 224, 3)
            )
            x = base(x, training=False)

            x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
            x = layers.Dense(self.config.dense_units, activation="relu", name="classifier_dense")(x)
            x = layers.Dropout(self.config.dropout_rate, name="classifier_dropout")(x)
            outputs = layers.Dense(num_classes, activation="softmax", name="classifier_output")(x)
            self.model = keras.Model(inputs, outputs, name="soilnet_inference")

            # 3. Load the trained weights
            self.model.load_weights(self.config.weights_path)
            logging.info("Model weights loaded successfully (HDF5 format).")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, image: np.ndarray):
        """Run inference on a single image.

        Args:
            image: numpy array of shape (H, W, 3), values in [0, 255].

        Returns:
            dict with keys 'class', 'class_raw', 'confidence', 'all_probs'.
        """
        try:
            # Resize to the expected spatial dimensions (height, width)
            img = tf.image.resize(image, self.config.input_shape[:2])
            img = tf.cast(img, tf.float32)
            img = tf.expand_dims(img, axis=0)

            preds = self.model.predict(img, verbose=0)[0]
            idx = np.argmax(preds)

            return {
                "class": self.display_names[idx],
                "class_raw": self.class_names[idx],
                "confidence": float(preds[idx]),
                "all_probs": {
                    self.display_names[i]: float(preds[i])
                    for i in range(len(self.class_names))
                },
            }
        except Exception as e:
            raise CustomException(e, sys)