import os
import sys
import pickle
import numpy as np
from dataclasses import dataclass
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from src.utils.exceptions import CustomException
from src.utils.logger import logging
from src.constants import (
    IMG_SIZE, DENSE_UNITS, DROPOUT_RATE,
    LEARNING_RATE_HEAD, LEARNING_RATE_FINE,
    EPOCHS_HEAD, EPOCHS_FINE,
    MODEL_PATH, CLASS_NAMES_PATH, ARTIFACTS_DIR,
)


@dataclass
class ModelTrainerConfig:
    img_size: tuple = IMG_SIZE
    model_path: str = MODEL_PATH
    class_names_path: str = CLASS_NAMES_PATH


class ModelTrainer:
    """
    Builds, trains, and saves the EfficientNetV2B0 transfer‑learning model.

    Training is split into two phases:
        1. Head‑only training (backbone frozen) with class weights.
        2. Fine‑tuning of the top 20% of backbone layers.
    """
    def __init__(self, config: ModelTrainerConfig = None):
        self.config = config or ModelTrainerConfig()
        self.model = None
        self.class_names = None
        self.augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomContrast(0.1),
            ],
            name="augmentation",
        )

    def _build_model(self, trainable_base: bool = False, num_classes: int = 7):
        """Create the transfer learning model."""
        base = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=(self.config.img_size[0], self.config.img_size[1], 3),
        )
        base.trainable = trainable_base

        inputs = keras.Input(shape=(self.config.img_size[0], self.config.img_size[1], 3))
        x = self.augmentation(inputs)
        x = keras.applications.efficientnet_v2.preprocess_input(x)
        x = base(x, training=False if not trainable_base else None)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(DENSE_UNITS, activation="relu")(x)
        x = layers.Dropout(DROPOUT_RATE)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        return keras.Model(inputs, outputs, name="soilnet_efficientnetv2b0")

    def _compute_class_weights(self, train_ds):
        """Compute balanced class weights from the training dataset labels."""
        all_labels = np.concatenate([y.numpy() for _, y in train_ds])
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(all_labels),
            y=all_labels,
        )
        return dict(enumerate(weights))

    def train(self, train_ds, val_ds, class_names):
        """Run the complete two‑phase training pipeline."""
        self.class_names = class_names
        num_classes = len(class_names)

        try:
            # ---------- Phase 1: head training ----------
            logging.info("Phase 1: Training classifier head (backbone frozen)...")
            self.model = self._build_model(trainable_base=False, num_classes=num_classes)
            self.model.compile(
                optimizer=keras.optimizers.Adam(LEARNING_RATE_HEAD),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            class_weights = self._compute_class_weights(train_ds)
            logging.info(f"Class weights: {class_weights}")

            history_head = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS_HEAD,
                class_weight=class_weights,
                verbose=1,
            )
            logging.info(f"Phase 1 complete. Final val_acc: {history_head.history['val_accuracy'][-1]:.4f}")

            # ---------- Phase 2: fine‑tuning ----------
            logging.info("Phase 2: Fine‑tuning top 20% of backbone...")
            base = self.model.get_layer("efficientnetv2-b0")
            base.trainable = True
            # Freeze the first 80% of layers
            for layer in base.layers[: int(len(base.layers) * 0.8)]:
                layer.trainable = False

            self.model.compile(
                optimizer=keras.optimizers.Adam(LEARNING_RATE_FINE),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            history_fine = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS_FINE,
                class_weight=class_weights,     # keep using class weights
                verbose=1,
            )
            logging.info(f"Phase 2 complete. Final val_acc: {history_fine.history['val_accuracy'][-1]:.4f}")

            return history_head, history_fine
        except Exception as e:
            raise CustomException(e, sys)

    def save_model(self):
        """Save the trained model and class names to disk."""
        try:
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            self.model.save(self.config.model_path)
            logging.info(f"Model saved to {self.config.model_path}")

            with open(self.config.class_names_path, "wb") as f:
                pickle.dump(self.class_names, f)
            logging.info(f"Class names saved to {self.config.class_names_path}")
        except Exception as e:
            raise CustomException(e, sys)