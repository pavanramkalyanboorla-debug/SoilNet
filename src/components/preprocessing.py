import sys
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import layers
from src.utils.exceptions import CustomException
from src.utils.logger import logging
from src.constants import IMG_SIZE, BATCH_SIZE


@dataclass
class PreprocessingConfig:
    img_size: tuple = IMG_SIZE
    batch_size: int = BATCH_SIZE
    cache: bool = True


class DataPreprocessor:
    """
    Applies caching, prefetching, and data augmentation.

    Caching is safe here because the dataset fits in RAM (≈200 MB for ~1k images).
    """
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()

    def optimize(self, train_ds, val_ds):
        """Apply cache + prefetch for faster CPU training."""
        try:
            if self.config.cache:
                logging.info("Caching datasets in memory (safe for ~1k images)...")
                train_ds = train_ds.cache()
                val_ds = val_ds.cache()
            train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            logging.info("Dataset optimisation complete.")
            return train_ds, val_ds
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def get_augmentation_layer():
        """Mild augmentation suitable for soil textures."""
        return tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomContrast(0.1),
            ],
            name="augmentation",
        )