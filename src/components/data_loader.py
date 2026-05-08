import os
import sys
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils.exceptions import CustomException
from src.utils.logger import logging
from src.constants import DATA_DIR, IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED


@dataclass
class DataLoaderConfig:
    data_dir: str = DATA_DIR
    img_size: tuple = IMG_SIZE
    batch_size: int = BATCH_SIZE
    validation_split: float = VALIDATION_SPLIT
    seed: int = RANDOM_SEED


class DataLoader:
    """
    Loads the soil image dataset and creates a stratified train/validation split.

    The stratified split is performed on the raw file paths — this avoids the
    class‑missing bug that occurs when using image_dataset_from_directory with
    validation_split on small, imbalanced datasets.
    """
    def __init__(self, config: DataLoaderConfig = None):
        self.config = config or DataLoaderConfig()
        self.class_names = None
        self.label_to_idx = None

    def _build_file_dataframe(self):
        """Walk the data directory and return a DataFrame of file paths and labels."""
        image_paths, labels = [], []
        for class_name in sorted(os.listdir(self.config.data_dir)):
            class_dir = os.path.join(self.config.data_dir, class_name)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    image_paths.append(os.path.join(class_dir, fname))
                    labels.append(class_name)
        df = pd.DataFrame({"filepath": image_paths, "label": labels})
        self.class_names = sorted(df["label"].unique())
        self.label_to_idx = {name: i for i, name in enumerate(self.class_names)}
        logging.info(f"Scanned dataset: {len(df)} images, {len(self.class_names)} classes.")
        return df

    def _decode_and_preprocess(self, path, label_idx):
        """Read an image file, decode it, resize, and cast to float32."""
        image = tf.io.read_file(path)
        # decode_image automatically detects JPEG/PNG
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, self.config.img_size)
        image = tf.cast(image, tf.float32)
        return image, label_idx

    def _make_dataset(self, df, shuffle=False):
        """Build a tf.data.Dataset from a DataFrame, optionally shuffling."""
        ds = tf.data.Dataset.from_tensor_slices(
            (df["filepath"].values, df["label_idx"].values)
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df), seed=self.config.seed)
        ds = ds.map(self._decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.config.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def get_train_val_split(self):
        """Returns (train_ds, val_ds) with a reproducible stratified split."""
        try:
            df = self._build_file_dataframe()
            df["label_idx"] = df["label"].map(self.label_to_idx)

            # Stratified split preserves class proportions in both sets
            train_df, val_df = train_test_split(
                df,
                test_size=self.config.validation_split,
                stratify=df["label"],
                random_state=self.config.seed,
            )

            train_ds = self._make_dataset(train_df, shuffle=True)
            val_ds = self._make_dataset(val_df, shuffle=False)   # deterministic for evaluation

            logging.info(f"Split: {len(train_df)} train, {len(val_df)} validation samples.")
            return train_ds, val_ds
        except Exception as e:
            raise CustomException(e, sys)