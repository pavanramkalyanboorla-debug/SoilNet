import os

# ------------------------------------------------------------
# Dataset Configuration
# ------------------------------------------------------------
DATA_DIR = os.path.join("data", "soil_dataset")

# ------------------------------------------------------------
# Image / Model Configuration
# ------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 15
EPOCHS_FINE = 3
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 1e-5
DROPOUT_RATE = 0.4
DENSE_UNITS = 256

# ------------------------------------------------------------
# Class names (must match folder names in the dataset)
# ------------------------------------------------------------
CLASS_NAMES = [
    "Alluvial_Soil",
    "Arid_Soil",
    "Black_Soil",
    "Laterite_Soil",
    "Mountain_Soil",
    "Red_Soil",
    "Yellow_Soil",
]

# Clean display names (for UI / confusion matrix)
CLASS_DISPLAY = {
    "Alluvial_Soil": "Alluvial",
    "Arid_Soil": "Arid",
    "Black_Soil": "Black",
    "Laterite_Soil": "Laterite",
    "Mountain_Soil": "Mountain",
    "Red_Soil": "Red",
    "Yellow_Soil": "Yellow",
}

# ------------------------------------------------------------
# Artifacts Paths
# ------------------------------------------------------------
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.keras")
CLASS_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "class_names.pkl")
EVALUATION_DIR = os.path.join(ARTIFACTS_DIR, "evaluation")
MODEL_WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "model_weights.weights.h5")