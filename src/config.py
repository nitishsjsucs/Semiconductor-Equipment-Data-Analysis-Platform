"""
Configuration settings for the ASML Equipment Data Analysis project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "secom"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Data files
SECOM_DATA_FILE = DATA_DIR / "secom.data"
SECOM_LABELS_FILE = DATA_DIR / "secom_labels.data"

# Random seed for reproducibility
RANDOM_STATE = 42

# Data processing settings
MISSING_VALUE_THRESHOLD = 0.5  # Remove features with >50% missing values
CORRELATION_THRESHOLD = 0.95   # Remove highly correlated features
VARIANCE_THRESHOLD = 0.01      # Remove low variance features

# Model settings
TEST_SIZE = 0.2
CV_FOLDS = 10
N_TOP_FEATURES = 50  # Number of top features to select

# Class labels
PASS_LABEL = -1
FAIL_LABEL = 1
