"""Locate the PlantVillage dataset folders without importing the runtime stack.

Kept separate from evaluate_model.py so the training scripts can use it inside an
environment that has TensorFlow but not onnxruntime.

The zip "kaggle dataset.zip" can be extracted either at the project root (which
produces PlantVillage/PlantVillage/<class>/) or into plant-village/ (which
produces plant-village/PlantVillage/<class>/).
"""

import os

# First folder that exists on disk wins, so the scripts work without flags
DATASET_CANDIDATES = (
    os.path.join("plant-village", "PlantVillage"),
    os.path.join("PlantVillage", "PlantVillage"),
    "PlantVillage",
)

# Matches DATASET_DIR semantics in the original train_model.py
DEFAULT_DATA_DIR = DATASET_CANDIDATES[0]


def resolve_data_dir(data_dir=None):
    """Return data_dir when it exists, otherwise the first candidate that does."""
    if data_dir and os.path.isdir(data_dir):
        return data_dir
    for candidate in DATASET_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    return data_dir or DEFAULT_DATA_DIR


def class_names_in(data_dir):
    """Sorted list of class folders, which is the label order used everywhere."""
    if not os.path.isdir(data_dir):
        return []
    return sorted(name for name in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, name)))
