import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
AUGMENTED_DIR = os.path.join(DATASET_DIR, "augmented")

FEATURES_FILE = os.path.join(DATASET_DIR, "features.npy")
LABELS_FILE = os.path.join(DATASET_DIR, "labels.npy")

print("PROJECT ROOT:", PROJECT_ROOT)
print("RAW DIR:", RAW_DIR)
print("AUGMENTED DIR:", AUGMENTED_DIR)
print("FEATURES FILE:", FEATURES_FILE)
print("LABELS FILE:", LABELS_FILE)
