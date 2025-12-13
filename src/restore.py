import os
import shutil
from config import AUGMENTED_DIR, PROJECT_ROOT, FEATURES_FILE, LABELS_FILE

# Paths for plots
PCA_PLOT = os.path.join(PROJECT_ROOT, "pca_plot.png")
TSNE_PLOT = os.path.join(PROJECT_ROOT, "tsne_plot.png")


def delete_folder_contents(folder):
    """Delete all files inside a folder but keep the folder."""
    if not os.path.exists(folder):
        print(f"[SKIP] Folder does not exist: {folder}")
        return

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"[ERROR] Could not delete {file_path}: {e}")

    print(f"[OK] Cleared folder: {folder}")


def delete_file(path):
    """Delete a file if it exists."""
    if os.path.exists(path):
        os.remove(path)
        print(f"[OK] Deleted file: {path}")
    else:
        print(f"[SKIP] File not found: {path}")


if __name__ == "__main__":
    print("=== RESTORING PROJECT ===")

    # 1. Clear augmented dataset
    delete_folder_contents(AUGMENTED_DIR)

    # 2. Delete visualization results
    delete_file(PCA_PLOT)
    delete_file(TSNE_PLOT)

    # 3. Delete feature matrix files
    delete_file(FEATURES_FILE)
    delete_file(LABELS_FILE)

    print("=== RESTORE COMPLETE ===")
