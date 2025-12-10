import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import random

DATASET_PATH = "D:/MaterialClassification/dataset/raw"
CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

print("Exploring original dataset...\n")
counts = {}
for cls in CLASSES:
    path = os.path.join(DATASET_PATH, cls)
    imgs = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    counts[cls] = len(imgs)
    print(f"{cls:10} (ID {CLASS_TO_ID[cls]}): {len(imgs):3} images")

total = sum(counts.values())
print(f"\nTotal original images: {total}\n")

# Save CSV
os.makedirs("../report/figures", exist_ok=True)
plt.figure(figsize=(10,6))

plt.plot(
    CLASSES,
    [counts[c] for c in CLASSES],
    marker='o',
    linewidth=2
)

plt.title("Original Dataset - Class Distribution (6 classes only)")
plt.ylabel("Number of Images")
plt.xlabel("Class (ID 0-5) → Unknown (6) handled by rejection")

for x, y in zip(CLASSES, [counts[c] for c in CLASSES]):
    plt.text(x, y + 5, str(y), ha='center', fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig("../report/figures/class_distribution_original_line.png", dpi=300, bbox_inches='tight')
plt.show()
