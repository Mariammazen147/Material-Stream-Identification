import os
import matplotlib.pyplot as plt
from config import RAW_DIR   

# Dataset classes
CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

print("Exploring original dataset...\n")
print("RAW DIR =", RAW_DIR)

counts = {}



for cls in CLASSES:
    class_path = os.path.join(RAW_DIR, cls)

    if not os.path.exists(class_path):
        print(f"[ERROR] Folder does not exist: {class_path}")
        counts[cls] = 0
        continue

    imgs = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    counts[cls] = len(imgs)
    print(f"{cls:10} (ID {CLASS_TO_ID[cls]}): {len(imgs)} images")

total = sum(counts.values())
print(f"\nTotal original images: {total}\n")



output_dir = os.path.join("..", "report", "figures")
os.makedirs(output_dir, exist_ok=True)




plt.figure(figsize=(10, 6))

plt.plot(
    CLASSES,
    [counts[c] for c in CLASSES],
    marker='o',
    linewidth=2
)

plt.title("Original Dataset - Class Distribution (6 classes)")
plt.ylabel("Number of Images")
plt.xlabel("Class (ID 0–5) -> Unknown handled separately")


for x, y in zip(CLASSES, [counts[c] for c in CLASSES]):
    plt.text(x, y + 5, str(y), ha='center', fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.5)

save_path = os.path.join(output_dir, "class_distribution_original_line.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\nSaved plot to: {save_path}")
