import numpy as np
import matplotlib.pyplot as plt
from config import FEATURES_FILE, LABELS_FILE
from feature_extraction import CLASSES


X = np.load(FEATURES_FILE)  
y = np.load(LABELS_FILE)

variances = np.var(X, axis=0)
top2_idx = np.argsort(variances)[-2:]       

f1, f2 = top2_idx
print(f"Selected feature indices: {f1} and {f2}")

plt.figure(figsize=(10,6))

for i, cls in enumerate(CLASSES):
    plt.scatter(
        X[y == i, f1],
        X[y == i, f2],
        s=8,
        label=cls
    )

plt.title("Lab-Style Feature Visualization (Feature 0 vs Feature 1)")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.savefig("lab_style_plot.png")
plt.show()
