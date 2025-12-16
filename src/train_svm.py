# src/train_svm.py - QUICK RECOVERY VERSION FOR 0.76 ACCURACY
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

from config import FEATURES_FILE, LABELS_FILE, PROJECT_ROOT

print("Loading features (this may take a moment)...")
X = np.load(FEATURES_FILE)
y = np.load(LABELS_FILE)

print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
print("Classes:", np.unique(y))

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale (this is the "scaler" you need)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Use known good params to avoid long grid search (fast & safe)
print("Training SVM with good parameters (C=50, gamma='scale')...")
svm = SVC(C=50, gamma='scale', kernel='rbf', class_weight='balanced', probability=True)
svm.fit(X_train_scaled, y_train)

# Validate
y_pred = svm.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)
print(f"\nACCURACY: {acc:.4f}")

# Rejection for Unknown
decisions = svm.decision_function(X_val_scaled)
confidences = np.max(np.abs(decisions), axis=1)
threshold = 0.8
y_rejected = np.where(confidences < threshold, 6, y_pred)
print(f"Accuracy with rejection: {accuracy_score(y_val, y_rejected):.4f}")

# Save plot
plt.hist(confidences[y_val < 6], bins=50, alpha=0.7, label='Known classes')
plt.axvline(threshold, color='red', linestyle='--')
plt.title('Confidence for Unknown Rejection')
plt.legend()
plt.savefig(os.path.join(PROJECT_ROOT, "report", "confidence.png"))
plt.close()

print("Retraining on full data...")
X_scaled = scaler.fit_transform(X)
final_svm = SVC(C=50, gamma='scale', kernel='rbf', class_weight='balanced', probability=True)
final_svm.fit(X_scaled, y)

joblib.dump(final_svm, "svm_model.joblib")        
joblib.dump(scaler, "scaler.joblib")              
np.save("rejection_threshold.npy", np.array([threshold]))

print("\nAll files saved:")
print("   svm_model.joblib")
print("   scaler.joblib")
print("   rejection_threshold.npy")
print("   confidence plot")
print("Accuracy was:", acc)