import numpy as np
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from config import FEATURES_FILE, LABELS_FILE, PROJECT_ROOT

print("Loading features (this may take a moment)...")
X = np.load(FEATURES_FILE)
y = np.load(LABELS_FILE)

print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
print("Classes:", np.unique(y))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Try feature selection
print("\nTrying feature selection...")
selector = SelectKBest(f_classif, k=min(500, X.shape[1]))
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)

# Try both scalers
best_overall = 0
best_config = {}

for scaler_type in ['standard', 'minmax']:
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)

    for metric in ['euclidean', 'manhattan']:
        for k in [3, 5, 7, 9, 11, 15, 20, 25, 30]:
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric=metric)
            knn.fit(X_train_scaled, y_train)
            score = knn.score(X_val_scaled, y_val)

            if score > best_overall:
                best_overall = score
                best_config = {
                    'k': k,
                    'scaler': scaler_type,
                    'metric': metric,
                    'score': score
                }

print(f"\nBest config: {best_config}")
print(f"ACCURACY: {best_overall:.4f}")

# Train final model with best config
if best_config['scaler'] == 'standard':
    final_scaler = StandardScaler()
else:
    final_scaler = MinMaxScaler()

X_train_scaled = final_scaler.fit_transform(X_train_selected)
X_val_scaled = final_scaler.transform(X_val_selected)

final_knn = KNeighborsClassifier(
    n_neighbors=best_config['k'],
    weights='distance',
    metric=best_config['metric']
)
final_knn.fit(X_train_scaled, y_train)

# Rejection
distances, _ = final_knn.kneighbors(X_val_scaled)
avg_distances = distances.mean(axis=1)
threshold = np.percentile(avg_distances, 80)

y_pred = final_knn.predict(X_val_scaled)
y_rejected = np.where(avg_distances > threshold, 6, y_pred)
print(f"Accuracy with rejection: {accuracy_score(y_val, y_rejected):.4f}")

# Retrain on full data
X_selected = selector.fit_transform(X, y)
X_scaled = final_scaler.fit_transform(X_selected)
final_knn.fit(X_scaled, y)

joblib.dump(final_knn, "knn_model.joblib")
joblib.dump(final_scaler, "knn_scaler.joblib")
joblib.dump(selector, "knn_selector.joblib")
np.save("knn_threshold.npy", np.array([threshold]))

print("\nAll files saved:")
print("   knn_model.joblib")
print("   knn_scaler.joblib")
print("   knn_selector.joblib")
print("   knn_threshold.npy")
print("Accuracy was:", best_overall)