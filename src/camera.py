import cv2
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from feature_extraction import extract_features
from config import FEATURES_FILE, LABELS_FILE

print("Loading validation data...")
X = np.load(FEATURES_FILE)
y = np.load(LABELS_FILE)
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Testing models...")
try:
    svm = joblib.load('svm_model.joblib')
    svm_scaler = joblib.load('scaler.joblib')
    svm_acc = accuracy_score(y_val, svm.predict(svm_scaler.transform(X_val)))
except:
    svm, svm_scaler, svm_acc = None, None, 0

try:
    knn = joblib.load('knn_model.joblib')
    knn_scaler = joblib.load('knn_scaler.joblib')
    knn_acc = accuracy_score(y_val, knn.predict(knn_scaler.transform(X_val)))
except:
    knn, knn_scaler, knn_acc = None, None, 0

if svm_acc >= knn_acc:
    model, scaler = svm, svm_scaler
    model_name = "SVM"
else:
    model, scaler = knn, knn_scaler
    model_name = "k-NN"

print(f"\nUsing {model_name}")

class_names = {0:'GLASS', 1:'PAPER', 2:'CARDBOARD', 3:'PLASTIC', 4:'METAL', 5:'TRASH', 6:'UNKNOWN'}
colors = {0:(255,200,100), 1:(240,240,240), 2:(100,150,200), 3:(100,255,255), 4:(200,200,200), 5:(50,50,200), 6:(0,0,255)}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        features = extract_features(frame).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Get prediction
        pred = model.predict(features_scaled)[0]

        # Get confidence
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_scaled)[0]
            confidence = probs[pred] * 100
        else:
            # For k-NN without probability
            distances, _ = model.kneighbors(features_scaled)
            confidence = max(0, 100 - distances.mean() * 20)

        label = class_names[pred]
        color = colors[pred]

        # Draw semi-transparent box for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (630, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # BIG material name
        cv2.putText(frame, label, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)

        # Confidence percentage
        cv2.putText(frame, f"Confidence: {confidence:.0f}%", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Colored border
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), color, 12)

    except Exception as e:
        cv2.putText(frame, "Processing...", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    cv2.imshow('Material Classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()