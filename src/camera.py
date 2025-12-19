import cv2
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from feature_extraction import extract_features
from config import FEATURES_FILE, LABELS_FILE

def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def texture_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return np.mean(np.sqrt(gx**2 + gy**2))


print("Loading validation data...")
X = np.load(FEATURES_FILE)
y = np.load(LABELS_FILE)
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

try:
    svm = joblib.load('svm_model.joblib')
    svm_scaler = joblib.load('scaler.joblib')
    svm_threshold = np.load("rejection_threshold.npy")[0]

    svm_acc = accuracy_score(y_val, svm.predict(svm_scaler.transform(X_val)))
except Exception as e:
    svm, svm_scaler, svm_threshold, svm_acc = None, None, None, 0

try:
    knn = joblib.load('knn_model.joblib')
    knn_scaler = joblib.load('knn_scaler.joblib')
    knn_selector = joblib.load("knn_selector.joblib")
    knn_threshold = np.load("knn_threshold.npy")[0]

    X_val_sel = knn_selector.transform(X_val)
    X_val_scaled = knn_scaler.transform(X_val_sel)


    knn_acc = accuracy_score(y_val, knn.predict(knn_scaler.transform(X_val_scaled)))
except Exception as e:
    knn, knn_scaler, knn_selector, knn_threshold, knn_acc = None, None, None, None, 0

if svm_acc >= knn_acc:
    model_type = "SVM"
else:
    model_type = "KNN"

print(f"\nUsing {model_type}")

class_names = {0:'GLASS', 1:'PAPER', 2:'CARDBOARD', 3:'PLASTIC', 4:'METAL', 5:'TRASH', 6:'UNKNOWN'}
colors = {0:(255,200,100), 1:(240,240,240), 2:(100,150,200), 3:(100,255,255), 4:(200,200,200), 5:(50,50,200), 6:(0,0,255)}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        h, w, _ = frame.shape
        crop = frame[h//4:3*h//4, w//4:3*w//4]

        blur = blur_score(crop)
        texture = texture_energy(crop)

        if blur < 60 or texture < 8:
            pred = 6
            confidence = 0
            label = class_names[pred]
            color = colors[pred]
        else:
            features = extract_features(crop).reshape(1, -1)




        if model_type == "SVM":
            features_scaled = svm_scaler.transform(features)

            decision = svm.decision_function(features_scaled)
            confidence_score = np.max(np.abs(decision))

            if confidence_score < svm_threshold:
                pred = 6
                confidence = 0
            else:
                pred = svm.predict(features_scaled)[0]
                confidence = min(100, max(0, (confidence_score - svm_threshold) / confidence_score * 100))


        else:
            features_selected = knn_selector.transform(features)
            features_scaled = knn_scaler.transform(features_selected)

            distances, _ = knn.kneighbors(features_scaled)
            avg_dist = distances.mean()

            if avg_dist > knn_threshold:
                pred = 6
                confidence = 0
            else:
                pred = knn.predict(features_scaled)[0]
                confidence = max(0, 100 - avg_dist * 20)

        label = class_names[pred]
        color = colors[pred]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (630, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, label, (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)

        cv2.putText(frame, f"Confidence: {confidence:.0f}%",
                    (30, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        cv2.rectangle(frame, (0, 0),
                      (frame.shape[1]-1, frame.shape[0]-1),
                      color, 12)

    except Exception as e:
        cv2.putText(frame, "Processing...",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 255, 255), 3)

    cv2.imshow("Material Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()