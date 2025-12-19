import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from config import AUGMENTED_DIR, FEATURES_FILE, LABELS_FILE

CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}

def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10)) 
    hist = hist.astype("float") / (hist.sum() + 1e-6)
    return hist  

def extract_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    hog_feat = hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
    return hog_feat 

def extract_color_hist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist 

def extract_edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    edges = cv2.Canny(gray, 100, 200)
    return np.array([edges.mean() / 255.0])  

def extract_shape(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(4)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / h if h != 0 else 0
    extent = area / (w * h) if w * h != 0 else 0
    return np.array([area, perimeter, aspect, extent])

def extract_features(img):
    img = cv2.resize(img, (256, 256))
    f1 = extract_lbp(img)
    f2 = extract_hog(img)
    f3 = extract_color_hist(img)
    f4 = extract_edge_density(img)
    f5 = extract_shape(img)
    return np.concatenate([f1, f2, f3, f4, f5])  

if __name__ == "__main__":
    features = []
    labels = []
    print("Extracting low-dim features...")
    for cls in CLASSES:
        cls_dir = os.path.join(AUGMENTED_DIR, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(cls_dir, fname))
                if img is None:
                    continue
                feat = extract_features(img)
                features.append(feat)
                labels.append(CLASS_TO_ID[cls])
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels)
    np.save(FEATURES_FILE, features)
    np.save(LABELS_FILE, labels)
    print("Done! Shape:", features.shape)