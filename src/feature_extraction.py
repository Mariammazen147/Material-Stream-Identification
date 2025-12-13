import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from config import AUGMENTED_DIR, FEATURES_FILE, LABELS_FILE

CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}

def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float") / (hist.sum() + 1e-6)

    return hist  


def extract_edge_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    sobel_hist, _ = np.histogram(mag.ravel(), bins=64, range=(0, 255))
    sobel_hist = sobel_hist / (sobel_hist.sum() + 1e-6)

    
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean() / 255.0  # 1 value

    return np.hstack([sobel_hist, edge_density])  # 64 + 1 = 65 features


def extract_shape_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros(6)

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / (h + 1e-6)
    extent = area / (w * h + 1e-6)

    hull = cv2.convexHull(cnt)
    solidity = area / (cv2.contourArea(hull) + 1e-6)

    circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)

    return np.array([area, perimeter, aspect_ratio, extent, solidity, circularity])


def extract_kmeans_colors(img, k=3):
    img_resized = cv2.resize(img, (64, 64))
    pixels = img_resized.reshape(-1, 3).astype(np.float32)

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        3,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    
    centers = centers.flatten()  
    return centers / 255.0


def extract_features(img):
    img = cv2.resize(img, (256, 256))

    f1 = extract_lbp(img)
    f2 = extract_edge_features(img)
    f3 = extract_shape_features(img)
    f4 = extract_kmeans_colors(img)

    return np.concatenate([f1, f2, f3, f4]) 


if __name__ == "__main__":
    features = []
    labels = []

    print("Extracting features from augmented dataset...")

    for cls in CLASSES:
        cls_dir = os.path.join(AUGMENTED_DIR, cls)
        class_id = CLASS_TO_ID[cls]

        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                fpath = os.path.join(cls_dir, fname)

                img = cv2.imread(fpath)
                if img is None:
                    continue

                feat = extract_features(img)
                features.append(feat)
                labels.append(class_id)

    features = np.array(features)
    labels = np.array(labels)

    print("Saving feature matrix...")
    np.save(FEATURES_FILE, features)
    np.save(LABELS_FILE, labels)

    print("Done! Extracted features shape:", features.shape)
    print("Labels shape:", labels.shape)
