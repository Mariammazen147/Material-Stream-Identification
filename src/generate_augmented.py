import os
import cv2
import albumentations as A
import random
from multiprocessing import Pool
from config import RAW_DIR, AUGMENTED_DIR

RAW_ROOT = RAW_DIR
AUG_ROOT = AUGMENTED_DIR

TARGET = 500
CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

transform = A.Compose([
    A.Rotate(limit=30, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.GaussNoise(p=0.5),
    A.RandomResizedCrop(size=(256,256), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.7),
    A.CoarseDropout(p=0.4),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
])

def augment(args):
    img_path, out_path, idx = args
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    aug = transform(image=img)['image']
    aug = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(out_path, f"aug_{idx:06d}.jpg"), aug)


if __name__ == "__main__":
    os.makedirs(AUG_ROOT, exist_ok=True)

    for cls in CLASSES:
        raw_path = os.path.join(RAW_ROOT, cls)
        aug_path = os.path.join(AUG_ROOT, cls)
        os.makedirs(aug_path, exist_ok=True)

        originals = [
            os.path.join(raw_path, f)
            for f in os.listdir(raw_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        valid_originals = []
        for p in originals:
            img = cv2.imread(p)
            if img is None:
                print(f"Skipping unreadable: {p}")
            else:
                valid_originals.append(p)

        current = len(valid_originals)
        needed = max(0, TARGET - current)

        # Save originals
        for i, p in enumerate(valid_originals):
            img = cv2.imread(p)
            cv2.imwrite(os.path.join(aug_path, f"img_{i:06d}.jpg"), img)

        # Generate augmented images
        if needed > 0:
            tasks = [
                (random.choice(valid_originals), aug_path, i + current)
                for i in range(needed)
            ]
            with Pool(6) as p:
                p.map(augment, tasks)

        print(f"{cls}: {current} ---> {TARGET} (generated {needed})")
