### Data Augmentation

**Introduction**: To improve generalization, we applied augmentations to increase the dataset by >30% and balance to ~500 images per class.

**Original Distribution**: [Insert class_distribution_original.png] Total: X images.

**Augmented Distribution**: [Insert class_distribution_augmented.png] Total: ~3000 images (increase of Y%).

**Pipeline Justification** (Albumentations):

| Transformation | Parameters | Justification |
|----------------|------------|--------------|
| Rotate | limit=30, p=0.7 | Simulates orientation variations. |
| HorizontalFlip | p=0.5 | No directional bias in materials. |
| RandomBrightnessContrast | limit=0.3, p=0.7 | Handles lighting variations. |
| GaussNoise | var_limit=(10,50), p=0.5 | Mimics sensor noise. |
| RandomResizedCrop | 224x224, scale=0.8-1.0, p=0.7 | Simulates scale/distance. |
| CoarseDropout | max_holes=8, p=0.4 | Simulates occlusions. |
| HueSaturationValue | limit=20/30/20, p=0.5 | Color variations. |

**Examples**: [Insert sample_images_original.png vs. augmented].