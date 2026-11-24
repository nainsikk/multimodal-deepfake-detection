# Small Synthetic Test Dataset

## 1. Purpose
This dataset contains 10 synthetic image-caption pairs created to test the data loading, preprocessing, and evaluation pipeline of the Multimodal Deepfake Detection project.

## 2. Dataset Structure
- `test_dataset.csv`: Contains sample IDs, image paths, captions, labels, and categories.
- `images/`: Contains five generated PNG images.
- `create_images.py`: Python script used to generate the synthetic images.

## 3. Dataset Distribution
- Total samples: 10
- REAL samples: 5
- FAKE samples: 5

## 4. Label Mapping
- Label 0 = FAKE (image and caption do not match).
- Label 1 = REAL (image and caption match).

## 5. Categories
The dataset includes five categories:
- Beach
- Soccer
- Mountain
- Dog
- Car

Each category has one matching pair and one mismatched pair.

## 6. Validation
The dataset was validated using Python:
- All 10 records were loaded successfully.
- All image paths were found.
- No images were missing.
- All labels were valid (0 or 1).
- The dataset has a balanced class distribution.

## 7. Limitations
This is a small synthetic dataset containing computer-generated illustrations. It is intended only for initial pipeline testing and debugging. It is not suitable for reliable model training, benchmarking, or research-level performance claims.

## 8. Future Work
The pilot dataset may later be expanded using real-world image-caption pairs and a larger, diverse dataset with carefully verified labels.