# Member 2 – Backend, Evaluation, Generalization and Robustness

## 1. Role and Responsibilities

Member 2 was responsible for backend development, database integration, evaluation utilities, mismatch scoring, and robustness and generalization analysis for the Multimodal Deepfake Detection project.

## 2. Backend Development

Implemented a backend using **FastAPI**.

Key functionalities:
- Text and image input handling.
- Input validation and error handling.
- Analysis API for submitting content.
- Health-check and model-status endpoint.
- Analysis history and individual result retrieval.
- PDF report generation.
- Sanity-check endpoint for checking predictor readiness.

### Database Integration

Used **SQLite** with **SQLAlchemy** for database management.

Implemented:
- Analysis record storage.
- Prediction and confidence storage.
- Timestamp and result storage.
- Retrieval of previous analysis records.

**Current limitation:** The backend currently uses a mock predictor. Integration with the final trained multimodal model is pending.

## 3. Evaluation Metrics

Implemented evaluation utilities in `evaluation/metrics.py`.

Metrics supported:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Classification Report

The project uses the following label convention:
- `0 = FAKE`
- `1 = REAL`

For evaluation, FAKE is treated as the positive class.

## 4. Mismatch Scoring

Implemented mismatch scoring in `evaluation/mismatch.py`.

The function compares the fake probabilities from the text and image modalities.

The mismatch score is calculated as:

**Mismatch Score = |P(FAKE from text) − P(FAKE from image)| × 100**

The score ranges from 0 to 100.

The implementation also provides:
- Probability validation.
- Configurable classification threshold.
- Disagreement detection.
- Low, Moderate and High mismatch interpretations.

The mismatch score measures differences between modality probabilities; it does not independently prove that content is fake.

## 5. Robustness Analysis

Implemented image and text perturbation utilities to prepare inputs for robustness testing.

### Image Robustness

File: `evaluation/robustness.py`

Supported image corruptions:
- Gaussian Blur
- Brightness Changes
- Gaussian Noise

Five severity levels are supported for each corruption.

Tests: **17 passed**

### Text Robustness

File: `evaluation/text_robustness.py`

Supported text perturbations:
- Word Deletion
- Word Swapping
- Character Noise

Five severity levels are supported.

Tests: **15 passed**

**Current limitation:** These utilities and their unit tests are implemented. Actual model-performance comparisons under corrupted inputs require integration with the trained model and have not yet been completed.

## 6. Generalization Analysis

File: `evaluation/generalization.py`

Implemented category-wise evaluation to measure model performance separately for different categories.

The module calculates:
- Number of samples per category.
- REAL and FAKE sample counts.
- Accuracy and other evaluation metrics.
- ROC-AUC and PR-AUC when suitable probability scores and both classes are available.

Tests: **9 passed**

A small synthetic pilot dataset is available for testing the dataset pipeline.

**Current limitation:** The synthetic dataset is intended for pipeline testing, not research-level performance claims. Actual category-wise generalization results require suitable category-labelled test predictions from the integrated model.

## 7. Evaluation Report Export

Implemented `evaluation/export.py` to export evaluation results into timestamped JSON reports.

The export utility:
- Creates the output directory if required.
- Stores model information.
- Records the evaluation date.
- Saves evaluation metrics in JSON format.

## 8. Testing and Verification

The complete project test suite was executed using:

```bash
python -m pytest -v
```

**Final result: 111 tests passed in 2.65 seconds.**

The test suite covers:
- Backend APIs.
- Database operations.
- Input validation.
- Evaluation metrics.
- Mismatch scoring.
- Image robustness utilities.
- Text robustness utilities.
- Generalization evaluation.
- Report generation.

The test suite also produced a non-fatal `pytest-asyncio` configuration warning.

## 9. Technologies and Libraries

- Python
- FastAPI
- SQLite
- SQLAlchemy
- Pydantic
- Pandas
- NumPy
- Pillow
- Scikit-learn
- Pytest

## 10. Pending Work and Limitations

The following work remains dependent on final model integration and suitable evaluation data:

1. Replace the mock predictor with the trained multimodal model.
2. Evaluate the integrated model on the actual test dataset.
3. Run image and text corruption experiments and compare model performance across severity levels.
4. Perform category-wise generalization experiments using category-labelled predictions.
5. Record and report actual experimental results.

## 11. Conclusion

Member 2 completed the backend foundation, database integration, evaluation utilities, mismatch scoring, image and text robustness utilities, and category-wise generalization evaluation module.

The complete project test suite passed all 111 tests. Further model-dependent experiments will be performed after integration of the final multimodal model.