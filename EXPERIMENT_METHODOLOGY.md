# Experiment Methodology
## Multimodal Deepfake Detection and Explainable Content Verification

## 1. Introduction

The objective of the experimental methodology is to evaluate and compare different machine learning approaches for multimodal image-caption classification.

The experiments investigate whether combining textual and visual information improves classification performance compared with using either modality independently.

The project also applies statistical analysis to measure performance uncertainty and determine whether differences between model predictions are statistically significant.

## 2. Dataset and Label Mapping

The experiments use a binary subset of the Fakeddit dataset.

| Property | Description |
|---|---|
| Dataset | Fakeddit binary subset |
| Test samples | 400 |
| Number of classes | 2 |
| Label 0 | FAKE |
| Label 1 | REAL |

The classification interpretation is based on image-caption consistency:

- FAKE (0): Image and caption do not match.
- REAL (1): Image and caption match.

The 10-sample synthetic dataset is used separately for initial pipeline testing and is not the benchmark used in these experiments.

## 3. Experimental Models

Eight experimental systems are compared.

### 3.1 Text-only models

**1. DistilBERT fine-tuned**

A transformer-based language model fine-tuned for text classification. It uses textual information to predict the class.

**2. Frozen text-only**

Uses frozen text representations as input to a classification model. The pretrained feature extractor is not fine-tuned during this experimental approach.

### 3.2 Image-only model

**3. Frozen image-only**

Uses visual features extracted from images. The image feature extractor remains frozen.

### 3.3 Multimodal models

**4. Concatenation**

Combines text and image feature vectors into a single representation before classification.

**5. Concatenation + CLIP similarity**

Combines text and image features with an additional CLIP-based similarity signal.

**6. Cross-attention**

Uses cross-attention to allow textual and visual representations to interact and exchange information.

**7. Cross-attention + CLIP similarity**

Combines cross-attention with an additional CLIP similarity feature.

**8. Late fusion**

Combines predictions or learned representations from separate modality-specific components to produce a final classification.

## 4. Experimental Workflow

The Part 9 experimental comparison follows these stages:

### Stage 1: Load the test dataset

The script loads the test dataset from:

`data/test/test_clean.csv`

It performs checks to confirm:

- Exactly 400 test samples are present.
- Both binary labels, 0 and 1, are available.
- Sample IDs and label ordering remain consistent across model prediction files.

### Stage 2: Load model predictions

Previously generated model predictions are loaded from the project's results directories.

The script loads:

- Fine-tuned DistilBERT predictions from Part 3.
- Frozen-feature and concatenation predictions from Part 6.
- Cross-attention and late-fusion predictions from Part 7.

The experiment script compares these predictions rather than training all eight models again.

### Stage 3: Select the best random seed

Multiple random seeds are evaluated during model development.

The selected seed is the one with the highest validation Macro-F1 score.

The selection is based on validation performance, not test performance.

This helps prevent test-set information from influencing model selection.

The fine-tuned DistilBERT baseline uses its saved Part 3 predictions and the seed recorded in the experiment report.

### Stage 4: Calculate evaluation metrics

The selected model predictions are evaluated on the test dataset.

The following metrics are calculated:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified samples |
| Precision | Measures classification precision using macro averaging |
| Recall | Measures classification recall using macro averaging |
| Macro-F1 | Calculates the average F1-score across both classes |
| ROC-AUC | Measures the model's ability to distinguish between the two classes |

Macro averaging gives equal importance to both classes when calculating Precision, Recall and F1-score.

### Stage 5: Calculate confidence intervals

The experiment uses stratified bootstrap sampling with 2,000 iterations.

In each iteration:

1. FAKE samples are sampled with replacement.
2. REAL samples are sampled with replacement.
3. Both groups are combined and shuffled.
4. Evaluation metrics are recalculated.

The 2.5th and 97.5th percentiles are used to calculate the 95% confidence interval.

This provides an estimate of uncertainty around the test metrics.

### Stage 6: Statistical significance testing

The exact two-sided McNemar test is used to compare pairs of models.

It examines samples where one model is correct and the other is incorrect.

The test helps determine whether the difference in paired classification errors is statistically significant.

Since multiple model comparisons are performed, the Holm correction is applied to adjust the p-values and control the family-wise error rate.

A result is considered statistically significant when the Holm-adjusted p-value is below 0.05.

### Stage 7: Complementarity analysis

Complementarity analysis investigates whether two models make similar or different classification errors.

It calculates four groups:

- Both correct: Both models classify the sample correctly.
- Only model A correct: Model A is correct and model B is incorrect.
- Only model B correct: Model B is correct and model A is incorrect.
- Both wrong: Both models classify the sample incorrectly.

This analysis helps understand whether combining different models or modalities may provide additional information.

### Stage 8: Disagreement analysis

The experiment compares the fine-tuned DistilBERT baseline with the best multimodal model selected using validation Macro-F1.

It identifies samples for which the two models produce different predictions.

The disagreement cases are saved with:

- Sample ID and true label.
- DistilBERT prediction.
- Multimodal prediction.
- Correctness of each model's prediction.

This helps identify examples where multimodal information changes the classification decision.

### Stage 9: Generate visualizations

The experiment generates:

- Confusion matrices for all evaluated models.
- A model comparison figure for Accuracy, Macro-F1 and ROC-AUC.
- Error bars representing 95% confidence intervals.

These visualizations make it easier to compare model performance and understand classification errors.

## 5. Experimental Configuration

The main experimental configuration in `experiments.py` is:

| Parameter | Value |
|---|---|
| Random state | 42 |
| Bootstrap iterations | 2,000 |
| Confidence level | 95% |
| Test samples | 400 |
| Classes | 2 |
| Significance threshold | 0.05 |
| Seed selection criterion | Highest validation Macro-F1 |
| Statistical test | Exact two-sided McNemar test |
| Multiple comparison correction | Holm correction |

## 6. Output Files

The experimental script saves its outputs in:

`results/part9/`

Important output files include:

| File | Purpose |
|---|---|
| `selected_seeds.csv` | Stores selected seeds and validation Macro-F1 |
| `test_metrics_with_ci.csv` | Stores test metrics and confidence intervals |
| `mcnemar_tests.csv` | Stores paired statistical comparison results |
| `complementarity.csv` | Stores model complementarity analysis |
| `disagreement_cases.csv` | Stores samples with differing model predictions |
| `all_model_predictions.csv` | Stores predictions for all evaluated models |
| `summary.json` | Stores the experiment summary |
| `report.md` | Contains the generated experimental report |
| `model_comparison.png` | Visual comparison of model metrics |
| `confusion_matrices/` | Contains confusion matrix images for each model |

## 7. Reported Experimental Results

According to the existing Part 9 experimental report:

| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|
| DistilBERT fine-tuned | 75.50% | 0.7549 | 0.8347 |
| Frozen text-only | 73.25% | 0.7319 | 0.8374 |
| Frozen image-only | 77.50% | 0.7749 | 0.8535 |
| Concatenation | 80.50% | 0.8050 | 0.8924 |
| Concatenation + CLIP similarity | 78.25% | 0.7824 | 0.8806 |
| Cross-attention | 75.50% | 0.7541 | 0.8442 |
| Cross-attention + CLIP similarity | 79.25% | 0.7922 | 0.8667 |
| Late fusion | **83.00%** | **0.8300** | 0.8913 |

Late fusion achieved the highest reported Accuracy and Macro-F1.

These results are taken from the team's existing experimental report.

## 8. Interpretation and Limitations

- Test results are used for final evaluation, not model or seed selection.
- Confidence intervals provide information about metric uncertainty.
- Statistical significance does not automatically indicate practical importance.
- The fine-tuned DistilBERT model and frozen-feature text-only model are different experimental systems.
- The image-only baseline uses frozen image features.
- Category-wise generalization and robustness under image or text corruption require separate evaluation.
- Image-caption mismatch classification does not establish whether an image or video is synthetically generated or manipulated.

## 9. Conclusion

The experimental methodology provides a structured approach to comparing text-only, image-only and multimodal classification systems.

By combining standard evaluation metrics, bootstrap confidence intervals, McNemar significance testing, complementarity analysis and disagreement analysis, the project evaluates both predictive performance and differences between model behaviour.

The methodology supports a more reliable comparison of multimodal approaches and helps identify the benefits and limitations of combining textual and visual information.