# Part 9: Experimental Comparison and Statistical Analysis

## 1. Experimental Setup

- Dataset: Fakeddit binary subset
- Test samples: 400
- Classes: FAKE (0) and REAL (1)
- Seed selection: highest validation Macro-F1
- Confidence intervals: 95% stratified bootstrap
- Statistical test: exact two-sided McNemar test
- Multiple comparison correction: Holm correction


## 2. Selected Seeds

| model                             |   selected_seed |   validation_macro_f1 |
|:----------------------------------|----------------:|----------------------:|
| Frozen text-only                  |              99 |              0.784952 |
| Frozen image-only                 |              99 |              0.784952 |
| Concatenation                     |              99 |              0.784952 |
| Concatenation + CLIP similarity   |              99 |              0.784952 |
| Cross-attention                   |              99 |              0.745077 |
| Cross-attention + CLIP similarity |              99 |              0.743918 |
| Late fusion                       |               7 |              0.8      |

## 3. Test Performance and Confidence Intervals

| model                             |   selected_seed |   validation_macro_f1 |   accuracy |   accuracy_ci_lower |   accuracy_ci_upper |   precision |   precision_ci_lower |   precision_ci_upper |   recall |   recall_ci_lower |   recall_ci_upper |   macro_f1 |   macro_f1_ci_lower |   macro_f1_ci_upper |   roc_auc |   roc_auc_ci_lower |   roc_auc_ci_upper |
|:----------------------------------|----------------:|----------------------:|-----------:|--------------------:|--------------------:|------------:|---------------------:|---------------------:|---------:|------------------:|------------------:|-----------:|--------------------:|--------------------:|----------:|-------------------:|-------------------:|
| DistilBERT fine-tuned             |              42 |              nan      |     0.7550 |              0.7100 |              0.7975 |      0.7554 |               0.7125 |               0.7988 |   0.7550 |            0.7100 |            0.7975 |     0.7549 |              0.7100 |              0.7975 |    0.8347 |             0.7950 |             0.8728 |
| Frozen text-only                  |              99 |                0.7850 |     0.7325 |              0.6900 |              0.7775 |      0.7346 |               0.6919 |               0.7805 |   0.7325 |            0.6900 |            0.7775 |     0.7319 |              0.6880 |              0.7773 |    0.8374 |             0.7981 |             0.8763 |
| Frozen image-only                 |              99 |                0.7850 |     0.7750 |              0.7325 |              0.8126 |      0.7752 |               0.7332 |               0.8139 |   0.7750 |            0.7325 |            0.8126 |     0.7749 |              0.7324 |              0.8125 |    0.8535 |             0.8147 |             0.8872 |
| Concatenation                     |              99 |                0.7850 |     0.8050 |              0.7650 |              0.8425 |      0.8053 |               0.7654 |               0.8432 |   0.8050 |            0.7650 |            0.8425 |     0.8050 |              0.7649 |              0.8425 |    0.8924 |             0.8598 |             0.9235 |
| Concatenation + CLIP similarity   |              99 |                0.7850 |     0.7825 |              0.7400 |              0.8225 |      0.7831 |               0.7415 |               0.8226 |   0.7825 |            0.7400 |            0.8225 |     0.7824 |              0.7400 |              0.8224 |    0.8806 |             0.8475 |             0.9111 |
| Cross-attention                   |              99 |                0.7451 |     0.7550 |              0.7125 |              0.7975 |      0.7587 |               0.7171 |               0.8018 |   0.7550 |            0.7125 |            0.7975 |     0.7541 |              0.7110 |              0.7971 |    0.8442 |             0.8058 |             0.8808 |
| Cross-attention + CLIP similarity |              99 |                0.7439 |     0.7925 |              0.7525 |              0.8325 |      0.7942 |               0.7533 |               0.8332 |   0.7925 |            0.7525 |            0.8325 |     0.7922 |              0.7522 |              0.8321 |    0.8667 |             0.8309 |             0.9012 |
| Late fusion                       |               7 |                0.8000 |     0.8300 |              0.7925 |              0.8650 |      0.8301 |               0.7927 |               0.8656 |   0.8300 |            0.7925 |            0.8650 |     0.8300 |              0.7924 |              0.8650 |    0.8913 |             0.8593 |             0.9212 |

## 4. McNemar Significance Tests

| model_a                           | model_b                           |   a_only_correct |   b_only_correct |   discordant_pairs |   p_value |   holm_adjusted_p_value | significant_0.05   |
|:----------------------------------|:----------------------------------|-----------------:|-----------------:|-------------------:|----------:|------------------------:|:-------------------|
| DistilBERT fine-tuned             | Frozen text-only                  |               30 |               21 |                 51 |   0.26244 |                 1.00000 | False              |
| DistilBERT fine-tuned             | Frozen image-only                 |               57 |               65 |                122 |   0.52642 |                 1.00000 | False              |
| DistilBERT fine-tuned             | Concatenation                     |               25 |               45 |                 70 |   0.02246 |                 0.40433 | False              |
| DistilBERT fine-tuned             | Concatenation + CLIP similarity   |               20 |               31 |                 51 |   0.16078 |                 1.00000 | False              |
| DistilBERT fine-tuned             | Cross-attention                   |               35 |               35 |                 70 |   1.00000 |                 1.00000 | False              |
| DistilBERT fine-tuned             | Cross-attention + CLIP similarity |               30 |               45 |                 75 |   0.10534 |                 1.00000 | False              |
| DistilBERT fine-tuned             | Late fusion                       |               27 |               57 |                 84 |   0.00140 |                 0.03371 | True               |
| Frozen text-only                  | Frozen image-only                 |               54 |               71 |                125 |   0.15212 |                 1.00000 | False              |
| Frozen text-only                  | Concatenation                     |                9 |               38 |                 47 |   0.00002 |                 0.00067 | True               |
| Frozen text-only                  | Concatenation + CLIP similarity   |                7 |               27 |                 34 |   0.00082 |                 0.02053 | True               |
| Frozen text-only                  | Cross-attention                   |               26 |               35 |                 61 |   0.30568 |                 1.00000 | False              |
| Frozen text-only                  | Cross-attention + CLIP similarity |               20 |               44 |                 64 |   0.00369 |                 0.08118 | False              |
| Frozen text-only                  | Late fusion                       |               19 |               58 |                 77 |   0.00001 |                 0.00027 | True               |
| Frozen image-only                 | Concatenation                     |               36 |               48 |                 84 |   0.22986 |                 1.00000 | False              |
| Frozen image-only                 | Concatenation + CLIP similarity   |               47 |               50 |                 97 |   0.83921 |                 1.00000 | False              |
| Frozen image-only                 | Cross-attention                   |               62 |               54 |                116 |   0.51591 |                 1.00000 | False              |
| Frozen image-only                 | Cross-attention + CLIP similarity |               45 |               52 |                 97 |   0.54261 |                 1.00000 | False              |
| Frozen image-only                 | Late fusion                       |               15 |               37 |                 52 |   0.00319 |                 0.07326 | False              |
| Concatenation                     | Concatenation + CLIP similarity   |               17 |                8 |                 25 |   0.10775 |                 1.00000 | False              |
| Concatenation                     | Cross-attention                   |               34 |               14 |                 48 |   0.00552 |                 0.11582 | False              |
| Concatenation                     | Cross-attention + CLIP similarity |               24 |               19 |                 43 |   0.54238 |                 1.00000 | False              |
| Concatenation                     | Late fusion                       |               14 |               24 |                 38 |   0.14331 |                 1.00000 | False              |
| Concatenation + CLIP similarity   | Cross-attention                   |               32 |               21 |                 53 |   0.16898 |                 1.00000 | False              |
| Concatenation + CLIP similarity   | Cross-attention + CLIP similarity |               24 |               28 |                 52 |   0.67781 |                 1.00000 | False              |
| Concatenation + CLIP similarity   | Late fusion                       |               16 |               35 |                 51 |   0.01097 |                 0.21947 | False              |
| Cross-attention                   | Cross-attention + CLIP similarity |               10 |               25 |                 35 |   0.01667 |                 0.31680 | False              |
| Cross-attention                   | Late fusion                       |               21 |               51 |                 72 |   0.00053 |                 0.01391 | True               |
| Cross-attention + CLIP similarity | Late fusion                       |               22 |               37 |                 59 |   0.06744 |                 1.00000 | False              |

## 5. Complementarity Analysis

| model_a               | model_b           |   both_correct |   only_a_correct |   only_b_correct |   both_wrong |
|:----------------------|:------------------|---------------:|-----------------:|-----------------:|-------------:|
| DistilBERT fine-tuned | Frozen image-only |            245 |               57 |               65 |           33 |
| Frozen text-only      | Frozen image-only |            239 |               54 |               71 |           36 |
| DistilBERT fine-tuned | Late fusion       |            275 |               27 |               57 |           41 |
| Frozen image-only     | Late fusion       |            295 |               15 |               37 |           53 |

## 6. Disagreement Analysis

- Baseline: DistilBERT fine-tuned
- Multimodal comparison: Late fusion
- Number of disagreement cases: 84


## 7. Interpretation Notes

- The selected seed for each model was chosen using validation Macro-F1.
- Test results were used only for final evaluation.
- Bootstrap confidence intervals quantify uncertainty in test metrics.
- McNemar tests compare paired classification errors on the same test samples.
- Holm-adjusted p-values account for multiple pairwise comparisons.
- Statistical significance does not automatically imply practical importance.
- The fine-tuned DistilBERT baseline and frozen-feature text-only model are different experimental systems.
- The image-only baseline is the frozen-feature Part 6 model; a separate fine-tuned Part 4 image-only result was not available.
