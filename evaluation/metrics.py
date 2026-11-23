
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)


def evaluate_model(y_true, y_pred, y_score=None):
    """
    Evaluate a binary classification model.

    Labels:
        0 = Real
        1 = Fake

    Parameters:
        y_true: Actual labels.
        y_pred: Predicted labels.
        y_score: Predicted probability of the Fake class.
                 Optional, values should be between 0 and 1.

    Returns:
        Dictionary containing evaluation metrics.
    """

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Actual labels and predicted labels must have equal lengths."
        )

    if len(y_true) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    # Validate actual labels.
    if not set(y_true).issubset({0, 1}):
        raise ValueError(
            "Actual labels must contain only 0 (Real) or 1 (Fake)."
        )

    # Validate predicted labels.
    if not set(y_pred).issubset({0, 1}):
        raise ValueError(
            "Predicted labels must contain only 0 (Real) or 1 (Fake)."
        )

    if y_score is not None:
        if len(y_score) != len(y_true):
            raise ValueError(
                "Prediction scores and actual labels must have equal lengths."
            )

        # Check that all prediction scores are finite.
        if not all(np.isfinite(score) for score in y_score):
            raise ValueError(
                "Prediction scores must contain only finite values."
            )

        # Check that all prediction scores are between 0 and 1.
        if not all(0 <= score <= 1 for score in y_score):
            raise ValueError(
                "Prediction scores must be between 0 and 1."
            )

    labels = [0, 1]

    results = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision": precision_score(
            y_true, y_pred,
            labels=labels,
            average="binary",
            zero_division=0
        ),

        "recall": recall_score(
            y_true, y_pred,
            labels=labels,
            average="binary",
            zero_division=0
        ),

        "f1_score": f1_score(
            y_true, y_pred,
            labels=labels,
            average="binary",
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_true, y_pred,
            labels=labels
        ).tolist(),

        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=["Real", "Fake"],
            zero_division=0
        ),

        "roc_auc": None,
        "pr_auc": None
    }

    # ROC-AUC and PR-AUC require probability scores.
    # Both classes must be present in the actual labels.
    if y_score is not None and len(set(y_true)) == 2:
        results["roc_auc"] = roc_auc_score(y_true, y_score)

        results["pr_auc"] = average_precision_score(
            y_true, y_score
        )

    return results


def print_evaluation(results):
    """Display evaluation results."""

    print("\n========== MODEL EVALUATION ==========")

    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1 Score  : {results['f1_score']:.4f}")

    if results["roc_auc"] is not None:
        print(f"ROC-AUC   : {results['roc_auc']:.4f}")
        print(f"PR-AUC    : {results['pr_auc']:.4f}")
    else:
        print("ROC-AUC   : Not available")
        print("PR-AUC    : Not available")

    print("\nConfusion Matrix:")
    print(np.array(results["confusion_matrix"]))

    print("\nClassification Report:")
    print(results["classification_report"])

    print("======================================")
