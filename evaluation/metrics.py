
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_pred):
    """
    Evaluate a classification model using standard metrics.

    Parameters:
        y_true: Actual labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing evaluation metrics.
    """

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Actual labels and predicted labels must have equal lengths."
        )

    if len(y_true) == 0:
        raise ValueError("Evaluation data cannot be empty.")

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
        )
    }

    return results


def print_evaluation(results):
    """Display evaluation results."""

    print("\n========== MODEL EVALUATION ==========")

    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1 Score  : {results['f1_score']:.4f}")

    print("\nConfusion Matrix:")
    print(np.array(results["confusion_matrix"]))

    print("\nClassification Report:")
    print(results["classification_report"])

    print("======================================")
