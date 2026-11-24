
"""Category-wise evaluation for generalization analysis."""

import pandas as pd

from evaluation.metrics import evaluate_model


def evaluate_by_category(
    data,
    category_col="category",
    true_col="true_label",
    pred_col="predicted_label",
    probability_real_col="probability_real"
):
    """
    Evaluate model performance separately for each category.

    Labels:
        0 = FAKE
        1 = REAL

    The input probability column contains P(REAL).
    It is converted to P(FAKE) for the existing evaluation metrics.

    Args:
        data: Pandas DataFrame containing predictions and categories.
        category_col: Column containing category names.
        true_col: Column containing actual labels.
        pred_col: Column containing predicted labels.
        probability_real_col: Optional column containing P(REAL).

    Returns:
        Dictionary containing metrics for each category.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Evaluation data cannot be empty.")

    required_columns = {
        category_col,
        true_col,
        pred_col
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if probability_real_col in data.columns:
        if data[probability_real_col].isna().any():
            raise ValueError("Probability values cannot be missing.")

        if not data[probability_real_col].between(0, 1).all():
            raise ValueError(
                "Probability values must be between 0 and 1."
            )

    if data[list(required_columns)].isna().any().any():
        raise ValueError(
            "Category names and labels cannot contain missing values."
        )

    if not set(data[true_col]).issubset({0, 1}):
        raise ValueError("Actual labels must be 0 (FAKE) or 1 (REAL).")

    if not set(data[pred_col]).issubset({0, 1}):
        raise ValueError("Predicted labels must be 0 (FAKE) or 1 (REAL).")

    results = {}

    for category, group in data.groupby(category_col, sort=True):

        y_true = group[true_col].tolist()
        y_pred = group[pred_col].tolist()

        y_score = None

        if probability_real_col in data.columns:
            # Existing metrics.py expects P(FAKE), not P(REAL).
            y_score = (
                1 - group[probability_real_col]
            ).tolist()

        metrics = evaluate_model(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score
        )

        results[str(category)] = {
            "sample_count": len(group),
            "real_count": sum(label == 1 for label in y_true),
            "fake_count": sum(label == 0 for label in y_true),
            "metrics": metrics
        }

    return results
