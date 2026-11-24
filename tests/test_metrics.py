
import pytest

from evaluation.metrics import evaluate_model


def test_evaluation_metrics():
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0]

    results = evaluate_model(y_true, y_pred)

    assert results["accuracy"] == pytest.approx(0.6667, abs=0.0001)
    assert results["precision"] == pytest.approx(0.6667, abs=0.0001)
    assert results["recall"] == pytest.approx(0.6667, abs=0.0001)
    assert results["f1_score"] == pytest.approx(0.6667, abs=0.0001)


def test_fake_class_is_positive():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    results = evaluate_model(y_true, y_pred)

    assert results["precision"] == pytest.approx(1.0)
    assert results["recall"] == pytest.approx(0.5)
    assert results["f1_score"] == pytest.approx(2 / 3)


def test_confusion_matrix():
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0]

    results = evaluate_model(y_true, y_pred)

    assert results["confusion_matrix"] == [
        [2, 1],
        [1, 2]
    ]


def test_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluate_model([0, 1], [0])


def test_empty_data():
    with pytest.raises(ValueError):
        evaluate_model([], [])


def test_roc_auc_and_pr_auc():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]

    # Higher probabilities for FAKE samples (label 0).
    y_score = [0.9, 0.8, 0.2, 0.1]

    results = evaluate_model(y_true, y_pred, y_score)

    assert results["roc_auc"] == pytest.approx(1.0)
    assert results["pr_auc"] == pytest.approx(1.0)


def test_auc_unavailable_without_scores():
    results = evaluate_model([0, 1], [0, 1])

    assert results["roc_auc"] is None
    assert results["pr_auc"] is None


def test_mismatched_score_lengths():
    with pytest.raises(ValueError):
        evaluate_model([0, 1], [0, 1], [0.5])


def test_invalid_prediction_scores():
    with pytest.raises(ValueError):
        evaluate_model([0, 1], [0, 1], [0.2, 1.5])


def test_invalid_actual_labels():
    with pytest.raises(ValueError):
        evaluate_model([0, 2], [0, 1])


def test_invalid_predicted_labels():
    with pytest.raises(ValueError):
        evaluate_model([0, 1], [0, 2])


def test_auc_unavailable_for_single_class():
    results = evaluate_model(
        [0, 0, 0],
        [0, 0, 0],
        [0.9, 0.8, 0.7]
    )

    assert results["roc_auc"] is None
    assert results["pr_auc"] is None


def test_rejects_invalid_actual_labels_only():
    with pytest.raises(ValueError, match="Actual labels"):
        evaluate_model([2, 2], [0, 0])


def test_rejects_invalid_predicted_labels_only():
    with pytest.raises(ValueError, match="Predicted labels"):
        evaluate_model([0, 0], [2, 2])


def test_rejects_nan_prediction_scores():
    with pytest.raises(ValueError, match="finite"):
        evaluate_model([0, 1], [0, 1], [0.2, float("nan")])


def test_rejects_infinite_prediction_scores():
    with pytest.raises(ValueError, match="finite"):
        evaluate_model([0, 1], [0, 1], [0.2, float("inf")])


def test_rejects_negative_infinite_prediction_scores():
    with pytest.raises(ValueError, match="finite"):
        evaluate_model([0, 1], [0, 1], [0.2, float("-inf")])
