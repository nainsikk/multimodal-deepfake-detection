
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
