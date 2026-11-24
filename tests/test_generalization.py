
import pandas as pd
import pytest

from evaluation.generalization import evaluate_by_category


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "category": ["beach", "beach", "soccer", "soccer"],
        "true_label": [1, 0, 1, 0],
        "predicted_label": [1, 0, 0, 0],
        "probability_real": [0.90, 0.20, 0.40, 0.10]
    })


def test_evaluates_each_category(sample_data):
    results = evaluate_by_category(sample_data)

    assert set(results.keys()) == {"beach", "soccer"}
    assert results["beach"]["sample_count"] == 2
    assert results["soccer"]["sample_count"] == 2


def test_counts_real_and_fake_samples(sample_data):
    results = evaluate_by_category(sample_data)

    assert results["beach"]["real_count"] == 1
    assert results["beach"]["fake_count"] == 1


def test_calculates_category_accuracy(sample_data):
    results = evaluate_by_category(sample_data)

    assert results["beach"]["metrics"]["accuracy"] == 1.0
    assert results["soccer"]["metrics"]["accuracy"] == 0.5


def test_auc_is_calculated_when_both_labels_exist(sample_data):
    results = evaluate_by_category(sample_data)

    assert results["beach"]["metrics"]["roc_auc"] is not None


def test_missing_required_column(sample_data):
    data = sample_data.drop(columns=["category"])

    with pytest.raises(ValueError):
        evaluate_by_category(data)


def test_empty_dataframe():
    with pytest.raises(ValueError):
        evaluate_by_category(pd.DataFrame())


def test_invalid_labels(sample_data):
    sample_data.loc[0, "true_label"] = 2

    with pytest.raises(ValueError):
        evaluate_by_category(sample_data)


def test_invalid_probability(sample_data):
    sample_data.loc[0, "probability_real"] = 1.5

    with pytest.raises(ValueError):
        evaluate_by_category(sample_data)


def test_non_dataframe_input():
    with pytest.raises(TypeError):
        evaluate_by_category([1, 2, 3])
