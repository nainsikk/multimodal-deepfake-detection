
import pytest

from evaluation.mismatch import calculate_mismatch_score


def test_high_mismatch_score():
    result = calculate_mismatch_score(0.10, 0.90)

    assert result["mismatch_score"] == 80.0
    assert result["disagreement"] is True
    assert result["text_prediction"] == "Real"
    assert result["image_prediction"] == "Fake"


def test_no_mismatch():
    result = calculate_mismatch_score(0.20, 0.20)

    assert result["mismatch_score"] == 0.0
    assert result["disagreement"] is False


def test_moderate_mismatch():
    result = calculate_mismatch_score(0.20, 0.50)

    assert result["mismatch_score"] == 30.0
    assert result["disagreement"] is True


def test_custom_threshold():
    result = calculate_mismatch_score(
        0.40,
        0.60,
        threshold=0.70
    )

    assert result["text_prediction"] == "Real"
    assert result["image_prediction"] == "Real"
    assert result["disagreement"] is False


@pytest.mark.parametrize(
    "text_probability, image_probability",
    [
        (-0.1, 0.5),
        (0.5, 1.1),
        (None, 0.5),
        (0.5, float("nan")),
        (float("inf"), 0.5),
        (True, 0.5)
    ]
)
def test_invalid_probabilities(text_probability, image_probability):
    with pytest.raises(ValueError):
        calculate_mismatch_score(
            text_probability,
            image_probability
        )


def test_invalid_threshold():
    with pytest.raises(ValueError):
        calculate_mismatch_score(
            0.2,
            0.8,
            threshold=1.5
        )
