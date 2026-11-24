
import pytest

from evaluation.text_robustness import apply_text_perturbation


@pytest.fixture
def sample_text():
    return "The government announced new education policies today"


@pytest.mark.parametrize(
    "perturbation",
    ["word_deletion", "word_swap", "character_noise"]
)
def test_perturbation_returns_string(sample_text, perturbation):
    result = apply_text_perturbation(
        sample_text, perturbation, severity=2
    )

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
def test_all_severity_levels(sample_text, severity):
    result = apply_text_perturbation(
        sample_text, "character_noise", severity=severity
    )

    assert isinstance(result, str)


def test_word_deletion_changes_text(sample_text):
    result = apply_text_perturbation(
        sample_text, "word_deletion", severity=3
    )

    assert len(result.split()) < len(sample_text.split())


def test_character_noise_changes_text(sample_text):
    result = apply_text_perturbation(
        sample_text, "character_noise", severity=3
    )

    assert result != sample_text


def test_same_seed_gives_same_result(sample_text):
    first = apply_text_perturbation(
        sample_text, "character_noise", severity=3, seed=42
    )
    second = apply_text_perturbation(
        sample_text, "character_noise", severity=3, seed=42
    )

    assert first == second


def test_invalid_perturbation(sample_text):
    with pytest.raises(ValueError):
        apply_text_perturbation(
            sample_text, "invalid_method", severity=2
        )


def test_invalid_severity(sample_text):
    with pytest.raises(ValueError):
        apply_text_perturbation(
            sample_text, "word_deletion", severity=6
        )


def test_invalid_text_type():
    with pytest.raises(TypeError):
        apply_text_perturbation(123, "word_deletion")


def test_empty_text():
    result = apply_text_perturbation(
        "", "word_deletion", severity=2
    )

    assert result == ""
