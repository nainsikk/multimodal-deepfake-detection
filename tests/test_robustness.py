
import pytest
import numpy as np
from PIL import Image

from evaluation.robustness import apply_image_corruption


@pytest.fixture
def sample_image():
    return Image.new("RGB", (100, 100), color=(100, 150, 200))


@pytest.mark.parametrize(
    "corruption",
    ["blur", "brightness", "gaussian_noise"]
)
def test_corruption_returns_valid_image(sample_image, corruption):
    result = apply_image_corruption(sample_image, corruption, severity=2)

    assert isinstance(result, Image.Image)
    assert result.size == sample_image.size
    assert result.mode == "RGB"


@pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
def test_all_severity_levels(sample_image, severity):
    result = apply_image_corruption(
        sample_image, "blur", severity=severity
    )

    assert isinstance(result, Image.Image)


def test_invalid_corruption_name(sample_image):
    with pytest.raises(ValueError):
        apply_image_corruption(sample_image, "rotation", severity=2)


@pytest.mark.parametrize("severity", [0, 6, -1])
def test_invalid_severity_value(sample_image, severity):
    with pytest.raises(ValueError):
        apply_image_corruption(
            sample_image, "blur", severity=severity
        )


@pytest.mark.parametrize("severity", [1.5, "2", True])
def test_invalid_severity_type(sample_image, severity):
    with pytest.raises(TypeError):
        apply_image_corruption(
            sample_image, "blur", severity=severity
        )


def test_gaussian_noise_is_reproducible(sample_image):
    first = apply_image_corruption(
        sample_image, "gaussian_noise", severity=3, seed=42
    )
    second = apply_image_corruption(
        sample_image, "gaussian_noise", severity=3, seed=42
    )

    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_invalid_image_type():
    with pytest.raises(TypeError):
        apply_image_corruption("not an image", "blur", severity=2)
