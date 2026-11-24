
"""Image corruption utilities for robustness evaluation."""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def apply_image_corruption(
    image,
    corruption,
    severity=1,
    seed=42
):
    """
    Apply a controlled image corruption.

    Args:
        image: PIL image.
        corruption: 'blur', 'brightness', or 'gaussian_noise'.
        severity: Integer from 1 (mild) to 5 (strong).
        seed: Random seed for reproducible noise.

    Returns:
        A corrupted RGB PIL image.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.")

    if corruption not in {"blur", "brightness", "gaussian_noise"}:
        raise ValueError(
            "corruption must be 'blur', 'brightness', "
            "or 'gaussian_noise'."
        )

    if isinstance(severity, bool) or not isinstance(severity, int):
        raise TypeError("severity must be an integer from 1 to 5.")

    if not 1 <= severity <= 5:
        raise ValueError("severity must be between 1 and 5.")

    image = image.convert("RGB")

    if corruption == "blur":
        radius = severity * 0.8
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    if corruption == "brightness":
        factor = 1.0 - (severity * 0.15)
        return ImageEnhance.Brightness(image).enhance(factor)

    if corruption == "gaussian_noise":
        rng = np.random.default_rng(seed)
        pixels = np.asarray(image).astype(np.float32)

        noise = rng.normal(
            loc=0.0,
            scale=severity * 8,
            size=pixels.shape
        )

        noisy_pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_pixels, mode="RGB")
