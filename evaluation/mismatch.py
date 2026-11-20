
import math


def calculate_mismatch_score(
    text_fake_probability,
    image_fake_probability,
    threshold=0.5
):
    """
    Compare text and image model predictions.

    Labels:
        0 = Real
        1 = Fake

    Mismatch score:
        0   = Complete agreement
        100 = Maximum disagreement

    The score measures prediction disagreement,
    not whether the content is actually fake.
    """

    values = {
        "text_fake_probability": text_fake_probability,
        "image_fake_probability": image_fake_probability,
        "threshold": threshold
    }

    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                f"{name} must be a number between 0 and 1."
            )

    # Convert probabilities into Real/Fake predictions.
    text_prediction = (
        "Fake" if text_fake_probability >= threshold else "Real"
    )

    image_prediction = (
        "Fake" if image_fake_probability >= threshold else "Real"
    )

    # Calculate the difference between both probabilities.
    mismatch_score = abs(
        text_fake_probability - image_fake_probability
    ) * 100

    # Check whether both modalities disagree.
    disagreement = text_prediction != image_prediction

    # Describe the level of disagreement.
    if mismatch_score < 20:
        interpretation = "Low disagreement"
    elif mismatch_score < 50:
        interpretation = "Moderate disagreement"
    else:
        interpretation = "High disagreement"

    return {
        "text_prediction": text_prediction,
        "image_prediction": image_prediction,
        "mismatch_score": round(mismatch_score, 2),
        "disagreement": disagreement,
        "interpretation": interpretation
    }
