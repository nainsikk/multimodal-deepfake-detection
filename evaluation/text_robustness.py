
"""Text perturbation utilities for robustness evaluation."""

import random
import string


def apply_text_perturbation(text, perturbation, severity=1, seed=42):
    """
    Apply controlled changes to input text.

    Args:
        text: Input sentence or paragraph.
        perturbation: 'word_deletion', 'word_swap', or 'character_noise'.
        severity: Integer from 1 (mild) to 5 (strong).
        seed: Random seed for reproducibility.

    Returns:
        Perturbed text as a string.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if perturbation not in {
        "word_deletion",
        "word_swap",
        "character_noise"
    }:
        raise ValueError(
            "perturbation must be 'word_deletion', "
            "'word_swap', or 'character_noise'."
        )

    if isinstance(severity, bool) or not isinstance(severity, int):
        raise TypeError("severity must be an integer from 1 to 5.")

    if not 1 <= severity <= 5:
        raise ValueError("severity must be between 1 and 5.")

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer.")

    words = text.split()

    if not words:
        return text

    rng = random.Random(seed)
    ratio = severity * 0.10

    # 1. Randomly delete words while keeping at least one word.
    if perturbation == "word_deletion":
        max_deletions = min(
            len(words) - 1,
            max(1, round(len(words) * ratio))
        )

        if max_deletions <= 0:
            return text

        indices = sorted(
            rng.sample(range(len(words)), max_deletions)
        )

        return " ".join(
            word for index, word in enumerate(words)
            if index not in indices
        )

    # 2. Swap randomly selected adjacent words.
    if perturbation == "word_swap":
        if len(words) < 2:
            return text

        max_swaps = max(1, round(len(words) * ratio))
        max_swaps = min(max_swaps, len(words) // 2)

        for _ in range(max_swaps):
            index = rng.randrange(len(words) - 1)
            words[index], words[index + 1] = (
                words[index + 1],
                words[index]
            )

        return " ".join(words)

    # 3. Replace randomly selected alphabetic characters.
    if perturbation == "character_noise":
        characters = list(text)

        eligible = [
            index for index, char in enumerate(characters)
            if char.isalpha()
        ]

        if not eligible:
            return text

        max_changes = max(1, round(len(eligible) * ratio))
        max_changes = min(max_changes, len(eligible))

        selected = rng.sample(eligible, max_changes)

        for index in selected:
            original = characters[index]

            alphabet = (
                string.ascii_uppercase
                if original.isupper()
                else string.ascii_lowercase
            )

            alternatives = [
                char for char in alphabet if char != original
            ]

            characters[index] = rng.choice(alternatives)

        return "".join(characters)
