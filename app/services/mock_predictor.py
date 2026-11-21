
import copy
import random


class MockPredictor:

    def __init__(self):
        self.loaded = True

        self.examples = [
            {
                "prediction": "REAL",
                "confidence": 0.94,
                "manipulation_type": "none",
                "modality_scores": {
                    "text": 0.93,
                    "image": 0.95,
                    "fusion": 0.94
                },
                "evidence": {
                    "text_tokens": [],
                    "image_heatmap_url": None,
                    "mismatch_score": 8
                },
                "explanation":
                    "The caption and image appear consistent."
            },
            {
                "prediction": "FAKE",
                "confidence": 0.91,
                "manipulation_type": "caption-image mismatch",
                "modality_scores": {
                    "text": 0.89,
                    "image": 0.82,
                    "fusion": 0.91
                },
                "evidence": {
                    "text_tokens": [
                        {"token": "breaking", "weight": 0.31},
                        {"token": "flood", "weight": 0.42}
                    ],
                    "image_heatmap_url": None,
                    "mismatch_score": 87
                },
                "explanation":
                    "The caption describes a flood, but the image "
                    "shows a dry street."
            },
            {
                "prediction": "UNCERTAIN",
                "confidence": 0.56,
                "manipulation_type": "uncertain evidence",
                "modality_scores": {
                    "text": 0.59,
                    "image": 0.62,
                    "fusion": 0.56
                },
                "evidence": {
                    "text_tokens": [
                        {"token": "reportedly", "weight": 0.18}
                    ],
                    "image_heatmap_url": None,
                    "mismatch_score": 49
                },
                "explanation":
                    "The available text and image evidence is "
                    "insufficient for a reliable decision."
            }
        ]

    def predict(
        self,
        text: str | None = None,
        image_path: str | None = None
    ) -> dict:

        if not text and not image_path:
            raise ValueError("At least one input is required.")

        result = copy.deepcopy(random.choice(self.examples))

        # Text-only prediction
        if text and not image_path:
            result["modality_scores"]["image"] = None
            result["modality_scores"]["fusion"] = None
            result["evidence"]["mismatch_score"] = None
            result["manipulation_type"] = "text-only analysis"
            result["explanation"] = (
                "Mock prediction generated using text only. "
                "This is not an actual model prediction."
            )

        # Image-only prediction
        elif image_path and not text:
            result["modality_scores"]["text"] = None
            result["modality_scores"]["fusion"] = None
            result["evidence"]["text_tokens"] = []
            result["evidence"]["mismatch_score"] = None
            result["manipulation_type"] = "image-only analysis"
            result["explanation"] = (
                "Mock prediction generated using image only. "
                "This is not an actual model prediction."
            )

        return result
