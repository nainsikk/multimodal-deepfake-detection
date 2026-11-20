
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
                    "mismatch_score": 0.08
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
                    "mismatch_score": 0.87
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
                    "mismatch_score": 0.49
                },

                "explanation":
                    "The available text and image evidence is "
                    "insufficient for a reliable decision."
            }
        ]

    def predict(self, text: str, image_path: str) -> dict:

        return random.choice(self.examples).copy()
