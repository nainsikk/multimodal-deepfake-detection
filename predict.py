
import os
import json
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

CONFIG = {
    "text_checkpoint":
        "results/fusion_crossattn/late_fusion/text/seed_7/model.pt",

    "image_checkpoint":
        "results/fusion_crossattn/late_fusion/image/seed_7/model.pt",

    "calibration":
        "results/calibration/late_fusion/calibration_config.json",

    "mismatch_checkpoint":
        "results/mismatch/mismatch_model.pt",

    "bert": "distilbert-base-uncased",
    "clip": "openai/clip-vit-base-patch32",

    "max_length": 64,

    "text_weight": 0.45,
    "image_weight": 0.55,
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. LATE FUSION MLP
# ============================================================

class LateFusionMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. FEATURE EXTRACTION
# ============================================================


class FeatureExtractor:

    def __init__(self):

        # Use the same device selection as Part 5
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print("Using device:", self.device)

        # Load DistilBERT
        print("Loading DistilBERT...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["bert"]
        )

        self.bert = AutoModel.from_pretrained(
            CONFIG["bert"]
        ).to(self.device)

        # Load CLIP
        print("Loading CLIP...")

        self.clip_processor = CLIPProcessor.from_pretrained(
            CONFIG["clip"]
        )

        self.clip = CLIPModel.from_pretrained(
            CONFIG["clip"]
        ).to(self.device)

        self.bert.eval()
        self.clip.eval()

    @torch.no_grad()
    def extract(self, text, image_path):

        # --------------------------------
        # 1. DistilBERT text features
        # --------------------------------

        bert_inputs = self.tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_length"],
            return_tensors="pt"
        ).to(self.device)

        bert_outputs = self.bert(**bert_inputs)

        hidden_states = bert_outputs.last_hidden_state

        mask = (
            bert_inputs["attention_mask"]
            .unsqueeze(-1)
            .float()
        )

        # Same masked mean pooling as Part 5
        bert_text = (
            (hidden_states * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1e-9)
        )

        # --------------------------------
        # 2. CLIP text features
        # --------------------------------

        clip_text_inputs = self.clip_processor.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.device)

        clip_text_outputs = self.clip.text_model(
            **clip_text_inputs
        )

        clip_text = self.clip.text_projection(
            clip_text_outputs.pooler_output
        )

        # Same normalization as Part 5
        clip_text = nn.functional.normalize(
            clip_text, dim=-1
        )

        # --------------------------------
        # 3. CLIP image features
        # --------------------------------

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            image_inputs = self.clip_processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)

        clip_vision_outputs = self.clip.vision_model(
            pixel_values=image_inputs["pixel_values"]
        )

        clip_image = self.clip.visual_projection(
            clip_vision_outputs.pooler_output
        )

        # Same normalization as Part 5
        clip_image = nn.functional.normalize(
            clip_image, dim=-1
        )

        # --------------------------------
        # 4. CLIP similarity
        # --------------------------------

        clip_similarity = (
            clip_text * clip_image
        ).sum(dim=-1)

        return {
            "text": bert_text,
            "image": clip_image,
            "clip_text": clip_text,
            "clip_similarity": clip_similarity
        }

# ============================================================
# 4. LOAD TRAINED MODELS
# ============================================================

class SingleModel:

    def __init__(self, checkpoint_path, input_dim):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False
        )

        if checkpoint["input_dim"] != input_dim:
            raise ValueError(
                f"Expected {input_dim} features, "
                f"got {checkpoint['input_dim']}"
            )

        self.mean = checkpoint["mean"].to(DEVICE)
        self.std = checkpoint["std"].to(DEVICE)

        self.model = LateFusionMLP(input_dim).to(DEVICE)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

    @torch.no_grad()
    def predict(self, features):

        # Apply training-set normalization
        features = (
            features - self.mean
        ) / self.std.clamp(min=1e-8)

        logits = self.model(features)

        # Column 0 = FAKE, column 1 = REAL
        probabilities = torch.softmax(logits, dim=1)

        return probabilities[0].cpu().numpy()


# ============================================================
# 5. MAIN PREDICTOR
# ============================================================

class Predictor:

    def __init__(self):

        print("\nInitializing prediction pipeline...")

        self.extractor = FeatureExtractor()

        self.text_model = SingleModel(
            CONFIG["text_checkpoint"],
            768
        )

        self.image_model = SingleModel(
            CONFIG["image_checkpoint"],
            512
        )

        with open(CONFIG["calibration"], "r") as file:
            self.calibration = json.load(file)

        self.temperature = float(
            self.calibration["temperature"]
        )

        self.tau = float(
            self.calibration["tau"]
        )

        print("Text model loaded.")
        print("Image model loaded.")
        print("Calibration loaded.")

        print("\nPrediction pipeline initialized.")

    def predict(self, text, image_path):

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please provide valid text.")

        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Extract text and image features
        text_features, image_features = (
            self.extractor.extract(text, image_path)
        )

        # Get individual model probabilities
        text_probs = self.text_model.predict(
            text_features
        )

        image_probs = self.image_model.predict(
            image_features
        )

        # Late fusion using the selected training weight
        fused_probs = (
            CONFIG["text_weight"] * text_probs
            + CONFIG["image_weight"] * image_probs
        )

        # Temperature calibration
        fused_probs = np.clip(
            fused_probs, 1e-7, 1.0
        )

        calibrated_probs = np.exp(
            np.log(fused_probs) / self.temperature
        )

        calibrated_probs /= calibrated_probs.sum()

        # Class mapping: 0 = FAKE, 1 = REAL
        predicted_class = int(
            np.argmax(calibrated_probs)
        )

        confidence = float(
            calibrated_probs[predicted_class]
        )

        if confidence < self.tau:
            prediction = "UNCERTAIN"
        elif predicted_class == 1:
            prediction = "REAL"
        else:
            prediction = "FAKE"

        return {
            "prediction": prediction,

            "probability_fake": round(
                float(calibrated_probs[0]), 4
            ),

            "probability_real": round(
                float(calibrated_probs[1]), 4
            ),

            "confidence": round(confidence, 4),

            "text_probability_fake": round(
                float(text_probs[0]), 4
            ),

            "text_probability_real": round(
                float(text_probs[1]), 4
            ),

            "image_probability_fake": round(
                float(image_probs[0]), 4
            ),

            "image_probability_real": round(
                float(image_probs[1]), 4
            ),

            "fusion_weights": {
                "text": 0.45,
                "image": 0.55
            },

            "model": "Late Fusion",
            "selected_seed": 7
        }


# ============================================================
# 6. DEMO
# ============================================================

if __name__ == "__main__":

    predictor = Predictor()

    text = "glenn beck goonswarm a cia front eve online news mmorpgcom"

    image_path = "data/test/images/10cpga.jpg"

    result = predictor.predict(
        text,
        image_path
    )

    print("\nPREDICTION RESULT")
    print("=" * 45)

    print(json.dumps(result, indent=4))
