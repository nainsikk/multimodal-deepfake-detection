"""
Part 5: Frozen Feature Extraction for Multimodal Fusion

Extracts features from:
    - DistilBERT (text)
    - CLIP (text and images)

Run:
    python feature_extraction.py --data_dir data --out features

Labels:
    0 = FAKE
    1 = REAL

Features are extracted using frozen pretrained models.
No model training takes place in this script.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

from dataset_builder import SPLITS, load_split_df


# ----------------------------------------
# Select device (Apple MPS / CUDA / CPU)
# ----------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ----------------------------------------
# Feature extraction
# ----------------------------------------
@torch.no_grad()
def extract_split(
    df,
    bert_tokenizer,
    bert_model,
    clip_model,
    clip_processor,
    device,
    batch_size,
    max_len,
    save_tokens,
):

    features = {
        "bert_text": [],
        "clip_text": [],
        "clip_image": [],
        "clip_sim": [],
    }

    if save_tokens:
        features.update({
            "text_tokens": [],
            "text_mask": [],
            "img_patches": [],
        })

    total = len(df)

    # Process data in batches
    for start in range(0, total, batch_size):

        batch_df = df.iloc[start:start + batch_size]

        texts = batch_df["text"].astype(str).tolist()

        # ========================================
        # 1. DistilBERT text features
        # ========================================
        bert_inputs = bert_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        bert_outputs = bert_model(**bert_inputs)

        # Token-level representations: [batch, sequence, 768]
        hidden_states = bert_outputs.last_hidden_state

        # Attention mask: exclude padding tokens from mean pooling
        mask = bert_inputs["attention_mask"].unsqueeze(-1).float()

        # Mean pooling -> sentence representation [batch, 768]
        pooled_text = (
            (hidden_states * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1e-9)
        )

        features["bert_text"].append(pooled_text.cpu())

        if save_tokens:
            features["text_tokens"].append(
                hidden_states.half().cpu()
            )
            features["text_mask"].append(
                bert_inputs["attention_mask"].cpu()
            )

        # ========================================
        # 2. CLIP text features
        # ========================================
        clip_text_inputs = clip_processor.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        clip_text_outputs = clip_model.text_model(
            **clip_text_inputs
        )

        clip_text_embeddings = clip_model.text_projection(
            clip_text_outputs.pooler_output
        )

        # Normalize embeddings
        clip_text_embeddings = F.normalize(
            clip_text_embeddings, dim=-1
        )

        # ========================================
        # 3. Load and preprocess images for CLIP
        # ========================================
        images = []

        for image_path in batch_df["image_path"]:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))

        image_inputs = clip_processor.image_processor(
            images,
            return_tensors="pt",
        )

        pixel_values = image_inputs["pixel_values"].to(device)

        # ========================================
        # 4. CLIP image features
        # ========================================
        clip_vision_outputs = clip_model.vision_model(
            pixel_values=pixel_values
        )

        clip_image_embeddings = clip_model.visual_projection(
            clip_vision_outputs.pooler_output
        )

        clip_image_embeddings = F.normalize(
            clip_image_embeddings, dim=-1
        )

        # ========================================
        # 5. Image patch features for cross-attention
        # ========================================
        if save_tokens:
            # CLS token + 49 image patches = 50 tokens
            features["img_patches"].append(
                clip_vision_outputs.last_hidden_state.half().cpu()
            )

        # ========================================
        # 6. Text-image similarity
        # ========================================
        # Cosine similarity between normalized embeddings
        clip_similarity = (
            clip_text_embeddings * clip_image_embeddings
        ).sum(dim=-1)

        features["clip_text"].append(
            clip_text_embeddings.cpu()
        )

        features["clip_image"].append(
            clip_image_embeddings.cpu()
        )

        features["clip_sim"].append(
            clip_similarity.cpu()
        )

        # Progress
        processed = min(start + batch_size, total)

        print(
            f"  Processed {processed}/{total} samples",
            end="\r",
            flush=True,
        )

    print()

    # Concatenate all batches
    features = {
        key: torch.cat(value, dim=0)
        for key, value in features.items()
    }

    return features


# ----------------------------------------
# Main
# ----------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Extract frozen BERT and CLIP features"
    )

    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out", default="features")

    parser.add_argument(
        "--bert",
        default="distilbert-base-uncased",
    )

    parser.add_argument(
        "--clip",
        default="openai/clip-vit-base-patch32",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=64)

    parser.add_argument(
        "--no_tokens",
        action="store_true",
        help="Skip token/patch features to save disk space",
    )

    args = parser.parse_args()

    # ----------------------------------------
    # Device
    # ----------------------------------------
    device = get_device()
    print("Device:", device)

    os.makedirs(args.out, exist_ok=True)

    # ----------------------------------------
    # Load pretrained models
    # ----------------------------------------
    print("\nLoading DistilBERT...")

    bert_tokenizer = AutoTokenizer.from_pretrained(
        args.bert
    )

    bert_model = AutoModel.from_pretrained(
        args.bert
    ).to(device)

    bert_model.eval()

    print("Loading CLIP...")

    clip_model = CLIPModel.from_pretrained(
        args.clip
    ).to(device)

    clip_model.eval()

    clip_processor = CLIPProcessor.from_pretrained(
        args.clip
    )

    print("Models loaded successfully.")

    # ----------------------------------------
    # Extract features for all official splits
    # ----------------------------------------
    for split in SPLITS:

        df = load_split_df(args.data_dir, split)

        print(f"\n[{split}] {len(df)} samples")

        features = extract_split(
            df=df,
            bert_tokenizer=bert_tokenizer,
            bert_model=bert_model,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
            batch_size=args.batch_size,
            max_len=args.max_len,
            save_tokens=not args.no_tokens,
        )

        # Store IDs and labels
        features["ids"] = df["id"].astype(str).tolist()

        features["label"] = torch.tensor(
            df["label"].values,
            dtype=torch.long,
        )

        features["label6"] = torch.tensor(
            df["label6"].values,
            dtype=torch.long,
        )

        # ----------------------------------------
        # Save features
        # ----------------------------------------
        output_path = os.path.join(
            args.out,
            f"{split}.pt",
        )

        torch.save(features, output_path)

        # ----------------------------------------
        # CLIP similarity analysis
        # ----------------------------------------
        similarity = features["clip_sim"]
        labels = features["label"]

        fake_similarity = similarity[labels == 0].mean().item()
        real_similarity = similarity[labels == 1].mean().item()
        overall_similarity = similarity.mean().item()

        print(f"Saved: {output_path}")

        print(
            f"CLIP similarity mean: {overall_similarity:.4f} | "
            f"FAKE: {fake_similarity:.4f} | "
            f"REAL: {real_similarity:.4f}"
        )

        # Feature shapes
        print("Feature shapes:")

        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {tuple(value.shape)}")

    print("\nFeature extraction completed successfully!")


if __name__ == "__main__":
    main()