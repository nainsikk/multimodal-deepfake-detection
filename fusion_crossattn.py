# ============================================================
# PART 7: CROSS-ATTENTION & LATE FUSION
# Project: Multimodal NLP for Deepfake Detection
#
# Experiments:
#   1. Cross-Attention
#   2. Cross-Attention + CLIP Similarity
#   3. Late Fusion (Text model + Image model)
#
# Labels:
#   0 = FAKE
#   1 = REAL
#
# Uses cached Part 5 features.
# Device: Apple Silicon MPS / CUDA / CPU
# ============================================================

import os
import json
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

FEATURE_DIR = "features"
OUTPUT_DIR = "results/fusion_crossattn"

SEEDS = [42, 123, 2024, 7, 99]

BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Cross-attention model settings
D_MODEL = 256
NUM_HEADS = 4
DROPOUT = 0.2

# Late fusion weight search
WEIGHT_GRID = np.arange(0.0, 1.01, 0.05)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 60)
print("PART 7: CROSS-ATTENTION & LATE FUSION")
print("=" * 60)
print("Device:", DEVICE)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 4. LOAD CACHED FEATURES
# ============================================================

def load_features(split):
    """
    Load cached features produced by Part 5.

    Each file contains:
        bert_text
        clip_text
        clip_image
        clip_sim
        text_tokens
        text_mask
        img_patches
        ids
        label
        label6
    """

    path = os.path.join(FEATURE_DIR, f"{split}.pt")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            "Run Part 5 feature extraction first."
        )

    # map_location ensures tensors load safely onto CPU first.
    data = torch.load(path, map_location="cpu", weights_only=False)

    print(f"\nLoaded {split}: {path}")

    for key in ["bert_text", "clip_image", "text_tokens", "text_mask", "img_patches", "label"]:
        if key in data:
            print(f"  {key}: {tuple(data[key].shape)}")

    return data


train_data = load_features("train")
val_data = load_features("validate")
test_data = load_features("test")


# ============================================================
# 5. PREPARE TENSORS
# ============================================================

def get_labels(data):
    """Extract labels as integer tensor."""

    labels = data["label"].long().reshape(-1)
    return labels


y_train = get_labels(train_data)
y_val = get_labels(val_data)
y_test = get_labels(test_data)

print("\nClass mapping: 0 = FAKE, 1 = REAL")
print("Train:", len(y_train))
print("Validation:", len(y_val))
print("Test:", len(y_test))


# ============================================================
# 6. METRICS
# ============================================================

def compute_metrics(y_true, probabilities):
    """
    probabilities[:, 0] = P(FAKE)
    probabilities[:, 1] = P(REAL)
    """

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)

    predictions = probabilities.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
    }

    try:
        # Positive class is REAL (label 1).
        metrics["roc_auc"] = roc_auc_score(
            y_true, probabilities[:, 1]
        )
    except ValueError:
        metrics["roc_auc"] = 0.0

    return metrics


def print_metrics(title, metrics):
    """Print metrics in readable format."""

    print(f"\n{title}")
    print("-" * 40)

    for name, value in metrics.items():
        print(f"{name:12s}: {value:.4f}")


def save_confusion_matrix(y_true, probabilities, name, folder):
    """Save a test confusion matrix."""

    predictions = np.asarray(probabilities).argmax(axis=1)

    cm = confusion_matrix(
        np.asarray(y_true),
        predictions,
        labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)

    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks([0, 1], ["FAKE", "REAL"])
    plt.yticks([0, 1], ["FAKE", "REAL"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center"
            )

    plt.colorbar()
    plt.tight_layout()

    path = os.path.join(folder, "confusion_matrix.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print("Confusion matrix saved:", path)


# ============================================================
# 7. CROSS-ATTENTION MODEL
# ============================================================

class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention.

    Text tokens attend to image patches:
        Text -> Image

    Image patches attend to text tokens:
        Image -> Text

    The resulting representations are pooled and concatenated
    for binary classification.

    Optional CLIP similarity is included in Experiment 2.
    """

    def __init__(
        self,
        input_dim=768,
        d_model=256,
        num_heads=4,
        dropout=0.2,
        use_similarity=False
    ):
        super().__init__()

        self.use_similarity = use_similarity

        # Project text and image features into the same dimension.
        self.text_projection = nn.Linear(input_dim, d_model)
        self.image_projection = nn.Linear(input_dim, d_model)

        # Cross-attention: text queries image patches.
        self.text_to_image = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention: image queries text tokens.
        self.image_to_text = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Residual connections + normalization.
        self.text_norm = nn.LayerNorm(d_model)
        self.image_norm = nn.LayerNorm(d_model)

        # Input to classifier:
        # pooled text + pooled image = 512
        # optional similarity = 1 additional feature
        classifier_input = d_model * 2

        if use_similarity:
            classifier_input += 1

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, text_tokens, text_mask, image_patches, clip_sim=None):
        """
        text_tokens:   [B, L, 768]
        text_mask:     [B, L] (1=valid, 0=padding)
        image_patches: [B, P, 768]
        clip_sim:      [B] (optional)
        """

        # Project features to common attention dimension.
        text = self.text_projection(text_tokens)
        image = self.image_projection(image_patches)

        # PyTorch key padding mask:
        # True means ignore that token.
        text_padding_mask = (text_mask == 0)

        # Text queries attend to image patches.
        text_attended, _ = self.text_to_image(
            query=text,
            key=image,
            value=image,
            need_weights=False
        )

        # Image queries attend to valid text tokens.
        image_attended, _ = self.image_to_text(
            query=image,
            key=text,
            value=text,
            key_padding_mask=text_padding_mask,
            need_weights=False
        )

        # Residual connections and normalization.
        text = self.text_norm(text + text_attended)
        image = self.image_norm(image + image_attended)

        # Masked average pooling for text.
        text_mask_float = text_mask.unsqueeze(-1).float()

        text_sum = (text * text_mask_float).sum(dim=1)
        text_count = text_mask_float.sum(dim=1).clamp(min=1.0)

        pooled_text = text_sum / text_count

        # Mean pooling over image patches.
        pooled_image = image.mean(dim=1)

        # Combine modalities.
        combined = torch.cat(
            [pooled_text, pooled_image],
            dim=1
        )

        # Add similarity for Experiment 2.
        if self.use_similarity:
            if clip_sim is None:
                raise ValueError("CLIP similarity required.")

            clip_sim = clip_sim.reshape(-1, 1).float()
            combined = torch.cat([combined, clip_sim], dim=1)

        logits = self.classifier(combined)

        return logits


# ============================================================
# 8. CROSS-ATTENTION DATASET
# ============================================================

def make_cross_attention_loader(data, shuffle=False):
    """Create a loader for cross-attention experiments."""

    dataset = TensorDataset(
        data["text_tokens"].float(),
        data["text_mask"].long(),
        data["img_patches"].float(),
        data["clip_sim"].float().reshape(-1),
        data["label"].long().reshape(-1)
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )


train_cross_loader = make_cross_attention_loader(train_data, shuffle=True)
train_eval_loader = make_cross_attention_loader(train_data, shuffle=False)
val_cross_loader = make_cross_attention_loader(val_data, shuffle=False)
test_cross_loader = make_cross_attention_loader(test_data, shuffle=False)


# ============================================================
# 9. CROSS-ATTENTION TRAINING & EVALUATION
# ============================================================

def evaluate_cross_attention(model, loader, use_similarity):
    """Evaluate cross-attention model and return probabilities."""

    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for text_tokens, text_mask, image_patches, clip_sim, labels in loader:

            text_tokens = text_tokens.to(DEVICE)
            text_mask = text_mask.to(DEVICE)
            image_patches = image_patches.to(DEVICE)
            clip_sim = clip_sim.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(
                text_tokens,
                text_mask,
                image_patches,
                clip_sim if use_similarity else None
            )

            probs = torch.softmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.asarray(all_labels), np.asarray(all_probs)


def train_cross_attention(use_similarity, experiment_name):
    """
    Train one cross-attention experiment across five seeds.

    Best checkpoint is selected by validation Macro-F1.
    The test set is used only for final evaluation.
    """

    experiment_dir = os.path.join(OUTPUT_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    seed_results = []
    best_overall_val_f1 = -1.0

    print("\n" + "=" * 60)
    print("EXPERIMENT:", experiment_name)
    print("=" * 60)

    for seed in SEEDS:

        set_seed(seed)

        model = CrossAttentionFusion(
            input_dim=768,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            use_similarity=use_similarity
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        criterion = nn.CrossEntropyLoss()

        best_val_f1 = -1.0
        best_epoch = 0
        best_state = None
        patience_counter = 0

        print(f"\nSeed: {seed}")

        for epoch in range(1, EPOCHS + 1):

            model.train()
            running_loss = 0.0

            for text_tokens, text_mask, image_patches, clip_sim, labels in train_cross_loader:

                text_tokens = text_tokens.to(DEVICE)
                text_mask = text_mask.to(DEVICE)
                image_patches = image_patches.to(DEVICE)
                clip_sim = clip_sim.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                logits = model(
                    text_tokens,
                    text_mask,
                    image_patches,
                    clip_sim if use_similarity else None
                )

                loss = criterion(logits, labels)
                loss.backward()

                # Prevent excessively large gradients.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                optimizer.step()

                running_loss += loss.item()

            # Validation after each epoch.
            val_labels, val_probs = evaluate_cross_attention(
                model, val_cross_loader, use_similarity
            )

            val_metrics = compute_metrics(val_labels, val_probs)
            val_f1 = val_metrics["macro_f1"]

            avg_loss = running_loss / len(train_cross_loader)

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val Macro-F1: {val_f1:.4f}"
            )

            # Save the best validation checkpoint.
            if val_f1 > best_val_f1:

                best_val_f1 = val_f1
                best_epoch = epoch

                best_state = copy.deepcopy(
                    model.state_dict()
                )

                patience_counter = 0

            else:
                patience_counter += 1

            # Early stopping.
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

        # Restore best validation model.
        model.load_state_dict(best_state)

        # Save checkpoint for this seed.
        seed_dir = os.path.join(
            experiment_dir, f"seed_{seed}"
        )
        os.makedirs(seed_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            seed_dir, "model.pt"
        )

        torch.save(
            {
                "model_state_dict": best_state,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_f1,
                "use_similarity": use_similarity,
            },
            checkpoint_path
        )

        # Save validation predictions.
        val_labels, val_probs = evaluate_cross_attention(
            model, val_cross_loader, use_similarity
        )

        np.savez(
            os.path.join(seed_dir, "val_predictions.npz"),
            labels=val_labels,
            probs=val_probs,
            preds=val_probs.argmax(axis=1)
        )

        # Final test evaluation.
        test_labels, test_probs = evaluate_cross_attention(
            model, test_cross_loader, use_similarity
        )

        test_metrics = compute_metrics(
            test_labels, test_probs
        )

        print_metrics(
            f"Seed {seed} Test Metrics",
            test_metrics
        )

        np.savez(
            os.path.join(seed_dir, "test_predictions.npz"),
            labels=test_labels,
            probs=test_probs,
            preds=test_probs.argmax(axis=1)
        )

        seed_results.append({
            "experiment": experiment_name,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            **test_metrics
        })

        # Keep the overall best validation checkpoint.
        if best_val_f1 > best_overall_val_f1:

            best_overall_val_f1 = best_val_f1

            torch.save(
                {
                    "model_state_dict": best_state,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                    "use_similarity": use_similarity,
                },
                os.path.join(experiment_dir, "best_model.pt")
            )

            # Save overall-best test predictions.
            np.savez(
                os.path.join(experiment_dir, "best_test_predictions.npz"),
                labels=test_labels,
                probs=test_probs,
                preds=test_probs.argmax(axis=1)
            )

            save_confusion_matrix(
                test_labels,
                test_probs,
                f"{experiment_name} - Best Seed {seed}",
                experiment_dir
            )

        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    # Calculate mean and standard deviation across seeds.
    df = pd.DataFrame(seed_results)

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "roc_auc"
    ]

    summary = {}

    for metric in metric_columns:
        summary[f"{metric}_mean"] = float(df[metric].mean())
        summary[f"{metric}_std"] = float(df[metric].std(ddof=1))

    summary["best_validation_macro_f1"] = float(
        best_overall_val_f1
    )

    # Save seed results and summary.
    df.to_csv(
        os.path.join(experiment_dir, "seed_results.csv"),
        index=False
    )

    with open(
        os.path.join(experiment_dir, "summary.json"),
        "w"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\nFINAL SUMMARY:", experiment_name)

    for metric in metric_columns:
        print(
            f"{metric:12s}: "
            f"{summary[f'{metric}_mean']:.4f} "
            f"+/- {summary[f'{metric}_std']:.4f}"
        )

    return seed_results, summary


# ============================================================
# 10. LATE FUSION MLP
# ============================================================

class LateFusionMLP(nn.Module):
    """
    Simple MLP used for text-only or image-only feature training.

    This is similar to the Part 6 single-modality baselines.
    """

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


def extract_late_features(data, modality):
    """Extract BERT text or CLIP image pooled features."""

    if modality == "text":
        return data["bert_text"].float()

    elif modality == "image":
        return data["clip_image"].float()

    raise ValueError("Unknown modality")


# Train and evaluate each single modality.
def train_late_model(modality, seed):
    """
    Train text-only or image-only MLP.

    Normalization statistics are calculated only from training data.
    """

    set_seed(seed)

    X_train = extract_late_features(train_data, modality)
    X_val = extract_late_features(val_data, modality)
    X_test = extract_late_features(test_data, modality)

    # Normalization uses training statistics only.
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp(min=1e-6)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = LateFusionMLP(
        input_dim=X_train.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()
        running_loss = 0.0

        for features, labels in train_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(features)
            loss = criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            running_loss += loss.item()

        # Validation.
        model.eval()

        with torch.no_grad():
            val_logits = model(X_val.to(DEVICE))
            val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()

        val_metrics = compute_metrics(
            y_val.numpy(), val_probs
        )

        val_f1 = val_metrics["macro_f1"]

        print(
            f"{modality} | Seed {seed} | "
            f"Epoch {epoch:02d} | "
            f"Loss {running_loss / len(train_loader):.4f} | "
            f"Val F1 {val_f1:.4f}"
        )

        if val_f1 > best_f1:

            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    # Restore best validation checkpoint.
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():

        val_probs = torch.softmax(
            model(X_val.to(DEVICE)), dim=1
        ).cpu().numpy()

        test_probs = torch.softmax(
            model(X_test.to(DEVICE)), dim=1
        ).cpu().numpy()

    # Save late-fusion model checkpoints.
    modality_dir = os.path.join(
        OUTPUT_DIR, "late_fusion", modality, f"seed_{seed}"
    )

    os.makedirs(modality_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "mean": mean,
            "std": std,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1,
            "input_dim": X_train.shape[1],
        },
        os.path.join(modality_dir, "model.pt")
    )

    return val_probs, test_probs, best_f1, best_epoch


# ============================================================
# 11. LATE FUSION
# ============================================================

def run_late_fusion():
    """
    Late fusion:

    1. Train separate text-only and image-only MLPs.
    2. Obtain their P(FAKE) validation and test probabilities.
    3. Tune the text weight using validation only.
    4. Apply the selected weight to test predictions.
    """

    late_dir = os.path.join(
        OUTPUT_DIR, "late_fusion"
    )

    os.makedirs(late_dir, exist_ok=True)

    all_seed_results = []

    best_seed_test_probs = None
    best_seed_test_labels = None
    best_seed_val_f1 = -1.0

    print("\n" + "=" * 60)
    print("EXPERIMENT: LATE FUSION")
    print("=" * 60)

    for seed in SEEDS:

        print(f"\nLate fusion seed: {seed}")

        # Train independent text and image models.
        text_val_probs, text_test_probs, text_f1, _ = train_late_model(
            "text", seed
        )

        image_val_probs, image_test_probs, image_f1, _ = train_late_model(
            "image", seed
        )

        # Column 0 represents P(FAKE).
        text_val_fake = text_val_probs[:, 0]
        image_val_fake = image_val_probs[:, 0]

        text_test_fake = text_test_probs[:, 0]
        image_test_fake = image_test_probs[:, 0]

        # Search the fusion weight on validation data only.
        best_weight = 0.5
        best_val_f1 = -1.0
        best_val_probs = None

        for weight in WEIGHT_GRID:

            fused_val_fake = (
                weight * text_val_fake
                + (1.0 - weight) * image_val_fake
            )

            # Convert P(FAKE) into two-class probabilities.
            fused_val_probs = np.column_stack([
                fused_val_fake,
                1.0 - fused_val_fake
            ])

            metrics = compute_metrics(
                y_val.numpy(),
                fused_val_probs
            )

            if metrics["macro_f1"] > best_val_f1:

                best_val_f1 = metrics["macro_f1"]
                best_weight = float(weight)
                best_val_probs = fused_val_probs

        # Apply the selected weight to test probabilities.
        fused_test_fake = (
            best_weight * text_test_fake
            + (1.0 - best_weight) * image_test_fake
        )

        fused_test_probs = np.column_stack([
            fused_test_fake,
            1.0 - fused_test_fake
        ])

        test_metrics = compute_metrics(
            y_test.numpy(),
            fused_test_probs
        )

        print(f"Selected text weight: {best_weight:.2f}")
        print(f"Validation Macro-F1: {best_val_f1:.4f}")

        print_metrics(
            f"Late Fusion Seed {seed} Test",
            test_metrics
        )

        # Save predictions for this seed.
        seed_dir = os.path.join(
            late_dir, f"seed_{seed}"
        )

        os.makedirs(seed_dir, exist_ok=True)

        np.savez(
            os.path.join(seed_dir, "val_predictions.npz"),
            labels=y_val.numpy(),
            probs=best_val_probs,
            preds=best_val_probs.argmax(axis=1),
            text_weight=best_weight
        )

        np.savez(
            os.path.join(seed_dir, "test_predictions.npz"),
            labels=y_test.numpy(),
            probs=fused_test_probs,
            preds=fused_test_probs.argmax(axis=1),
            text_weight=best_weight
        )

        all_seed_results.append({
            "experiment": "late_fusion",
            "seed": seed,
            "text_weight": best_weight,
            "best_val_macro_f1": best_val_f1,
            **test_metrics
        })

        # Store the best validation seed.
        if best_val_f1 > best_seed_val_f1:

            best_seed_val_f1 = best_val_f1
            best_seed_test_probs = fused_test_probs
            best_seed_test_labels = y_test.numpy()

    # Save confusion matrix for best validation-selected seed.
    save_confusion_matrix(
        best_seed_test_labels,
        best_seed_test_probs,
        "Late Fusion - Best Validation Seed",
        late_dir
    )

    # Summary across seeds.
    df = pd.DataFrame(all_seed_results)

    df.to_csv(
        os.path.join(late_dir, "seed_results.csv"),
        index=False
    )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "roc_auc"
    ]

    summary = {}

    for metric in metric_columns:
        summary[f"{metric}_mean"] = float(df[metric].mean())
        summary[f"{metric}_std"] = float(df[metric].std(ddof=1))

    summary["best_validation_macro_f1"] = float(
        best_seed_val_f1
    )

    with open(
        os.path.join(late_dir, "summary.json"),
        "w"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\nLATE FUSION FINAL SUMMARY")

    for metric in metric_columns:
        print(
            f"{metric:12s}: "
            f"{summary[f'{metric}_mean']:.4f} "
            f"+/- {summary[f'{metric}_std']:.4f}"
        )

    return all_seed_results, summary


# ============================================================
# 12. RUN EXPERIMENTS
# ============================================================

if __name__ == "__main__":

    all_results = {}
    all_summaries = {}

    # --------------------------------------------------------
    # Experiment 1: Bidirectional Cross-Attention
    # --------------------------------------------------------

    results1, summary1 = train_cross_attention(
        use_similarity=False,
        experiment_name="crossattn"
    )

    all_results["crossattn"] = results1
    all_summaries["crossattn"] = summary1

    # --------------------------------------------------------
    # Experiment 2: Cross-Attention + CLIP Similarity
    # --------------------------------------------------------

    results2, summary2 = train_cross_attention(
        use_similarity=True,
        experiment_name="crossattn_sim"
    )

    all_results["crossattn_sim"] = results2
    all_summaries["crossattn_sim"] = summary2

    # --------------------------------------------------------
    # Experiment 3: Late Fusion
    # --------------------------------------------------------

    results3, summary3 = run_late_fusion()

    all_results["late_fusion"] = results3
    all_summaries["late_fusion"] = summary3

    # --------------------------------------------------------
    # Save overall summary
    # --------------------------------------------------------

    with open(
        os.path.join(OUTPUT_DIR, "summary.json"),
        "w"
    ) as f:
        json.dump(all_summaries, f, indent=4)

    # Build final comparison table.
    rows = []

    for experiment, summary in all_summaries.items():

        rows.append({
            "experiment": experiment,
            "accuracy": (
                f"{summary['accuracy_mean']:.4f} "
                f"+/- {summary['accuracy_std']:.4f}"
            ),
            "macro_f1": (
                f"{summary['macro_f1_mean']:.4f} "
                f"+/- {summary['macro_f1_std']:.4f}"
            ),
            "roc_auc": (
                f"{summary['roc_auc_mean']:.4f} "
                f"+/- {summary['roc_auc_std']:.4f}"
            )
        })

    final_table = pd.DataFrame(rows)

    final_table.to_csv(
        os.path.join(OUTPUT_DIR, "results_table.csv"),
        index=False
    )

    print("\n" + "=" * 60)
    print("PART 7 COMPLETE")
    print("=" * 60)

    print(final_table.to_string(index=False))

    print("\nResults saved inside:", OUTPUT_DIR)