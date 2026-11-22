
# fusion_concat.py
# Part 6: Feature Concatenation Fusion
# Compares text-only, image-only, and multimodal feature fusion.
# Uses cached features from Part 5; does NOT load BERT or CLIP again.

import os
import json
import csv
import random
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# 2. Device selection
# ---------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ---------------------------------------------------------
# 3. Load cached features from Part 5
# ---------------------------------------------------------

def load_features(path):
    # Load the saved feature dictionary.
    data = torch.load(path, map_location="cpu", weights_only=False)

    print(f"\nLoaded: {path}")
    print("Available keys:", list(data.keys()))

    return data


# ---------------------------------------------------------
# 4. Extract feature combinations
# ---------------------------------------------------------

def get_features(data, variant):
    # BERT text features: [N, 768]
    bert_text = data["bert_text"].float()

    # CLIP image features: [N, 512]
    clip_image = data["clip_image"].float()

    # CLIP image-text similarity: [N, 1]
    clip_sim = data["clip_sim"].float().reshape(-1, 1)

    # Labels: 0 = FAKE, 1 = REAL
    labels = data["label"].long().reshape(-1)

    # Select features based on the experiment.
    if variant == "text_only":
        x = bert_text

    elif variant == "image_only":
        x = clip_image

    elif variant == "concat":
        # Concatenate BERT text and CLIP image features.
        x = torch.cat([bert_text, clip_image], dim=1)

    elif variant == "concat_sim":
        # Add CLIP similarity score as an extra feature.
        x = torch.cat([bert_text, clip_image, clip_sim], dim=1)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return x, labels


# ---------------------------------------------------------
# 5. Feature normalization
# ---------------------------------------------------------

def normalize_features(train_x, val_x, test_x):
    # Calculate mean and standard deviation using TRAIN only.
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True)

    # Prevent division by zero for constant features.
    std = std.clamp(min=1e-6)

    # Apply the same train statistics to all splits.
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std

    return train_x, val_x, test_x


# ---------------------------------------------------------
# 6. MLP classifier
# ---------------------------------------------------------

class FusionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Lightweight classifier over extracted features.
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------
# 7. Calculate metrics
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }

    # ROC-AUC requires probabilities for both classes.
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


# ---------------------------------------------------------
# 8. Evaluation function
# ---------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            # Forward pass.
            logits = model(x)

            # Convert logits into probabilities.
            probs = torch.softmax(logits, dim=1)

            # Predicted class.
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(y.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = calculate_metrics(
        all_labels, all_preds, all_probs
    )

    return metrics, all_labels, all_preds, all_probs


# ---------------------------------------------------------
# 9. Train one model for one seed
# ---------------------------------------------------------

def train_one_seed(
    train_x,
    train_y,
    val_x,
    val_y,
    test_x,
    test_y,
    seed,
    device,
    out_dir,
    variant,
    epochs=30,
    patience=6,
    batch_size=32,
    learning_rate=1e-3,
):
    set_seed(seed)

    # Move the model to the selected device.
    model = FusionMLP(train_x.shape[1]).to(device)

    # Adam optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    # Cross-entropy for binary classification (2 output logits).
    criterion = nn.CrossEntropyLoss()

    # Create data loaders.
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=batch_size,
        shuffle=False,
    )

    # Early stopping based on validation Macro-F1.
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    print(f"\nSeed {seed} | {variant}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0

        # Train one epoch.
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()

            # Prevent excessively large gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        # Evaluate on validation set.
        val_metrics, _, _, _ = evaluate(
            model, val_loader, device
        )

        avg_loss = total_loss / max(len(train_loader), 1)
        val_f1 = val_metrics["macro_f1"]

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        # Save the model with the best validation Macro-F1.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Stop if validation performance doesn't improve.
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Restore the best validation checkpoint.
    model.load_state_dict(best_state)

    # Evaluate the restored model on the test set.
    test_metrics, labels, preds, probs = evaluate(
        model, test_loader, device
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation Macro-F1: {best_val_f1:.4f}")
    print("Test metrics:")

    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save checkpoint.
    model_dir = os.path.join(out_dir, "models", variant)
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        model_dir, f"seed_{seed}.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": train_x.shape[1],
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )

    # Return results for later aggregation.
    return {
        "variant": variant,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        **test_metrics,
        "labels": labels,
        "preds": preds,
        "probs": probs,
    }


# ---------------------------------------------------------
# 10. Save confusion matrix plot
# ---------------------------------------------------------

def save_confusion_matrix(labels, preds, save_path, title):
    cm = confusion_matrix(
        labels, preds, labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks([0, 1], ["FAKE (0)", "REAL (1)"])
    plt.yticks([0, 1], ["FAKE (0)", "REAL (1)"])

    # Write the count inside each cell.
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=12,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Confusion matrix saved: {save_path}")


# ---------------------------------------------------------
# 11. Save results as CSV
# ---------------------------------------------------------

def save_results_csv(results, path):
    # Remove arrays before writing results.
    fields = [
        "variant",
        "seed",
        "best_epoch",
        "best_val_macro_f1",
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "roc_auc",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for result in results:
            row = {
                key: result[key]
                for key in fields
            }
            writer.writerow(row)

    print(f"Results CSV saved: {path}")


# ---------------------------------------------------------
# 12. Aggregate results (mean and standard deviation)
# ---------------------------------------------------------

def aggregate_results(results, variants):
    summary = {}

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "roc_auc",
    ]

    for variant in variants:
        rows = [
            r for r in results
            if r["variant"] == variant
        ]

        summary[variant] = {}

        for metric in metric_names:
            values = np.array(
                [r[metric] for r in rows],
                dtype=float,
            )

            summary[variant][metric] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
            }

    return summary


# ---------------------------------------------------------
# 13. Main pipeline
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Part 6: Feature Concatenation Fusion"
    )

    parser.add_argument(
        "--features_dir",
        type=str,
        default="features",
        help="Folder containing Part 5 feature files",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="results/fusion_concat",
        help="Output folder",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 2024, 7, 99],
        help="Random seeds for repeated experiments",
    )

    args = parser.parse_args()

    # Select device.
    device = get_device()

    print("=" * 65)
    print("PART 6: FEATURE CONCATENATION FUSION")
    print("=" * 65)
    print("Device:", device)
    print("Seeds:", args.seeds)

    os.makedirs(args.out, exist_ok=True)

    # Load all three cached feature files.
    train_data = load_features(
        os.path.join(args.features_dir, "train.pt")
    )

    val_data = load_features(
        os.path.join(args.features_dir, "validate.pt")
    )

    test_data = load_features(
        os.path.join(args.features_dir, "test.pt")
    )

    # Four feature configurations.
    variants = [
        "text_only",
        "image_only",
        "concat",
        "concat_sim",
    ]

    all_results = []

    # Run each experiment.
    for variant in variants:
        print("\n" + "=" * 65)
        print(f"EXPERIMENT: {variant.upper()}")
        print("=" * 65)

        # Prepare features and labels.
        train_x, train_y = get_features(
            train_data, variant
        )

        val_x, val_y = get_features(
            val_data, variant
        )

        test_x, test_y = get_features(
            test_data, variant
        )

        print("Train shape:", train_x.shape)
        print("Validation shape:", val_x.shape)
        print("Test shape:", test_x.shape)

        # Normalize using training data statistics only.
        train_x, val_x, test_x = normalize_features(
            train_x, val_x, test_x
        )

        # Run experiments with different seeds.
        for seed in args.seeds:
            result = train_one_seed(
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                test_x=test_x,
                test_y=test_y,
                seed=seed,
                device=device,
                out_dir=args.out,
                variant=variant,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
            )

            all_results.append(result)

            # Save prediction files for each run.
            pred_dir = os.path.join(
                args.out, "predictions", variant
            )
            os.makedirs(pred_dir, exist_ok=True)

            np.savez(
                os.path.join(pred_dir, f"seed_{seed}.npz"),
                labels=result["labels"],
                preds=result["preds"],
                probs=result["probs"],
            )

    # Save raw experiment results.
    csv_path = os.path.join(
        args.out, "all_seed_results.csv"
    )

    save_results_csv(all_results, csv_path)

    # Calculate mean and standard deviation across seeds.
    summary = aggregate_results(
        all_results, variants
    )

    # Save summary JSON.
    json_path = os.path.join(
        args.out, "summary.json"
    )

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 65)
    print("FINAL SUMMARY (mean ± std)")
    print("=" * 65)

    for variant in variants:
        print(f"\n{variant.upper()}")

        for metric in summary[variant]:
            mean = summary[variant][metric]["mean"]
            std = summary[variant][metric]["std"]

            print(f"{metric}: {mean:.4f} ± {std:.4f}")

    # Save a confusion matrix for the last seed of each variant.
    for variant in variants:
        variant_results = [
            r for r in all_results
            if r["variant"] == variant
        ]

        # Use the last seed's predictions for a representative CM.
        last_result = variant_results[-1]

        cm_dir = os.path.join(
            args.out, "confusion_matrices"
        )
        os.makedirs(cm_dir, exist_ok=True)

        save_confusion_matrix(
            last_result["labels"],
            last_result["preds"],
            os.path.join(
                cm_dir, f"{variant}.png"
            ),
            f"Confusion Matrix - {variant}",
        )

    print("\n" + "=" * 65)
    print("PART 6 COMPLETE")
    print("=" * 65)
    print(f"Results saved in: {args.out}")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
