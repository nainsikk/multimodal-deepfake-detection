"""
Part 4: Image-only baseline using ResNet.

Run:
    python image_model_training.py --data_dir data --out results/image_only

Labels:
    0 = FAKE
    1 = REAL
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

from dataset_builder import load_split_df, make_loaders
from text_model_training import (
    get_device,
    set_seed,
    compute_metrics,
    save_confusion,
)

NAMES = ["FAKE", "REAL"]


def build_model(arch="resnet18", freeze_backbone=False):
    """Load pretrained ResNet and replace its final layer."""

    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Optionally freeze pretrained layers
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final classification layer with 2 classes
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )

    return model


@torch.no_grad()
def predict_loader(model, loader, device):
    """Predict fake/real probabilities for a DataLoader."""

    model.eval()

    ids = []
    labels = []
    probabilities = []

    for batch in loader:
        images = batch["image"].to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        probabilities.append(probs.cpu().numpy())
        labels.extend(batch["label"].tolist())
        ids.extend(batch["id"])

    return ids, np.array(labels), np.concatenate(probabilities)


def export_predictions(data_dir, split, ids, probs, output_path):
    """Save predictions in CSV format."""

    df = load_split_df(data_dir, split).set_index("id")

    rows = []

    for sample_id, probability in zip(ids, probs):
        row = df.loc[sample_id]

        pred = int(probability.argmax())

        rows.append({
            "id": sample_id,
            "text": row["text"],
            "image_path": row["image_path"],
            "true_label": NAMES[int(row["label"])],
            "pred_label": NAMES[pred],
            "confidence": float(probability.max()),
            "text_score": "",
            "image_score": float(probability[0]),  # P(FAKE)
            "fusion_score": "",
            "label6": int(row["label6"]),
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)


def main():

    parser = argparse.ArgumentParser(
        description="Train an image-only fake/real classifier"
    )

    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out", default="results/image_only")
    parser.add_argument(
        "--arch",
        default="resnet18",
        choices=["resnet18", "resnet50"]
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = get_device()
    print("Device:", device)

    # Create image-only DataLoaders
    train_loader, val_loader, test_loader = make_loaders(
        args.data_dir,
        tokenizer=None,
        batch_size=args.batch_size,
        use_text=False,
        use_image=True,
    )

    # Build model
    model = build_model(args.arch, args.freeze_backbone).to(device)

    # Optimizer only updates trainable parameters
    trainable_params = [
        p for p in model.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    history = []

    best_model_path = os.path.join(args.out, "best_model.pt")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, args.epochs + 1):

        model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

        scheduler.step()

        # Validation after each epoch
        _, val_labels, val_probs = predict_loader(
            model, val_loader, device
        )

        val_metrics = compute_metrics(val_labels, val_probs)

        avg_loss = total_loss / total_samples

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        # Save best model based on validation Macro-F1
        if val_metrics["macro_f1"] > best_f1:

            best_f1 = val_metrics["macro_f1"]

            torch.save(model.state_dict(), best_model_path)

            print("  -> Best model saved!")

    # -------------------------
    # Load best model
    # -------------------------
    model.load_state_dict(
        torch.load(best_model_path, map_location=device)
    )

    results = {
        "model": args.arch,
        "device": str(device),
        "args": vars(args),
        "history": history,
    }

    # -------------------------
    # Final evaluation
    # -------------------------
    for split, loader in [
        ("val", val_loader),
        ("test", test_loader),
    ]:

        ids, labels, probs = predict_loader(
            model, loader, device
        )

        split_metrics = compute_metrics(labels, probs)
        results[split] = split_metrics

        actual_split = "validate" if split == "val" else "test"

        export_predictions(
            args.data_dir,
            actual_split,
            ids,
            probs,
            os.path.join(args.out, f"{split}_predictions.csv"),
        )

        if split == "test":
            save_confusion(
                split_metrics["confusion_matrix"],
                os.path.join(args.out, "confusion_matrix_test.png"),
                "Image-only Test",
            )

    # Save all metrics
    with open(os.path.join(args.out, "metrics.json"), "w") as file:
        json.dump(results, file, indent=2)

    test_metrics = results["test"]

    print("\n===== IMAGE-ONLY TEST RESULTS =====")
    print(
        f"Accuracy: {test_metrics['accuracy']:.4f} | "
        f"Macro-F1: {test_metrics['macro_f1']:.4f} | "
        f"ROC-AUC: {test_metrics.get('roc_auc', float('nan')):.4f}"
    )

    print(
        "Confusion matrix [rows=true FAKE, REAL]:",
        test_metrics["confusion_matrix"]
    )

    print("Saved results to:", args.out)


if __name__ == "__main__":
    main()