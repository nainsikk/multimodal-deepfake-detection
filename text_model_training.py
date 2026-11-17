"""
Part 3: Text-only baseline (Baseline 1) - fine-tune DistilBERT on Fakeddit titles.

Run (after Parts 1 and 2):
    python part3_train_text.py --data_dir data --out results/text_only

Install extras:  pip install scikit-learn matplotlib

Outputs in --out:
    best_model/                 fine-tuned model + tokenizer (best val F1)
    metrics.json                val + test metrics, training history
    test_predictions.csv        for the team (error analysis, evaluation harness)
    val_predictions.csv
    confusion_matrix_test.png

Label convention: 0 = FAKE, 1 = REAL.
text_score in the predictions CSV = P(FAKE) from the text model.
"""
import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from dataset_builder import load_split_df, make_loaders

NAMES = ["FAKE", "REAL"]


def get_device():
    if torch.backends.mps.is_available():      # Apple Silicon (M1)
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    ids, labels, probs = [], [], []
    for b in loader:
        tok = {k: v.to(device) for k, v in b["tokens"].items()}
        logits = model(**tok).logits
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labels.extend(b["label"].tolist())
        ids.extend(b["id"])
    return ids, np.array(labels), np.concatenate(probs)   # probs: [N, 2] (FAKE, REAL)


def compute_metrics(y_true, probs):
    y_pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": {
            NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                       "f1": float(f[i]), "support": int(s[i])} for i in (0, 1)},
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if len(set(y_true)) == 2:   # AUC needs both classes present
        m["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        m["pr_auc_real"] = float(average_precision_score(y_true, probs[:, 1]))
    return m


def save_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(NAMES); ax.set_yticklabels(NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def export_predictions(data_dir, split, ids, probs, path):
    df = load_split_df(data_dir, split).set_index("id")
    rows = []
    for i, p in zip(ids, probs):
        r = df.loc[i]
        pred = int(p.argmax())
        rows.append({
            "id": i, "text": r["text"], "image_path": r["image_path"],
            "true_label": NAMES[int(r["label"])], "pred_label": NAMES[pred],
            "confidence": float(p.max()),
            "text_score": float(p[0]),        # P(FAKE)
            "image_score": "", "fusion_score": "",
            "label6": int(r["label6"]),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="results/text_only")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    set_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    device = get_device()
    print("Device:", device)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=2).to(device)

    train_dl, val_dl, test_dl = make_loaders(
        a.data_dir, tokenizer=tok, batch_size=a.batch_size,
        use_text=True, use_image=False, max_len=a.max_len)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(train_dl) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    best_f1, history = -1.0, []
    best_dir = os.path.join(a.out, "best_model")
    for ep in range(1, a.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for b in train_dl:
            tokd = {k: v.to(device) for k, v in b["tokens"].items()}
            y = b["label"].to(device)
            out = model(**tokd, labels=y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            loss_sum += out.loss.item() * len(y); n += len(y)
        _, yv, pv = predict_loader(model, val_dl, device)
        vm = compute_metrics(yv, pv)
        history.append({"epoch": ep, "train_loss": loss_sum / n,
                        "val_acc": vm["accuracy"], "val_macro_f1": vm["macro_f1"]})
        print(f"epoch {ep}: train_loss={loss_sum/n:.4f}  "
              f"val_acc={vm['accuracy']:.4f}  val_macro_f1={vm['macro_f1']:.4f}")
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            model.save_pretrained(best_dir); tok.save_pretrained(best_dir)

    # Reload the best checkpoint; evaluate on val and (once) on test.
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    results = {"model": a.model, "device": str(device), "args": vars(a),
               "history": history}
    for split, dl in [("val", val_dl), ("test", test_dl)]:
        ids, y, probs = predict_loader(model, dl, device)
        results[split] = compute_metrics(y, probs)
        sp = "validate" if split == "val" else "test"
        export_predictions(a.data_dir, sp, ids, probs,
                           os.path.join(a.out, f"{split}_predictions.csv"))
        if split == "test":
            save_confusion(results["test"]["confusion_matrix"],
                           os.path.join(a.out, "confusion_matrix_test.png"),
                           "Text-only (test)")
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    t = results["test"]
    print("\n=== TEST (text-only) ===")
    print(f"accuracy {t['accuracy']:.4f} | macro-F1 {t['macro_f1']:.4f} | "
          f"ROC-AUC {t.get('roc_auc', float('nan')):.4f}")
    print("confusion matrix [rows=true FAKE,REAL]:", t["confusion_matrix"])
    print("Saved to", a.out)


if __name__ == "__main__":
    main()