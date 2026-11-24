
"""
PART 9: Experimental Comparison and Statistical Analysis

Project: Multimodal Content Verification
Dataset: Fakeddit binary subset

Labels:
    0 = FAKE
    1 = REAL

Important:
    - Select seeds using validation Macro-F1 only.
    - Evaluate the selected seed on the test set.
    - Never select a model or seed using test performance.
    - Part 3 DistilBERT is a fine-tuned text baseline.
    - Part 6 text/image models use frozen features.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import binomtest

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

RANDOM_STATE = 42
N_BOOTSTRAP = 2000

RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "part9"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = Path("data/test/test_clean.csv")

SEEDS = [42, 123, 2024, 7, 99]

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------


def load_json(path):
    """Read a JSON file."""
    with open(path, "r") as file:
        return json.load(file)


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate classification metrics."""

    return {
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
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def bootstrap_confidence_intervals(
    y_true,
    y_pred,
    y_prob,
    n_bootstrap=N_BOOTSTRAP,
):
    """
    Calculate 95% bootstrap confidence intervals.

    Stratified bootstrap preserves the number of FAKE and REAL
    examples in every resampled test set.
    """

    rng = np.random.default_rng(RANDOM_STATE)

    class_0 = np.where(y_true == 0)[0]
    class_1 = np.where(y_true == 1)[0]

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "roc_auc",
    ]

    bootstrap_values = {name: [] for name in metric_names}

    for _ in range(n_bootstrap):

        sampled_0 = rng.choice(
            class_0, size=len(class_0), replace=True
        )

        sampled_1 = rng.choice(
            class_1, size=len(class_1), replace=True
        )

        indices = np.concatenate([sampled_0, sampled_1])
        rng.shuffle(indices)

        yt = y_true[indices]
        yp = y_pred[indices]
        ys = y_prob[indices]

        values = calculate_metrics(yt, yp, ys)

        for name in metric_names:
            bootstrap_values[name].append(values[name])

    intervals = {}

    for name in metric_names:

        lower = np.percentile(bootstrap_values[name], 2.5)
        upper = np.percentile(bootstrap_values[name], 97.5)

        intervals[name] = (lower, upper)

    return intervals


def select_best_seed(csv_path, model_column):
    """
    Select the seed with the highest validation Macro-F1.
    """

    df = pd.read_csv(csv_path)

    best_row = df.loc[df["best_val_macro_f1"].idxmax()]

    seed = int(best_row["seed"])
    val_score = float(best_row["best_val_macro_f1"])

    return seed, val_score


def load_npz_predictions(path, expected_labels):
    """Load and validate an NPZ prediction file."""

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    data = np.load(path)

    required = {"labels", "preds", "probs"}

    if not required.issubset(data.files):
        raise ValueError(
            f"{path} must contain {required}. Found: {data.files}"
        )

    labels = data["labels"].astype(int)
    preds = data["preds"].astype(int)
    probs = data["probs"].astype(float)

    if not np.array_equal(labels, expected_labels):
        raise ValueError(
            f"Test labels/order do not match the dataset: {path}"
        )

    if probs.shape != (len(expected_labels), 2):
        raise ValueError(f"Unexpected probability shape in {path}: {probs.shape}")

    if not np.array_equal(preds, np.argmax(probs, axis=1)):
        raise ValueError(
            f"Predictions do not match probability argmax: {path}"
        )

    if not np.all(np.isfinite(probs)):
        raise ValueError(f"Non-finite probabilities found: {path}")

    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError(f"Probabilities outside [0, 1]: {path}")

    return preds, probs[:, 1]


# ---------------------------------------------------------
# 3. LOAD TEST DATASET
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("PART 9: EXPERIMENTAL COMPARISON")
print("=" * 65)

test_df = pd.read_csv(DATASET_PATH)

sample_ids = test_df["id"].astype(str).to_numpy()
y_true = test_df["label"].to_numpy(dtype=int)

if len(test_df) != 400:
    raise ValueError(f"Expected 400 test samples, found {len(test_df)}")

if set(np.unique(y_true)) != {0, 1}:
    raise ValueError("Expected binary labels 0 and 1.")

print("\nTest dataset loaded successfully.")
print("Number of samples:", len(y_true))
print("FAKE samples:", int(np.sum(y_true == 0)))
print("REAL samples:", int(np.sum(y_true == 1)))


# ---------------------------------------------------------
# 4. LOAD PART 3: FINE-TUNED DISTILBERT BASELINE
# ---------------------------------------------------------

print("\nLoading Part 3: Fine-tuned DistilBERT...")

part3_path = RESULTS_DIR / "text_only" / "test_predictions.csv"

part3_df = pd.read_csv(part3_path)

if not np.array_equal(
    part3_df["id"].astype(str).to_numpy(),
    sample_ids,
):
    raise ValueError("Part 3 test sample IDs/order do not match dataset.")

part3_labels = (
    part3_df["true_label"]
    .astype(str)
    .str.upper()
    .map({"FAKE": 0, "REAL": 1})
    .to_numpy()
)

if not np.array_equal(part3_labels, y_true):
    raise ValueError("Part 3 labels do not match test dataset.")

part3_preds = (
    part3_df["pred_label"]
    .astype(str)
    .str.upper()
    .map({"FAKE": 0, "REAL": 1})
    .to_numpy()
)

# The confidence column stores confidence in the predicted class.
# Convert it into P(REAL), which is required for ROC-AUC.

confidence = part3_df["confidence"].to_numpy(dtype=float)

part3_prob = np.where(
    part3_preds == 1,
    confidence,
    1.0 - confidence,
)

if not np.all(np.isfinite(part3_prob)):
    raise ValueError("Invalid Part 3 confidence values.")

predictions = {}

predictions["DistilBERT fine-tuned"] = {
    "preds": part3_preds,
    "prob": part3_prob,
    "seed": 42,
    "val_macro_f1": None,
    "source": "Part 3",
}

# Compare reconstructed AUC with saved Part 3 AUC.
part3_metrics = load_json(RESULTS_DIR / "text_only" / "metrics.json")

saved_auc = float(part3_metrics["test"]["roc_auc"])
calculated_auc = roc_auc_score(y_true, part3_prob)

print("Part 3 reconstructed ROC-AUC:", round(calculated_auc, 5))
print("Part 3 saved ROC-AUC:", round(saved_auc, 5))

if abs(calculated_auc - saved_auc) > 0.005:
    warnings.warn(
        "Reconstructed Part 3 ROC-AUC differs from saved metrics. "
        "Check the meaning of the confidence column."
    )


# ---------------------------------------------------------
# 5. SELECT BEST SEEDS USING VALIDATION MACRO-F1
# ---------------------------------------------------------

print("\nSelecting best seeds using validation Macro-F1...")

model_configs = {
    "Frozen text-only": {
        "type": "part6",
        "variant": "text_only",
        "metrics": RESULTS_DIR / "fusion_concat" / "all_seed_results.csv",
    },

    "Frozen image-only": {
        "type": "part6",
        "variant": "image_only",
        "metrics": RESULTS_DIR / "fusion_concat" / "all_seed_results.csv",
    },

    "Concatenation": {
        "type": "part6",
        "variant": "concat",
        "metrics": RESULTS_DIR / "fusion_concat" / "all_seed_results.csv",
    },

    "Concatenation + CLIP similarity": {
        "type": "part6",
        "variant": "concat_sim",
        "metrics": RESULTS_DIR / "fusion_concat" / "all_seed_results.csv",
    },

    "Cross-attention": {
        "type": "part7",
        "variant": "crossattn",
        "metrics": RESULTS_DIR / "fusion_crossattn" / "crossattn" / "seed_results.csv",
    },

    "Cross-attention + CLIP similarity": {
        "type": "part7",
        "variant": "crossattn_sim",
        "metrics": RESULTS_DIR / "fusion_crossattn" / "crossattn_sim" / "seed_results.csv",
    },

    "Late fusion": {
        "type": "part7",
        "variant": "late_fusion",
        "metrics": RESULTS_DIR / "fusion_crossattn" / "late_fusion" / "seed_results.csv",
    },
}

selected_seeds = []

for model_name, config in model_configs.items():

    metrics_df = pd.read_csv(config["metrics"])

    best_row = metrics_df.loc[
        metrics_df["best_val_macro_f1"].idxmax()
    ]

    seed = int(best_row["seed"])
    val_score = float(best_row["best_val_macro_f1"])

    selected_seeds.append({
        "model": model_name,
        "selected_seed": seed,
        "validation_macro_f1": val_score,
    })

    if config["type"] == "part6":

        prediction_path = (
            RESULTS_DIR
            / "fusion_concat"
            / "predictions"
            / config["variant"]
            / f"seed_{seed}.npz"
        )

    else:

        prediction_path = (
            RESULTS_DIR
            / "fusion_crossattn"
            / config["variant"]
            / f"seed_{seed}"
            / "test_predictions.npz"
        )

    preds, prob = load_npz_predictions(
        prediction_path,
        y_true,
    )

    predictions[model_name] = {
        "preds": preds,
        "prob": prob,
        "seed": seed,
        "val_macro_f1": val_score,
        "source": config["type"],
    }

    print(
        f"{model_name:40s} "
        f"seed={seed:<5} "
        f"validation Macro-F1={val_score:.4f}"
    )

selected_df = pd.DataFrame(selected_seeds)

selected_df.to_csv(
    OUTPUT_DIR / "selected_seeds.csv",
    index=False,
)


# ---------------------------------------------------------
# 6. TEST METRICS + 95% BOOTSTRAP CONFIDENCE INTERVALS
# ---------------------------------------------------------

print("\nCalculating test metrics and bootstrap confidence intervals...")

metric_rows = []

for model_name, result in predictions.items():

    preds = result["preds"]
    prob = result["prob"]

    metrics = calculate_metrics(y_true, preds, prob)

    intervals = bootstrap_confidence_intervals(
        y_true,
        preds,
        prob,
    )

    row = {
        "model": model_name,
        "selected_seed": result["seed"],
        "validation_macro_f1": result["val_macro_f1"],
    }

    for metric_name, value in metrics.items():

        lower, upper = intervals[metric_name]

        row[metric_name] = value
        row[f"{metric_name}_ci_lower"] = lower
        row[f"{metric_name}_ci_upper"] = upper

    metric_rows.append(row)

    print(
        f"{model_name:40s} "
        f"Accuracy={metrics['accuracy']:.4f} "
        f"Macro-F1={metrics['macro_f1']:.4f} "
        f"AUC={metrics['roc_auc']:.4f}"
    )

metrics_df = pd.DataFrame(metric_rows)

metrics_df.to_csv(
    OUTPUT_DIR / "test_metrics_with_ci.csv",
    index=False,
)


# ---------------------------------------------------------
# 7. EXACT MCNEMAR TESTS
# ---------------------------------------------------------

print("\nRunning exact McNemar tests...")

model_names = list(predictions.keys())

mcnemar_rows = []

for i in range(len(model_names)):

    for j in range(i + 1, len(model_names)):

        model_a = model_names[i]
        model_b = model_names[j]

        pred_a = predictions[model_a]["preds"]
        pred_b = predictions[model_b]["preds"]

        correct_a = pred_a == y_true
        correct_b = pred_b == y_true

        # Samples where A is correct and B is incorrect.
        a_only_correct = int(
            np.sum(correct_a & ~correct_b)
        )

        # Samples where B is correct and A is incorrect.
        b_only_correct = int(
            np.sum(~correct_a & correct_b)
        )

        discordant = a_only_correct + b_only_correct

        if discordant == 0:
            p_value = 1.0
        else:
            p_value = binomtest(
                a_only_correct,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue

        mcnemar_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "a_only_correct": a_only_correct,
            "b_only_correct": b_only_correct,
            "discordant_pairs": discordant,
            "p_value": p_value,
        })


# Holm correction controls family-wise error rate.
def holm_correction(p_values):

    p_values = np.asarray(p_values, dtype=float)

    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)

    total = len(p_values)
    running_max = 0.0

    for rank, index in enumerate(order):

        corrected = (total - rank) * p_values[index]
        running_max = max(running_max, corrected)

        adjusted[index] = min(running_max, 1.0)

    return adjusted


raw_p_values = [
    row["p_value"] for row in mcnemar_rows
]

adjusted_p_values = holm_correction(raw_p_values)

for row, adjusted in zip(mcnemar_rows, adjusted_p_values):

    row["holm_adjusted_p_value"] = adjusted
    row["significant_0.05"] = adjusted < 0.05

mcnemar_df = pd.DataFrame(mcnemar_rows)

mcnemar_df.to_csv(
    OUTPUT_DIR / "mcnemar_tests.csv",
    index=False,
)

print("McNemar comparisons completed:", len(mcnemar_df))


# ---------------------------------------------------------
# 8. COMPLEMENTARITY ANALYSIS
# ---------------------------------------------------------

print("\nCalculating model complementarity...")

def complementarity(model_a, model_b):

    pred_a = predictions[model_a]["preds"]
    pred_b = predictions[model_b]["preds"]

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    return {
        "model_a": model_a,
        "model_b": model_b,
        "both_correct": int(np.sum(correct_a & correct_b)),
        "only_a_correct": int(np.sum(correct_a & ~correct_b)),
        "only_b_correct": int(np.sum(~correct_a & correct_b)),
        "both_wrong": int(np.sum(~correct_a & ~correct_b)),
    }


# Find the strongest multimodal model using validation Macro-F1.
multimodal_names = [
    "Concatenation",
    "Concatenation + CLIP similarity",
    "Cross-attention",
    "Cross-attention + CLIP similarity",
    "Late fusion",
]

best_multimodal = max(
    multimodal_names,
    key=lambda name: predictions[name]["val_macro_f1"],
)

print("Best multimodal model by validation Macro-F1:", best_multimodal)

complementarity_pairs = [
    ("DistilBERT fine-tuned", "Frozen image-only"),
    ("Frozen text-only", "Frozen image-only"),
    ("DistilBERT fine-tuned", best_multimodal),
    ("Frozen image-only", best_multimodal),
]

complementarity_rows = [
    complementarity(a, b)
    for a, b in complementarity_pairs
]

complementarity_df = pd.DataFrame(complementarity_rows)

complementarity_df.to_csv(
    OUTPUT_DIR / "complementarity.csv",
    index=False,
)


# ---------------------------------------------------------
# 9. SAVE DISAGREEMENT CASES
# ---------------------------------------------------------

print("\nSaving disagreement cases...")

baseline_name = "DistilBERT fine-tuned"
comparison_name = best_multimodal

baseline_preds = predictions[baseline_name]["preds"]
comparison_preds = predictions[comparison_name]["preds"]

disagreement_mask = baseline_preds != comparison_preds

disagreement_df = test_df.loc[
    disagreement_mask
].copy()

disagreement_df["true_label_text"] = disagreement_df["label_name"]

disagreement_df["distilbert_prediction"] = np.where(
    baseline_preds[disagreement_mask] == 1,
    "REAL",
    "FAKE",
)

disagreement_df["multimodal_prediction"] = np.where(
    comparison_preds[disagreement_mask] == 1,
    "REAL",
    "FAKE",
)

disagreement_df["distilbert_correct"] = (
    baseline_preds[disagreement_mask]
    == y_true[disagreement_mask]
)

disagreement_df["multimodal_correct"] = (
    comparison_preds[disagreement_mask]
    == y_true[disagreement_mask]
)

disagreement_df.to_csv(
    OUTPUT_DIR / "disagreement_cases.csv",
    index=False,
)

print("Disagreement cases:", len(disagreement_df))


# ---------------------------------------------------------
# 10. CONFUSION MATRICES
# ---------------------------------------------------------

print("\nGenerating confusion matrices...")

confusion_dir = OUTPUT_DIR / "confusion_matrices"
confusion_dir.mkdir(exist_ok=True)

for model_name, result in predictions.items():

    cm = confusion_matrix(
        y_true,
        result["preds"],
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["FAKE", "REAL"],
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(model_name)
    fig.tight_layout()

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )

    fig.savefig(
        confusion_dir / f"{safe_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------
# 11. PERFORMANCE COMPARISON FIGURE
# ---------------------------------------------------------

print("\nGenerating performance comparison figure...")

plot_metrics = ["accuracy", "macro_f1", "roc_auc"]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6),
)

for ax, metric in zip(axes, plot_metrics):

    values = metrics_df[metric].to_numpy()

    lower_errors = (
        values - metrics_df[f"{metric}_ci_lower"].to_numpy()
    )

    upper_errors = (
        metrics_df[f"{metric}_ci_upper"].to_numpy() - values
    )

    ax.bar(
        range(len(metrics_df)),
        values,
        yerr=np.vstack([lower_errors, upper_errors]),
        capsize=4,
    )

    ax.set_title(metric.replace("_", " ").title())
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(metrics_df)))
    ax.set_xticklabels(
        metrics_df["model"],
        rotation=75,
        ha="right",
        fontsize=8,
    )

    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "model_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------
# 12. SAVE ALL PREDICTIONS WITH SAMPLE IDS
# ---------------------------------------------------------

prediction_rows = []

for model_name, result in predictions.items():

    for index in range(len(y_true)):

        prediction_rows.append({
            "id": sample_ids[index],
            "true_label": int(y_true[index]),
            "model": model_name,
            "predicted_label": int(result["preds"][index]),
            "probability_real": float(result["prob"][index]),
            "correct": bool(
                result["preds"][index] == y_true[index]
            ),
        })

all_predictions_df = pd.DataFrame(prediction_rows)

all_predictions_df.to_csv(
    OUTPUT_DIR / "all_model_predictions.csv",
    index=False,
)


# ---------------------------------------------------------
# 13. SAVE SUMMARY JSON
# ---------------------------------------------------------

summary = {
    "project": "Multimodal Content Verification",
    "dataset": "Fakeddit binary subset",
    "test_samples": int(len(y_true)),
    "label_mapping": {
        "0": "FAKE",
        "1": "REAL",
    },
    "bootstrap_iterations": N_BOOTSTRAP,
    "confidence_interval": "95%",
    "seed_selection": "Highest validation Macro-F1",
    "best_multimodal_model_by_validation": best_multimodal,
    "models": {},
}

for _, row in metrics_df.iterrows():

    model_name = row["model"]

    summary["models"][model_name] = {
        "selected_seed": int(row["selected_seed"]),
        "validation_macro_f1": (
            None
            if pd.isna(row["validation_macro_f1"])
            else float(row["validation_macro_f1"])
        ),
        "test_metrics": {
            metric: float(row[metric])
            for metric in [
                "accuracy",
                "precision",
                "recall",
                "macro_f1",
                "roc_auc",
            ]
        },
        "confidence_intervals": {
            metric: [
                float(row[f"{metric}_ci_lower"]),
                float(row[f"{metric}_ci_upper"]),
            ]
            for metric in [
                "accuracy",
                "precision",
                "recall",
                "macro_f1",
                "roc_auc",
            ]
        },
    }

with open(OUTPUT_DIR / "summary.json", "w") as file:
    json.dump(summary, file, indent=4)


# ---------------------------------------------------------
# 14. GENERATE MARKDOWN REPORT
# ---------------------------------------------------------

report = []

report.append("# Part 9: Experimental Comparison and Statistical Analysis\n")

report.append("## 1. Experimental Setup\n")

report.append(
    "- Dataset: Fakeddit binary subset\n"
    "- Test samples: 400\n"
    "- Classes: FAKE (0) and REAL (1)\n"
    "- Seed selection: highest validation Macro-F1\n"
    "- Confidence intervals: 95% stratified bootstrap\n"
    "- Statistical test: exact two-sided McNemar test\n"
    "- Multiple comparison correction: Holm correction\n"
)

report.append("\n## 2. Selected Seeds\n")

report.append(selected_df.to_markdown(index=False))

report.append("\n## 3. Test Performance and Confidence Intervals\n")

report.append(metrics_df.to_markdown(index=False, floatfmt=".4f"))

report.append("\n## 4. McNemar Significance Tests\n")

report.append(mcnemar_df.to_markdown(index=False, floatfmt=".5f"))

report.append("\n## 5. Complementarity Analysis\n")

report.append(complementarity_df.to_markdown(index=False))

report.append("\n## 6. Disagreement Analysis\n")

report.append(
    f"- Baseline: {baseline_name}\n"
    f"- Multimodal comparison: {best_multimodal}\n"
    f"- Number of disagreement cases: {len(disagreement_df)}\n"
)

report.append("\n## 7. Interpretation Notes\n")

report.append(
    "- The selected seed for each model was chosen using validation Macro-F1.\n"
    "- Test results were used only for final evaluation.\n"
    "- Bootstrap confidence intervals quantify uncertainty in test metrics.\n"
    "- McNemar tests compare paired classification errors on the same test samples.\n"
    "- Holm-adjusted p-values account for multiple pairwise comparisons.\n"
    "- Statistical significance does not automatically imply practical importance.\n"
    "- The fine-tuned DistilBERT baseline and frozen-feature text-only model "
    "are different experimental systems.\n"
    "- The image-only baseline is the frozen-feature Part 6 model; "
    "a separate fine-tuned Part 4 image-only result was not available.\n"
)

with open(OUTPUT_DIR / "report.md", "w") as file:
    file.write("\n".join(report))


# ---------------------------------------------------------
# 15. FINAL OUTPUT
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("PART 9 COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nBest multimodal model:", best_multimodal)

print("\nMain output files:")

for path in sorted(OUTPUT_DIR.rglob("*")):

    if path.is_file():
        print(" -", path)

print("\nAll Part 9 results saved in:", OUTPUT_DIR)
