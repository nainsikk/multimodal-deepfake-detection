# ============================================================
# PART 8: CALIBRATION + UNCERTAINTY
# Multimodal Deepfake / Fake Content Detection
#
# Input:
#   Part 7 validation/test .npz predictions
#
# Label mapping:
#   0 = FAKE
#   1 = REAL
#
# Probability mapping:
#   probs[:, 0] = P(FAKE)
#   probs[:, 1] = P(REAL)
#
# Important:
#   - Best seed selected using VALIDATION Macro-F1
#   - Temperature fitted using VALIDATION only
#   - Uncertainty threshold selected using VALIDATION only
#   - TEST used for final reporting only
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = "results/fusion_crossattn"

EXPERIMENTS = [
    "crossattn",
    "crossattn_sim",
    "late_fusion",
]

SEEDS = [42, 123, 2024, 7, 99]

# Confidence threshold search
TARGET_ACCURACY = 0.80
MIN_COVERAGE = 0.50

# Temperature search range
TEMPERATURES = np.exp(
    np.linspace(np.log(0.05), np.log(10.0), 500)
)

# Confidence threshold search range
TAUS = np.arange(0.50, 0.991, 0.01)

EPS = 1e-7

print("=" * 65)
print("PART 8: CALIBRATION & UNCERTAINTY")
print("=" * 65)


# ============================================================
# 2. LOAD NPZ PREDICTIONS
# ============================================================

def load_predictions(path):
    """
    Load predictions saved by Part 7.

    Expected NPZ keys:
        labels : actual labels
        probs : predicted probabilities
        preds : predicted classes

    probs:
        column 0 = P(FAKE)
        column 1 = P(REAL)
    """

    data = np.load(path)

    labels = data["labels"].astype(int)
    probs = data["probs"].astype(float)

    # Check that the probability array has two classes.
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError(
            f"Unexpected probability shape in {path}: {probs.shape}"
        )

    # Extract probability of FAKE.
    p_fake = probs[:, 0]

    # Prevent log(0) during calibration.
    p_fake = np.clip(p_fake, EPS, 1 - EPS)

    return labels, p_fake


# ============================================================
# 3. BASIC PROBABILITY FUNCTIONS
# ============================================================

def sigmoid(x):
    """Convert logits into probabilities."""

    x = np.clip(x, -50, 50)

    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    """Convert probabilities into logits."""

    p = np.clip(p, EPS, 1 - EPS)

    return np.log(p / (1 - p))


def apply_temperature(p_fake, temperature):
    """
    Temperature scaling.

    T > 1:
        probabilities become less confident.

    T < 1:
        probabilities become more confident.

    T = 1:
        probabilities remain unchanged.
    """

    logits = logit(p_fake)

    calibrated_logits = logits / temperature

    return sigmoid(calibrated_logits)


def get_predictions(p_fake):
    """
    Convert P(FAKE) into predicted labels.

    Label 0 = FAKE
    Label 1 = REAL
    """

    return (p_fake < 0.5).astype(int)


def get_confidence(p_fake):
    """
    Confidence is the probability assigned
    to the predicted class.

    Example:
        P(FAKE) = 0.90 -> confidence = 0.90
        P(FAKE) = 0.30 -> confidence = 0.70
    """

    return np.maximum(p_fake, 1 - p_fake)


# ============================================================
# 4. METRICS
# ============================================================

def calculate_nll(p_fake, y_true):
    """
    Negative Log-Likelihood.

    Measures how well predicted probabilities
    match the actual labels.

    Lower is better.
    """

    # Probability assigned to the correct class.
    p_correct = np.where(
        y_true == 0,
        p_fake,       # FAKE probability
        1 - p_fake    # REAL probability
    )

    p_correct = np.clip(p_correct, EPS, 1.0)

    return float(-np.mean(np.log(p_correct))


def calculate_brier(p_fake, y_true):
    """
    Brier Score.

    Measures squared probability prediction error.

    Lower is better.
    """

    # Convert the probability to REAL.
    p_real = 1 - p_fake

    return float(np.mean((p_real - y_true) ** 2))


def calculate_ece(p_fake, y_true, bins=10):
    """
    Expected Calibration Error.

    Groups predictions into confidence bins.

    ECE compares:
        average confidence
        actual accuracy

    Lower is better.
    """

    predictions = get_predictions(p_fake)
    confidence = get_confidence(p_fake)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    ece = 0.0

    for i in range(bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        # Select samples inside this confidence bin.
        if i == bins - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        if np.sum(mask) == 0:
            continue

        # Accuracy within the current bin.
        bin_accuracy = np.mean(
            predictions[mask] == y_true[mask]
        )

        # Average confidence within the current bin.
        bin_confidence = np.mean(confidence[mask])

        # Number of samples in this bin.
        bin_fraction = np.mean(mask)

        # Weighted calibration error.
        ece += bin_fraction * abs(
            bin_accuracy - bin_confidence
        )

    return float(ece)


def calculate_metrics(p_fake, y_true):
    """Calculate the main classification and calibration metrics."""

    predictions = get_predictions(p_fake)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),

        "precision_macro": float(
            precision_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0
            )
        ),

        "recall_macro": float(
            recall_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0
            )
        ),

        "NLL": calculate_nll(
            p_fake, y_true
        ),

        "Brier": calculate_brier(
            p_fake, y_true
        ),

        "ECE": calculate_ece(
            p_fake, y_true
        ),
    }

    try:
        metrics["ROC_AUC"] = float(
            roc_auc_score(
                y_true,
                1 - p_fake
            )
        )
    except ValueError:
        metrics["ROC_AUC"] = float("nan")

    return metrics


# ============================================================
# 5. FIT TEMPERATURE ON VALIDATION DATA
# ============================================================

def fit_temperature(p_fake_val, y_val):
    """
    Find the temperature that gives the lowest
    validation Negative Log-Likelihood.

    Test data is NOT used here.
    """

    best_temperature = 1.0
    best_nll = float("inf")

    for temperature in TEMPERATURES:

        calibrated = apply_temperature(
            p_fake_val,
            temperature
        )

        current_nll = calculate_nll(
            calibrated,
            y_val
        )

        if current_nll < best_nll:

            best_nll = current_nll
            best_temperature = float(temperature)

    return best_temperature, best_nll


# ============================================================
# 6. SELECT UNCERTAINTY THRESHOLD ON VALIDATION
# ============================================================

def evaluate_threshold(p_fake, y_true, tau):
    """
    Only answer when confidence >= tau.

    Samples below threshold become UNCERTAIN.
    """

    predictions = get_predictions(p_fake)
    confidence = get_confidence(p_fake)

    answered = confidence >= tau

    coverage = float(np.mean(answered))

    if np.sum(answered) == 0:

        return {
            "tau": float(tau),
            "coverage": 0.0,
            "answered_accuracy": 0.0,
            "answered_count": 0,
        }

    answered_accuracy = float(
        accuracy_score(
            y_true[answered],
            predictions[answered]
        )
    )

    return {
        "tau": float(tau),
        "coverage": coverage,
        "answered_accuracy": answered_accuracy,
        "answered_count": int(np.sum(answered)),
    }


def select_threshold(p_fake_val, y_val):
    """
    Choose the smallest tau that achieves:

        answered accuracy >= TARGET_ACCURACY
        coverage >= MIN_COVERAGE

    If no tau meets the accuracy goal:
        choose the threshold with the highest
        accuracy while satisfying minimum coverage.
    """

    results = []

    for tau in TAUS:

        row = evaluate_threshold(
            p_fake_val,
            y_val,
            tau
        )

        results.append(row)

    sweep = pd.DataFrame(results)

    # Find thresholds satisfying both requirements.
    eligible = sweep[
        (sweep["answered_accuracy"] >= TARGET_ACCURACY)
        &
        (sweep["coverage"] >= MIN_COVERAGE)
    ]

    if len(eligible) > 0:

        # Use the smallest threshold meeting requirements.
        selected = eligible.sort_values(
            "tau"
        ).iloc[0]

        selection_reason = (
            "Reached target accuracy and minimum coverage"
        )

    else:

        # No threshold met the target.
        # Find the highest accuracy among thresholds
        # that maintain minimum coverage.
        eligible_coverage = sweep[
            sweep["coverage"] >= MIN_COVERAGE
        ]

        if len(eligible_coverage) > 0:

            selected = eligible_coverage.sort_values(
                ["answered_accuracy", "tau"],
                ascending=[False, True]
            ).iloc[0]

            selection_reason = (
                "Fallback: highest accuracy while maintaining minimum coverage"
            )

        else:

            # If even the minimum coverage constraint
            # cannot be met, use tau = 0.50.
            selected = sweep.iloc[0]

            selection_reason = (
                "Fallback: no threshold met minimum coverage"
            )

    return (
        float(selected["tau"]),
        sweep,
        selection_reason
    )


# ============================================================
# 7. REAL / FAKE / UNCERTAIN DECISION
# ============================================================

def final_decisions(p_fake, tau):
    """
    Final decision rules:

        confidence >= tau:
            FAKE or REAL

        confidence < tau:
            UNCERTAIN
    """

    predictions = get_predictions(p_fake)
    confidence = get_confidence(p_fake)

    decisions = []

    for pred, conf in zip(predictions, confidence):

        if conf < tau:

            decisions.append("UNCERTAIN")

        elif pred == 0:

            decisions.append("FAKE")

        else:

            decisions.append("REAL")

    return decisions, confidence, predictions


# ============================================================
# 8. RELIABILITY DIAGRAM
# ============================================================

def reliability_plot(
    p_fake,
    y_true,
    title,
    save_path,
    bins=10
):
    """
    Plot confidence vs actual accuracy.

    A perfectly calibrated model follows
    the diagonal line.
    """

    predictions = get_predictions(p_fake)
    confidence = get_confidence(p_fake)

    bin_edges = np.linspace(0, 1, bins + 1)

    avg_confidence = []
    avg_accuracy = []

    for i in range(bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == bins - 1:

            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        if np.sum(mask) == 0:
            continue

        avg_confidence.append(
            np.mean(confidence[mask])
        )

        avg_accuracy.append(
            np.mean(
                predictions[mask] == y_true[mask]
            )
        )

    plt.figure(figsize=(7, 6))

    # Perfect calibration reference.
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(
        avg_confidence,
        avg_accuracy,
        marker="o",
        label="Model"
    )

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()

    print("Saved:", save_path)


# ============================================================
# 9. ACCURACY-COVERAGE PLOT
# ============================================================

def accuracy_coverage_plot(
    p_fake,
    y_true,
    save_path
):
    """
    Shows how answered-sample accuracy changes
    when confidence threshold increases.

    Higher threshold:
        fewer samples answered
        usually higher confidence
    """

    results = []

    for tau in TAUS:

        results.append(
            evaluate_threshold(
                p_fake,
                y_true,
                tau
            )
        )

    df = pd.DataFrame(results)

    plt.figure(figsize=(7, 6))

    plt.plot(
        df["coverage"],
        df["answered_accuracy"],
        marker="o"
    )

    plt.xlabel("Coverage (fraction of samples answered)")
    plt.ylabel("Accuracy on answered samples")

    plt.title("Accuracy vs Coverage")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()

    print("Saved:", save_path)

    return df


# ============================================================
# 10. PROCESS ONE EXPERIMENT
# ============================================================

def process_experiment(experiment):
    """
    Complete Part 8 for one experiment.

    1. Load validation and test predictions for all seeds.
    2. Select best seed using validation Macro-F1.
    3. Fit temperature using validation.
    4. Select tau using validation.
    5. Apply calibration to test.
    6. Save metrics, plots, CSV and JSON.
    """

    print("\n")
    print("=" * 65)
    print("EXPERIMENT:", experiment)
    print("=" * 65)

    experiment_dir = os.path.join(
        BASE_DIR,
        experiment
    )

    # --------------------------------------------------------
    # 10A. Find best seed using VALIDATION ONLY
    # --------------------------------------------------------

    best_seed = None
    best_val_f1 = -1.0

    best_val_labels = None
    best_val_p_fake = None
    best_test_labels = None
    best_test_p_fake = None

    for seed in SEEDS:

        seed_dir = os.path.join(
            experiment_dir,
            f"seed_{seed}"
        )

        val_path = os.path.join(
            seed_dir,
            "val_predictions.npz"
        )

        test_path = os.path.join(
            seed_dir,
            "test_predictions.npz"
        )

        # Skip missing seeds.
        if not os.path.exists(val_path):
            print(f"Skipping missing validation file: {val_path}")
            continue

        if not os.path.exists(test_path):
            print(f"Skipping missing test file: {test_path}")
            continue

        y_val, p_fake_val = load_predictions(val_path)

        y_test, p_fake_test = load_predictions(test_path)

        # Calculate validation Macro-F1.
        val_metrics = calculate_metrics(
            p_fake_val,
            y_val
        )

        print(
            f"Seed {seed} | "
            f"Validation Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        # Select the seed with highest validation Macro-F1.
        if val_metrics["macro_f1"] > best_val_f1:

            best_val_f1 = val_metrics["macro_f1"]
            best_seed = seed

            best_val_labels = y_val
            best_val_p_fake = p_fake_val

            best_test_labels = y_test
            best_test_p_fake = p_fake_test

    # Ensure prediction files were found.
    if best_seed is None:

        print(f"No prediction files found for {experiment}.")
        print("Skipping experiment.")

        return None

    print("\nSelected seed:", best_seed)
    print("Selected validation Macro-F1:", round(best_val_f1, 4))

    # --------------------------------------------------------
    # 10B. Fit temperature on VALIDATION
    # --------------------------------------------------------

    temperature, val_nll_after_temp = fit_temperature(
        best_val_p_fake,
        best_val_labels
    )

    val_calibrated = apply_temperature(
        best_val_p_fake,
        temperature
    )

    print("\nTemperature:", round(temperature, 4))
    print("Validation NLL after calibration:", round(val_nll_after_temp, 4))

    # --------------------------------------------------------
    # 10C. Select uncertainty threshold on VALIDATION
    # --------------------------------------------------------

    tau, threshold_sweep, reason = select_threshold(
        val_calibrated,
        best_val_labels
    )

    print("\nSelected tau:", round(tau, 3))
    print("Threshold selection:", reason)

    # --------------------------------------------------------
    # 10D. Apply temperature to TEST
    # --------------------------------------------------------

    test_calibrated = apply_temperature(
        best_test_p_fake,
        temperature
    )

    # --------------------------------------------------------
    # 10E. Test metrics BEFORE and AFTER calibration
    # --------------------------------------------------------

    test_metrics_before = calculate_metrics(
        best_test_p_fake,
        best_test_labels
    )

    test_metrics_after = calculate_metrics(
        test_calibrated,
        best_test_labels
    )

    print("\nTEST METRICS BEFORE CALIBRATION")

    for name, value in test_metrics_before.items():
        print(f"{name:18s}: {value:.4f}")

    print("\nTEST METRICS AFTER CALIBRATION")

    for name, value in test_metrics_after.items():
        print(f"{name:18s}: {value:.4f}")

    # --------------------------------------------------------
    # 10F. Apply UNCERTAIN rule to TEST
    # --------------------------------------------------------

    decisions, confidence, raw_predictions = final_decisions(
        test_calibrated,
        tau
    )

    answered_mask = np.array(
        [d != "UNCERTAIN" for d in decisions]
    )

    coverage = float(np.mean(answered_mask))

    if np.sum(answered_mask) > 0:

        answered_accuracy = float(
            accuracy_score(
                best_test_labels[answered_mask],
                raw_predictions[answered_mask]
            )
        )

    else:

        answered_accuracy = 0.0

    # Overall test accuracy (before abstaining).
    overall_accuracy = float(
        accuracy_score(
            best_test_labels,
            raw_predictions
        )
    )

    print("\nUNCERTAINTY RESULTS")

    print("Threshold tau:", round(tau, 3))
    print("Overall test accuracy:", round(overall_accuracy, 4))
    print("Coverage:", round(coverage, 4))
    print("Answered samples:", int(np.sum(answered_mask)))
    print("Uncertain samples:", int(np.sum(~answered_mask)))
    print("Accuracy on answered samples:", round(answered_accuracy, 4))

    # --------------------------------------------------------
    # 10G. Create output directory
    # --------------------------------------------------------

    output_dir = os.path.join(
        "results",
        "calibration",
        experiment
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 10H. Save calibrated test CSV
    # --------------------------------------------------------

    test_df = pd.DataFrame({
        "true_label": [
            "FAKE" if y == 0 else "REAL"
            for y in best_test_labels
        ],

        "p_fake_before": best_test_p_fake,

        "p_fake_calibrated": test_calibrated,

        "p_real_calibrated": 1 - test_calibrated,

        "predicted_class": [
            "FAKE" if p == 0 else "REAL"
            for p in raw_predictions
        ],

        "confidence": confidence,

        "decision": decisions,

        "answered": answered_mask,
    })

    test_csv_path = os.path.join(
        output_dir,
        "test_calibrated.csv"
    )

    test_df.to_csv(
        test_csv_path,
        index=False
    )

    print("\nSaved:", test_csv_path)

    # --------------------------------------------------------
    # 10I. Save threshold sweep
    # --------------------------------------------------------

    threshold_sweep.to_csv(
        os.path.join(
            output_dir,
            "threshold_sweep_val.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # 10J. Reliability plots
    # --------------------------------------------------------

    reliability_plot(
        best_test_p_fake,
        best_test_labels,
        "Test Reliability - Before Calibration",
        os.path.join(
            output_dir,
            "reliability_before.png"
        )
    )

    reliability_plot(
        test_calibrated,
        best_test_labels,
        "Test Reliability - After Calibration",
        os.path.join(
            output_dir,
            "reliability_after.png"
        )
    )

    # --------------------------------------------------------
    # 10K. Accuracy-coverage plot
    # --------------------------------------------------------

    coverage_df = accuracy_coverage_plot(
        test_calibrated,
        best_test_labels,
        os.path.join(
            output_dir,
            "accuracy_coverage_test.png"
        )
    )

    coverage_df.to_csv(
        os.path.join(
            output_dir,
            "accuracy_coverage_test.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # 10L. Save configuration
    # --------------------------------------------------------

    config = {
        "experiment": experiment,

        "selected_seed": int(best_seed),

        "validation_macro_f1": float(best_val_f1),

        "temperature": float(temperature),

        "tau": float(tau),

        "target_accuracy": float(TARGET_ACCURACY),

        "minimum_coverage": float(MIN_COVERAGE),

        "threshold_selection_reason": reason,

        "class_mapping": {
            "0": "FAKE",
            "1": "REAL"
        },

        "probability_mapping": {
            "column_0": "P(FAKE)",
            "column_1": "P(REAL)"
        },

        "test_overall_accuracy": overall_accuracy,

        "test_coverage": coverage,

        "test_answered_accuracy": answered_accuracy,

        "test_metrics_before_calibration": test_metrics_before,

        "test_metrics_after_calibration": test_metrics_after,
    }

    config_path = os.path.join(
        output_dir,
        "calibration_config.json"
    )

    with open(
        config_path,
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    print("Saved:", config_path)

    # --------------------------------------------------------
    # 10M. Return summary
    # --------------------------------------------------------

    summary = {
        "experiment": experiment,

        "seed": best_seed,

        "temperature": temperature,

        "tau": tau,

        "test_accuracy_before": test_metrics_before["accuracy"],

        "test_accuracy_after": test_metrics_after["accuracy"],

        "test_f1_before": test_metrics_before["macro_f1"],

        "test_f1_after": test_metrics_after["macro_f1"],

        "ECE_before": test_metrics_before["ECE"],

        "ECE_after": test_metrics_after["ECE"],

        "NLL_before": test_metrics_before["NLL"],

        "NLL_after": test_metrics_after["NLL"],

        "Brier_before": test_metrics_before["Brier"],

        "Brier_after": test_metrics_after["Brier"],

        "coverage": coverage,

        "answered_accuracy": answered_accuracy,

        "uncertain_count": int(np.sum(~answered_mask)),
    }

    return summary


# ============================================================
# 11. RUN ALL PART 7 EXPERIMENTS
# ============================================================

if __name__ == "__main__":

    all_summaries = []

    for experiment in EXPERIMENTS:

        summary = process_experiment(
            experiment
        )

        if summary is not None:

            all_summaries.append(summary)

    # --------------------------------------------------------
    # Save comparison table
    # --------------------------------------------------------

    if len(all_summaries) > 0:

        final_df = pd.DataFrame(
            all_summaries
        )

        comparison_path = os.path.join(
            "results",
            "calibration",
            "calibration_comparison.csv"
        )

        os.makedirs(
            os.path.dirname(comparison_path),
            exist_ok=True
        )

        final_df.to_csv(
            comparison_path,
            index=False
        )

        print("\n" + "=" * 65)
        print("PART 8 COMPLETE")
        print("=" * 65)

        print(final_df.to_string(index=False))

        print("\nComparison saved to:", comparison_path)

    else:

        print("\nNo experiments were successfully calibrated.")