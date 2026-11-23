"""
Part 9: Experiments 1, 2 and 4 - comparison tables, plots and significance tests.

Run after Parts 3, 4, 6, 7 (same folder is fine; it only reads CSVs):
    python part9_experiments.py --root results --out results/experiments

Reads <root>/<model>/{val,test}_predictions.csv for every model that exists
(missing ones are skipped with a message). Nothing is retrained.

Experiments covered
  Exp 1  single-modality baselines vs the best multimodal model
  Exp 2  fusion strategies compared (concat, concat+sim, cross-attn, late)
  Exp 4  modality ablation: full model vs removing text / removing image
         (frozen-feature family, so the comparison is like-for-like)

"Best" models are chosen by VALIDATION macro-F1, never by test.
Test results get 95% bootstrap confidence intervals and McNemar tests,
because small test sets make raw accuracy differences easy to over-read.

Outputs in --out: exp1_*.csv, exp2_*.csv, exp4_*.csv, significance.json,
disagreement_cases.csv, report.md, roc_curves.png, f1_comparison.png,
confusion_matrices.png, all_models_summary.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)

SCORE_COLS = ["fusion_score", "text_score", "image_score"]

# (display name, folder relative to --root, group)
MODELS = [
    ("Text (fine-tuned DistilBERT)", "text_only", "single"),
    ("Image (fine-tuned ResNet)", "image_only", "single"),
    ("Text (frozen + MLP)", "fusion_concat/text_only", "single"),
    ("Image (frozen CLIP + MLP)", "fusion_concat/image_only", "single"),
    ("Concat (BERT + CLIP)", "fusion_concat/concat", "fusion"),
    ("Concat + CLIP-sim", "fusion_concat/concat_sim", "fusion"),
    ("Cross-attention", "fusion_crossattn/crossattn", "fusion"),
    ("Cross-attention + CLIP-sim", "fusion_crossattn/crossattn_sim", "fusion"),
    ("Late fusion (decision-level)", "fusion_crossattn/late_fusion", "fusion"),
]


# ----------------------------------------------------------------- loading
def read_preds(folder, split):
    p = os.path.join(folder, f"{split}_predictions.csv")
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p, dtype={"id": str})
    col = next((c for c in SCORE_COLS if c in df.columns and
                pd.to_numeric(df[c], errors="coerce").notna().all()), None)
    return {
        "df": df, "ids": df["id"].values,
        "y": (df["true_label"] == "REAL").astype(int).values,          # 1 = REAL
        "pred": (df["pred_label"] == "REAL").astype(int).values,
        "pf": pd.to_numeric(df[col]).values if col else None,          # P(FAKE)
    }


def metrics(d):
    y, pred, pf = d["y"], d["pred"], d["pf"]
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    m = {"accuracy": accuracy_score(y, pred), "precision": p.mean(), "recall": r.mean(),
         "macro_f1": f.mean(), "f1_FAKE": f[0], "f1_REAL": f[1],
         "roc_auc": np.nan, "pr_auc": np.nan}
    if pf is not None and len(set(y)) == 2:
        m["roc_auc"] = roc_auc_score(y, 1 - pf)                         # REAL = positive
        m["pr_auc"] = average_precision_score(y, 1 - pf)
    return m


def bootstrap_ci(d, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    y, pred, n = d["y"], d["pred"], len(d["y"])
    accs, f1s = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        accs.append(accuracy_score(y[i], pred[i]))
        f1s.append(f1_score(y[i], pred[i], average="macro", labels=[0, 1]))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"acc_ci": q(accs), "f1_ci": q(f1s)}


def mcnemar(a, b):
    """a, b: dicts on the same ids. Exact McNemar on correctness."""
    ca, cb = a["pred"] == a["y"], b["pred"] == b["y"]
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = only_a + only_b
    p = float(binomtest(only_a, n, 0.5).pvalue) if n > 0 else 1.0
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_value": p}


def align(d, ids):
    order = pd.Series(range(len(d["ids"])), index=d["ids"]).loc[ids].values
    out = {k: (v[order] if isinstance(v, np.ndarray) else v) for k, v in d.items() if k != "df"}
    out["df"] = d["df"].iloc[order].reset_index(drop=True)
    return out


# ------------------------------------------------------------------ output
def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def fmt(x):
    return "-" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.3f}"


def row_for(name, group, t, v, ci):
    return {"model": name, "group": group,
            "accuracy": f"{fmt(t['accuracy'])} [{ci['acc_ci'][0]:.3f}, {ci['acc_ci'][1]:.3f}]",
            "precision": fmt(t["precision"]), "recall": fmt(t["recall"]),
            "macro_f1": f"{fmt(t['macro_f1'])} [{ci['f1_ci'][0]:.3f}, {ci['f1_ci'][1]:.3f}]",
            "f1_FAKE": fmt(t["f1_FAKE"]), "f1_REAL": fmt(t["f1_REAL"]),
            "roc_auc": fmt(t["roc_auc"]), "pr_auc": fmt(t["pr_auc"]),
            "val_macro_f1": fmt(v["macro_f1"])}


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="results/experiments")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    loaded = {}
    for name, rel, group in MODELS:
        folder = os.path.join(a.root, rel)
        t, v = read_preds(folder, "test"), read_preds(folder, "val")
        if t is None or v is None:
            print(f"skip: {name}  (no val/test predictions in {folder})")
            continue
        loaded[name] = {"group": group, "test": t, "val": v}
    if len(loaded) < 2:
        raise SystemExit("Need at least two models with predictions. Run Parts 3-7 first.")

    # align all models on the same test ids (and val ids)
    for split in ["test", "val"]:
        common = None
        for m in loaded.values():
            s = set(m[split]["ids"])
            common = s if common is None else common & s
        ids = sorted(common)
        for m in loaded.values():
            m[split] = align(m[split], ids)
    n_test = len(next(iter(loaded.values()))["test"]["y"])
    print(f"{len(loaded)} models, {n_test} common test samples")

    rows, summary = [], {}
    for name, m in loaded.items():
        t, v = metrics(m["test"]), metrics(m["val"])
        ci = bootstrap_ci(m["test"], a.n_boot)
        m["t"], m["v"], m["ci"] = t, v, ci
        rows.append(row_for(name, m["group"], t, v, ci))
        summary[name] = {"group": m["group"], "test": t, "val": v, "ci": ci}
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(a.out, "all_models.csv"), index=False)

    pick = lambda g: max((n for n, m in loaded.items() if m["group"] == g),
                         key=lambda n: loaded[n]["v"]["macro_f1"], default=None)
    best_single, best_fusion = pick("single"), pick("fusion")
    print(f"best single (by val macro-F1): {best_single}")
    print(f"best multimodal (by val macro-F1): {best_fusion}")

    # ---- Exp 1: baselines vs multimodal
    e1_names = [n for n, m in loaded.items() if m["group"] == "single"] + \
               ([best_fusion] if best_fusion else [])
    exp1 = table[table["model"].isin(e1_names)]
    exp1.to_csv(os.path.join(a.out, "exp1_baselines_vs_multimodal.csv"), index=False)

    # ---- Exp 2: fusion strategies
    exp2 = table[table["group"] == "fusion"]
    exp2.to_csv(os.path.join(a.out, "exp2_fusion_comparison.csv"), index=False)

    # ---- Exp 4: modality ablation (frozen-feature family)
    abl_spec = [("Full: text + image", "Concat (BERT + CLIP)"),
                ("Remove image (text only)", "Text (frozen + MLP)"),
                ("Remove text (image only)", "Image (frozen CLIP + MLP)"),
                ("Full + CLIP-sim", "Concat + CLIP-sim")]
    full = loaded.get("Concat (BERT + CLIP)")
    abl_rows = []
    for label, mn in abl_spec:
        if mn not in loaded:
            continue
        t = loaded[mn]["t"]
        d = "-" if full is None else f"{t['macro_f1'] - full['t']['macro_f1']:+.3f}"
        abl_rows.append({"setting": label, "model": mn, "accuracy": fmt(t["accuracy"]),
                         "macro_f1": fmt(t["macro_f1"]), "roc_auc": fmt(t["roc_auc"]),
                         "delta_macro_f1_vs_full": d})
    exp4 = pd.DataFrame(abl_rows)
    exp4.to_csv(os.path.join(a.out, "exp4_modality_ablation.csv"), index=False)

    # ---- significance: best multimodal vs every single-modality model
    sig = {}
    if best_fusion:
        for n, m in loaded.items():
            if m["group"] == "single":
                sig[f"{best_fusion} vs {n}"] = {
                    **mcnemar(loaded[best_fusion]["test"], m["test"]),
                    "macro_f1_diff": float(loaded[best_fusion]["t"]["macro_f1"] - m["t"]["macro_f1"])}
    with open(os.path.join(a.out, "significance.json"), "w") as f:
        json.dump(sig, f, indent=2)

    # ---- complementarity: where fusion fixes / breaks single-modality errors
    comp = {}
    if best_fusion and best_single:
        bf, bs = loaded[best_fusion]["test"], loaded[best_single]["test"]
        cf, cs = bf["pred"] == bf["y"], bs["pred"] == bs["y"]
        comp = {"best_single": best_single, "best_multimodal": best_fusion,
                "multimodal_fixes_single_errors": int((cf & ~cs).sum()),
                "multimodal_breaks_single_correct": int((~cf & cs).sum()),
                "both_correct": int((cf & cs).sum()), "both_wrong": int((~cf & ~cs).sum())}

    # ---- disagreement cases (feeds Experiment 5 error analysis)
    t_n, i_n = "Text (frozen + MLP)", "Image (frozen CLIP + MLP)"
    if best_fusion and t_n in loaded and i_n in loaded:
        T, I, Fm = loaded[t_n]["test"], loaded[i_n]["test"], loaded[best_fusion]["test"]
        dis = pd.DataFrame({
            "id": T["ids"], "text": T["df"]["text"], "image_path": T["df"]["image_path"],
            "true_label": T["df"]["true_label"],
            "text_pred": T["df"]["pred_label"], "image_pred": I["df"]["pred_label"],
            "fusion_pred": Fm["df"]["pred_label"],
            "text_P_fake": T["pf"], "image_P_fake": I["pf"], "fusion_P_fake": Fm["pf"],
            "fusion_correct": Fm["pred"] == Fm["y"]})
        if "label6" in T["df"].columns:
            dis["label6"] = T["df"]["label6"]
        dis = dis[dis["text_pred"] != dis["image_pred"]]
        dis.to_csv(os.path.join(a.out, "disagreement_cases.csv"), index=False)
        comp["text_image_disagreements"] = int(len(dis))
        comp["fusion_correct_on_disagreements"] = int(dis["fusion_correct"].sum())

    summary["_complementarity"] = comp
    summary["_selection"] = {"best_single_by_val": best_single, "best_multimodal_by_val": best_fusion}
    with open(os.path.join(a.out, "all_models_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---------------------------------------------------------------- plots
    names = list(loaded)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(names))
    f1v = [loaded[n]["t"]["macro_f1"] for n in names]
    err = [[loaded[n]["t"]["macro_f1"] - loaded[n]["ci"]["f1_ci"][0] for n in names],
           [loaded[n]["ci"]["f1_ci"][1] - loaded[n]["t"]["macro_f1"] for n in names]]
    colors = ["#888" if loaded[n]["group"] == "single" else "#2a6fdb" for n in names]
    ax.bar(xs, f1v, yerr=err, capsize=3, color=colors)
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("test macro-F1 (95% bootstrap CI)"); ax.set_ylim(0, 1)
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.set_title("Single-modality (grey) vs multimodal (blue)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "f1_comparison.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for n in names:
        d = loaded[n]["test"]
        if d["pf"] is not None and len(set(d["y"])) == 2:
            fpr, tpr, _ = roc_curve(d["y"], 1 - d["pf"])
            ax.plot(fpr, tpr, label=f"{n} ({loaded[n]['t']['roc_auc']:.2f})",
                    ls="-" if loaded[n]["group"] == "fusion" else "--")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves (test)"); ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "roc_curves.png"), dpi=150); plt.close(fig)

    cols = 3
    rws = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(3.2 * cols, 3.2 * rws))
    for ax, n in zip(np.array(axes).flatten(), names):
        d = loaded[n]["test"]
        cm = confusion_matrix(d["y"], d["pred"], labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["FAKE", "REAL"], fontsize=7); ax.set_yticklabels(["FAKE", "REAL"], fontsize=7)
        ax.set_title(n, fontsize=7)
    for ax in np.array(axes).flatten()[len(names):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "confusion_matrices.png"), dpi=150); plt.close(fig)

    # ---------------------------------------------------------- report.md
    parts = [f"# Experiment results (test set, n = {n_test})\n",
             "Values in brackets are 95% bootstrap confidence intervals. "
             "Best models were selected on validation macro-F1.\n",
             "## Experiment 1: single-modality baselines vs multimodal\n", md_table(exp1), "",
             "## Experiment 2: fusion strategies\n", md_table(exp2), "",
             "## Experiment 4: modality ablation\n", md_table(exp4), ""]
    if sig:
        parts += ["## Significance (exact McNemar test, best multimodal vs each single-modality)\n",
                  md_table(pd.DataFrame([{"comparison": k, "only multimodal correct": v["only_first_correct"],
                                          "only single correct": v["only_second_correct"],
                                          "macro-F1 diff": f"{v['macro_f1_diff']:+.3f}",
                                          "p-value": f"{v['p_value']:.3f}"} for k, v in sig.items()])), ""]
    if comp:
        parts += ["## Complementarity\n", "```", json.dumps(comp, indent=2), "```", ""]
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(parts))

    # ------------------------------------------------------- console summary
    print("\n=== TEST macro-F1 (95% CI) ===")
    for n in names:
        m = loaded[n]
        print(f"{n:32s} {m['t']['macro_f1']:.3f} [{m['ci']['f1_ci'][0]:.3f}, {m['ci']['f1_ci'][1]:.3f}]"
              f"  acc {m['t']['accuracy']:.3f}  ({m['group']})")
    for k, v in sig.items():
        verdict = "significant" if v["p_value"] < 0.05 else "NOT significant"
        print(f"McNemar: {k}: p={v['p_value']:.3f} ({verdict} at 0.05)")
    if comp:
        print("Complementarity:", comp)
    print("\nSaved to", a.out)


if __name__ == "__main__":
    main()