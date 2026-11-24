"""
Part 9b: Caption-image MISMATCH head -> calibrated P(mismatch) in [0, 1].

Run (after Part 5, same folder as dataset_builder.py is fine):
    python mismatch.py --feat_dir features --out results/mismatch

Idea
  Training labels are synthetic: for each original (caption, image) pair we make a
  mismatched version by giving the image a caption taken from ANOTHER sample of the
  SAME split (random partner, or a "hard" partner with a similar caption).
    consistent = 0, mismatched = 1
  Original pairs labelled 'False connection' (label6 = 4) are EXCLUDED from the
  positives, because they are themselves mismatched; they are kept as a real-data check.
  Partners never cross splits, so there is no train/val/test leakage.

Model: small MLP on CLIP text/image embeddings (from Part 5):
  [t, i, t*i, |t-i|, cos(t,i)] -> logit.  Temperature scaling is fitted on VALIDATION pairs.

Reported
  - synthetic pairs (random and hard negatives): AUC, accuracy, ECE vs raw CLIP similarity
  - REAL check on original test samples: mean P(mismatch) per label6 category and
    AUC for 'False connection' (4) vs 'True' (0)

Important: P(mismatch) is calibrated for a 50/50 mix of consistent/mismatched pairs
(the training prior). On real data the prevalence of mismatch differs, so treat it as a
calibrated score, not a real-world frequency.

For Part 10:  from mismatch import load_mismatch_model, predict_mismatch
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

SIX = {0: "True", 1: "Satire", 2: "Misleading", 3: "Imposter",
       4: "False connection", 5: "Manipulated"}
EPS = 1e-6


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------- model
class MismatchHead(nn.Module):
    def __init__(self, d_in, hidden=256, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)            # logit of P(mismatch)


def pair_feats(t, i):
    return torch.cat([t, i, t * i, (t - i).abs(), (t * i).sum(-1, keepdim=True)], 1)


@torch.no_grad()
def logits_of(model, t, i, bs=1024):
    model.eval()
    return torch.cat([model(pair_feats(t[s:s + bs], i[s:s + bs]))
                      for s in range(0, len(t), bs)]).numpy()


def load_mismatch_model(path):
    ck = torch.load(path, map_location="cpu")
    m = MismatchHead(ck["d_in"], ck["hidden"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, float(ck["T"])


@torch.no_grad()
def predict_mismatch(model, T, clip_text, clip_image):
    """clip_text, clip_image: [N,512] L2-normalised CLIP embeddings (Part 5).
    Returns numpy array of calibrated P(mismatch) in [0,1]."""
    return sigmoid(logits_of(model, clip_text, clip_image) / T)


# --------------------------------------------------------------- pair maker
def neg_partner(ct, idx, rng, hard_frac, k):
    """For each i in idx choose a partner j != i from idx (same split).
    hard_frac of them are drawn among the k captions most similar to i's caption."""
    n = len(idx)
    pos = (np.arange(n) + rng.integers(1, n, n)) % n            # random, never self
    if hard_frac > 0:
        c = ct[idx]
        sim = c @ c.T
        sim.fill_diagonal_(-1e9)
        top = sim.topk(min(k, n - 1), dim=1).indices.numpy()
        hard = rng.random(n) < hard_frac
        pick = top[np.arange(n), rng.integers(0, top.shape[1], n)]
        pos = np.where(hard, pick, pos)
    return idx[pos]


def eval_pairs(F, idx, seed, hard, k):
    rng = np.random.default_rng(seed)
    j = neg_partner(F["clip_text"], idx, rng, 1.0 if hard else 0.0, k)
    t = torch.cat([F["clip_text"][idx], F["clip_text"][j]])
    i = torch.cat([F["clip_image"][idx], F["clip_image"][idx]])
    y = np.concatenate([np.zeros(len(idx)), np.ones(len(idx))])
    return t, i, y


# ------------------------------------------------------------ calibration
def fit_T(z, y):
    best_T, best = 1.0, 1e9
    for T in np.exp(np.linspace(np.log(0.1), np.log(10), 300)):
        p = np.clip(sigmoid(z / T), EPS, 1 - EPS)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if loss < best:
            best, best_T = loss, T
    return float(best_T)


def ece_bin(p, y, n_bins=10):
    conf, correct = np.maximum(p, 1 - p), ((p >= 0.5) == (y == 1)).astype(float)
    bins, out = np.linspace(0.5, 1.0, n_bins + 1), 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def pair_metrics(p, y, sim):
    m = {"auc": float(roc_auc_score(y, p)),
         "accuracy": float(accuracy_score(y, p >= 0.5)),
         "ece": ece_bin(p, y),
         "clip_sim_auc": float(roc_auc_score(y, -sim))}
    return m


# ---------------------------------------------------------------- training
def train_one(F, idx_tr, val, seed, a):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MismatchHead(2049, a.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    ct, ci = F["clip_text"], F["clip_image"]
    vt, vi, vy = val
    best, best_state, bad = -1.0, None, 0
    n = len(idx_tr)
    for _ in range(a.epochs):
        j = neg_partner(ct, idx_tr, rng, a.hard_frac, a.hard_k)   # fresh negatives
        T_all = torch.cat([ct[idx_tr], ct[j]])
        I_all = torch.cat([ci[idx_tr], ci[idx_tr]])
        Y = torch.cat([torch.zeros(n), torch.ones(n)])
        perm = torch.randperm(2 * n)
        model.train()
        for s in range(0, 2 * n, a.batch_size):
            b = perm[s:s + a.batch_size]
            loss = loss_fn(model(pair_feats(T_all[b], I_all[b])), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        auc = roc_auc_score(vy, logits_of(model, vt, vi))
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return model, best


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default="features")
    ap.add_argument("--out", default="results/mismatch")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--hard_frac", type=float, default=0.3,
                    help="fraction of training negatives that are 'hard' (similar caption)")
    ap.add_argument("--hard_k", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    F = {s: torch.load(os.path.join(a.feat_dir, f"{s}.pt"))
         for s in ["train", "validate", "test"]}
    elig = {s: np.where(F[s]["label6"].numpy() != 4)[0] for s in F}   # drop 'False connection'
    for s in F:
        print(f"[{s}] {len(F[s]['label'])} samples, {len(elig[s])} eligible as consistent pairs "
              f"({int((F[s]['label6'] == 4).sum())} 'False connection' excluded)")

    val_r, val_h = (eval_pairs(F["validate"], elig["validate"], 1, h, a.hard_k) for h in (False, True))
    te_r, te_h = (eval_pairs(F["test"], elig["test"], 2, h, a.hard_k) for h in (False, True))

    runs = []
    for seed in range(a.seeds):
        model, vauc = train_one(F["train"], elig["train"], val_r, seed, a)
        z = logits_of(model, *te_r[:2])
        runs.append({"seed": seed, "val_auc": vauc, "model": model,
                     "test_auc": float(roc_auc_score(te_r[2], z))})
        print(f"seed {seed}: val AUC {vauc:.3f} | test AUC (random pairs) {runs[-1]['test_auc']:.3f}")
    best = max(runs, key=lambda r: r["val_auc"])                    # chosen on VAL only
    model = best["model"]

    # temperature on validation pairs (random + hard)
    zv = np.concatenate([logits_of(model, *val_r[:2]), logits_of(model, *val_h[:2])])
    yv = np.concatenate([val_r[2], val_h[2]])
    T = fit_T(zv, yv)
    print(f"temperature T = {T:.3f}")

    res = {"temperature": T, "selected_seed": best["seed"],
           "seeds_test_auc_mean": float(np.mean([r["test_auc"] for r in runs])),
           "seeds_test_auc_std": float(np.std([r["test_auc"] for r in runs]))}
    probs = {}
    for name, (t, i, y) in [("random_negatives", te_r), ("hard_negatives", te_h)]:
        p = sigmoid(logits_of(model, t, i) / T)
        sim = (t * i).sum(-1).numpy()
        res[name] = pair_metrics(p, y, sim)
        probs[name] = (p, y, sim)
        print(f"test [{name}] AUC {res[name]['auc']:.3f} (raw CLIP-sim AUC {res[name]['clip_sim_auc']:.3f}) "
              f"| acc {res[name]['accuracy']:.3f} | ECE {res[name]['ece']:.3f}")

    # raw CLIP-similarity threshold baseline (threshold fitted on validation random pairs)
    sv = (val_r[0] * val_r[1]).sum(-1).numpy()
    cands = np.unique(sv)
    thr = float(cands[np.argmax([accuracy_score(val_r[2], sv < c) for c in cands])])
    st = probs["random_negatives"][2]
    res["clip_sim_baseline"] = {"threshold": thr,
                                "test_accuracy_random": float(accuracy_score(te_r[2], st < thr))}
    print(f"raw CLIP-sim threshold baseline accuracy {res['clip_sim_baseline']['test_accuracy_random']:.3f}")

    # ---- REAL-data check on original test samples
    Ft = F["test"]
    p_real = sigmoid(logits_of(model, Ft["clip_text"], Ft["clip_image"]) / T)
    sim_real = (Ft["clip_text"] * Ft["clip_image"]).sum(-1).numpy()
    l6 = Ft["label6"].numpy()
    rows = []
    for k_, name in SIX.items():
        m = l6 == k_
        rows.append({"label6": k_, "name": name, "n": int(m.sum()),
                     "mean_P_mismatch": float(p_real[m].mean()) if m.any() else np.nan,
                     "mean_clip_sim": float(sim_real[m].mean()) if m.any() else np.nan,
                     "frac_P_gt_0.5": float((p_real[m] > 0.5).mean()) if m.any() else np.nan})
    cat = pd.DataFrame(rows)
    cat.to_csv(os.path.join(a.out, "mismatch_by_label6_test.csv"), index=False)
    n4, n0 = int((l6 == 4).sum()), int((l6 == 0).sum())
    if n4 >= 5 and n0 >= 5:
        sel = (l6 == 4) | (l6 == 0)
        res["real_check_false_connection_vs_true"] = {
            "n_false_connection": n4, "n_true": n0,
            "auc_model": float(roc_auc_score((l6[sel] == 4).astype(int), p_real[sel])),
            "auc_clip_sim": float(roc_auc_score((l6[sel] == 4).astype(int), -sim_real[sel]))}
    else:
        res["real_check_false_connection_vs_true"] = (
            f"too few samples (False connection={n4}, True={n0}); need >= 5 each. "
            "Use a larger --per_class in Part 1.")
    print("\nMean P(mismatch) by Fakeddit category (test, original pairs):")
    print(cat.to_string(index=False))
    print("Real-data check:", res["real_check_false_connection_vs_true"])

    pd.DataFrame({"id": Ft["ids"], "label6": l6, "p_mismatch": p_real,
                  "clip_sim": sim_real}).to_csv(
        os.path.join(a.out, "test_p_mismatch.csv"), index=False)

    # ---- plots
    p, y, sim = probs["random_negatives"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].hist(p[y == 0], bins=20, alpha=0.7, label="consistent")
    ax[0].hist(p[y == 1], bins=20, alpha=0.7, label="mismatched")
    ax[0].set_xlabel("P(mismatch)"); ax[0].set_title("Test pairs (random negatives)"); ax[0].legend()
    for nm, sc in [("mismatch head", p), ("raw CLIP similarity", -sim)]:
        fpr, tpr, _ = roc_curve(y, sc)
        ax[1].plot(fpr, tpr, label=nm)
    ax[1].plot([0, 1], [0, 1], "k:", lw=0.8); ax[1].legend()
    ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].set_title("ROC")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "mismatch_eval.png"), dpi=150); plt.close(fig)

    torch.save({"state_dict": model.state_dict(), "T": T, "hidden": a.hidden, "d_in": 2049},
               os.path.join(a.out, "mismatch_model.pt"))
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved to", a.out)


if __name__ == "__main__":
    main()