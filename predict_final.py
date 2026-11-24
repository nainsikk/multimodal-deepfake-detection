"""
predict_final.py - end-to-end inference for the multimodal misinformation detector.

SELF-CONTAINED: imports nothing from your other project scripts (so importing it never
runs calibration.py etc.). It rebuilds the exact model classes used in your Part 7 and
reads your saved files:

    results/fusion_crossattn/<exp>/seed_<S>/model.pt        (crossattn, crossattn_sim)
    results/fusion_crossattn/late_fusion/{text,image}/seed_<S>/model.pt
    results/fusion_crossattn/late_fusion/seed_<S>/val_predictions.npz   (text_weight)
    results/calibration/<exp>/calibration_config.json       (temperature, tau, selected_seed)
    results/mismatch/mismatch_model.pt                      (optional)

Usage (run from the project root, venv active):
    python predict_final.py --text "some caption" --image data/test/images/10cpga.jpg
    python predict_final.py --demo data/test/test_clean.csv --out results/predict_demo.json
    python predict_final.py ... --final crossattn      # default: auto = best VALIDATION macro-F1

From code (e.g. the backend):
    from predict_final import predict
    result = predict(text, image_path)                         # text + image
    result = predict(text, image_path, modalities=["text"])    # ablation

Returned JSON:
{
  "prediction": "REAL" | "FAKE" | "UNCERTAIN",
  "confidence": float,              # calibrated confidence in the predicted class
  "manipulation_type": str,         # heuristic wording, NOT forensic detection
  "modality_scores": {"text": P(FAKE), "image": P(FAKE), "fusion": P(FAKE)},
  "evidence": {"text_tokens": [{"token","weight"}], "image_heatmap_url": str|None,
               "mismatch_score": P(mismatch)|None},
  "explanation": str,
  "p_fake": float, "meta": {...}
}
Evidence uses OCCLUSION: remove each word / grey out each cell of a 7x7 image grid and
measure the drop in the predicted class probability.
"""
import argparse
import json
import os
import uuid

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

# ------------------------------------------------------------------ paths
RESULTS = "results_small"
CROSS_DIR = os.path.join(RESULTS, "fusion_crossattn")
CALIB_DIR = os.path.join(RESULTS, "calibration")
MISMATCH_PATH = os.path.join(RESULTS, "mismatch", "mismatch_model.pt")
HEATMAP_DIR = "static/heatmaps"
FINAL_CHOICES = ["crossattn", "crossattn_sim", "late_fusion"]

BERT_NAME = "distilbert-base-uncased"
CLIP_NAME = "openai/clip-vit-base-patch32"
MAX_LEN = 64
EPS = 1e-7
GRID = 7
MAX_WORDS = 40
MISMATCH_FLAG = 0.5
SCORE_GAP = 0.15


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


# ------------------------------------------- model classes (same as Part 7)
class CrossAttentionFusion(nn.Module):
    def __init__(self, input_dim=768, d_model=256, num_heads=4, dropout=0.2,
                 use_similarity=False):
        super().__init__()
        self.use_similarity = use_similarity
        self.text_projection = nn.Linear(input_dim, d_model)
        self.image_projection = nn.Linear(input_dim, d_model)
        self.text_to_image = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.image_to_text = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.text_norm = nn.LayerNorm(d_model)
        self.image_norm = nn.LayerNorm(d_model)
        classifier_input = d_model * 2 + (1 if use_similarity else 0)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 2))

    def forward(self, text_tokens, text_mask, image_patches, clip_sim=None):
        text = self.text_projection(text_tokens)
        image = self.image_projection(image_patches)
        pad = text_mask == 0
        text_att, _ = self.text_to_image(query=text, key=image, value=image,
                                         need_weights=False)
        image_att, _ = self.image_to_text(query=image, key=text, value=text,
                                          key_padding_mask=pad, need_weights=False)
        text = self.text_norm(text + text_att)
        image = self.image_norm(image + image_att)
        m = text_mask.unsqueeze(-1).float()
        pooled_text = (text * m).sum(1) / m.sum(1).clamp(min=1.0)
        pooled_image = image.mean(1)
        combined = torch.cat([pooled_text, pooled_image], 1)
        if self.use_similarity:
            combined = torch.cat([combined, clip_sim.reshape(-1, 1).float()], 1)
        return self.classifier(combined)


class LateFusionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 2))

    def forward(self, x):
        return self.net(x)


class MismatchHead(nn.Module):               # same as mismatch.py (optional)
    def __init__(self, d_in, hidden=256, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pair_feats(t, i):
    return torch.cat([t, i, t * i, (t - i).abs(), (t * i).sum(-1, keepdim=True)], 1)


# ------------------------------------------------ feature extraction (Part 5)
class FeatureExtractor:
    """Identical recipe to feature_extraction.py (frozen DistilBERT + CLIP)."""

    def __init__(self):
        self.device = get_device()
        self.bert_tok = AutoTokenizer.from_pretrained(BERT_NAME)
        self.bert = AutoModel.from_pretrained(BERT_NAME).to(self.device).eval()
        self.clip = CLIPModel.from_pretrained(CLIP_NAME).to(self.device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(CLIP_NAME)

    @torch.no_grad()
    def extract(self, texts, images, bs=32):
        keys = ["bert_text", "clip_text", "clip_image", "clip_sim",
                "text_tokens", "text_mask", "img_patches"]
        out = {k: [] for k in keys}
        for s in range(0, len(texts), bs):
            tx, im = texts[s:s + bs], images[s:s + bs]
            t = self.bert_tok(tx, padding="max_length", truncation=True,
                              max_length=MAX_LEN, return_tensors="pt").to(self.device)
            hs = self.bert(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).float()
            out["bert_text"].append(((hs * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu())
            out["text_tokens"].append(hs.half().float().cpu())     # fp16 round trip = training
            out["text_mask"].append(t["attention_mask"].cpu())
            ct = self.clip_proc.tokenizer(tx, padding=True, truncation=True, max_length=77,
                                          return_tensors="pt").to(self.device)
            te = F.normalize(self.clip.text_projection(
                self.clip.text_model(**ct).pooler_output), dim=-1)
            px = self.clip_proc.image_processor(im, return_tensors="pt")["pixel_values"].to(self.device)
            vo = self.clip.vision_model(pixel_values=px)
            ie = F.normalize(self.clip.visual_projection(vo.pooler_output), dim=-1)
            out["img_patches"].append(vo.last_hidden_state.half().float().cpu())
            out["clip_text"].append(te.cpu())
            out["clip_image"].append(ie.cpu())
            out["clip_sim"].append((te * ie).sum(-1).cpu())
        return {k: torch.cat(v) for k, v in out.items()}


# -------------------------------------------------------------------- scorers
# Every scorer maps a feature batch to a numpy array of P(FAKE).
class CrossScorer:
    def __init__(self, exp, seed):
        ck = _load(os.path.join(CROSS_DIR, exp, f"seed_{seed}", "model.pt"))
        self.use_sim = bool(ck.get("use_similarity", False))
        self.model = CrossAttentionFusion(768, 256, 4, 0.2, self.use_sim)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def p_fake(self, f):
        logits = self.model(f["text_tokens"].float(), f["text_mask"].long(),
                            f["img_patches"].float(),
                            f["clip_sim"].float() if self.use_sim else None)
        return torch.softmax(logits, -1)[:, 0].numpy()


class ModalityScorer:
    """Text-only or image-only MLP from the late-fusion experiment."""

    def __init__(self, modality, seed):
        ck = _load(os.path.join(CROSS_DIR, "late_fusion", modality, f"seed_{seed}", "model.pt"))
        self.key = "bert_text" if modality == "text" else "clip_image"
        self.mean, self.std = ck["mean"].float(), ck["std"].float()
        self.model = LateFusionMLP(int(ck["input_dim"]))
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def p_fake(self, f):
        x = (f[self.key].float() - self.mean) / self.std
        return torch.softmax(self.model(x), -1)[:, 0].numpy()


class LateScorer:
    def __init__(self, text_scorer, image_scorer, text_weight):
        self.t, self.i, self.w = text_scorer, image_scorer, float(text_weight)

    def p_fake(self, f):
        return self.w * self.t.p_fake(f) + (1 - self.w) * self.i.p_fake(f)


# ---------------------------------------------------- calibration helpers
def calibrate_pf(pf, T):
    pf = np.clip(np.asarray(pf, dtype=float), EPS, 1 - EPS)
    return 1 / (1 + np.exp(-np.log(pf / (1 - pf)) / T))


def decide(pf, T, tau):
    pc = float(calibrate_pf(pf, T))
    conf = max(pc, 1 - pc)
    if conf < tau:
        return "UNCERTAIN", conf
    return ("FAKE" if pc >= 0.5 else "REAL"), conf


def read_calibration(exp):
    p = os.path.join(CALIB_DIR, exp, "calibration_config.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


# ----------------------------------------------------- evidence helpers
def support(base_pf, new_pf):
    sign = 1.0 if base_pf >= 0.5 else -1.0
    return sign * (base_pf - np.asarray(new_pf))


def normalise(x):
    x = np.clip(np.asarray(x, dtype=float), 0, None)
    return x / x.max() if x.max() > 0 else x


def overlay_heatmap(img224, heat, path, alpha=0.5):
    h = Image.fromarray((heat * 255).astype(np.uint8)).resize(img224.size, Image.BICUBIC)
    rgb = matplotlib.colormaps["jet"](np.asarray(h) / 255.0)[..., :3]
    base = np.asarray(img224).astype(float) / 255.0
    Image.fromarray((((1 - alpha) * base + alpha * rgb) * 255).astype(np.uint8)).save(path)


def manipulation_type(label, mismatch, p_text, p_image):
    """Heuristic wording only: the system does NOT inspect pixels for edits / AI generation."""
    if label == "REAL":
        return "none detected"
    if label == "UNCERTAIN":
        return "undetermined"
    if mismatch is not None and mismatch >= MISMATCH_FLAG:
        return "caption-image mismatch"
    if p_text is not None and p_image is not None:
        if p_text - p_image > SCORE_GAP:
            return "misleading text"
        if p_image - p_text > SCORE_GAP:
            return "suspicious image content"
    return "general misinformation (text + image)"


def generate_explanation(label, conf, p_fake, mismatch, tokens, tau):
    words = ", ".join(t["token"] for t in tokens[:3]) if tokens else ""
    if mismatch is None:
        m = ""
    elif mismatch >= MISMATCH_FLAG:
        m = f"The caption and image look unrelated (mismatch probability {mismatch:.2f})."
    else:
        m = f"The caption and image appear to fit each other (mismatch probability {mismatch:.2f})."
    if label == "REAL":
        s = f"{m} The model found no strong signs of misinformation (P(fake) = {p_fake:.2f})."
    elif label == "FAKE":
        s = f"{m} The model judges this content likely to be misinformation (P(fake) = {p_fake:.2f})."
        if words:
            s += f" Words that most influenced the decision: {words}."
    else:
        s = (f"The model is not confident enough to decide (confidence {conf:.2f} is below the "
             f"{tau:.2f} threshold; P(fake) = {p_fake:.2f}). {m} Human review is recommended.")
    return " ".join(s.split())


# ------------------------------------------------------------- predictor
class Predictor:
    def __init__(self, final="auto", explain_fn=None):
        print("Loading models (first call takes a few seconds)...")
        self.explain_fn = explain_fn
        self.fx = FeatureExtractor()

        cals = {e: read_calibration(e) for e in FINAL_CHOICES}
        cals = {e: c for e, c in cals.items() if c is not None}
        if not cals:
            raise FileNotFoundError("No calibration_config.json under results/calibration/. "
                                    "Run calibration.py first.")
        if final == "auto":
            final = max(cals, key=lambda e: cals[e].get("validation_macro_f1", -1))
            print(f"final model (best validation macro-F1): {final}")
        if final not in cals:
            raise ValueError(f"{final} not available. Choose from {list(cals)}")
        self.final_name, self.cal = final, cals[final]
        self.T, self.tau = float(self.cal["temperature"]), float(self.cal["tau"])

        # text-only / image-only models (modality_scores + ablation)
        late_seed = int(cals["late_fusion"]["selected_seed"]) if "late_fusion" in cals else 42
        self.text_scorer = self._try(lambda: ModalityScorer("text", late_seed))
        self.image_scorer = self._try(lambda: ModalityScorer("image", late_seed))

        seed = int(self.cal["selected_seed"])
        if final == "late_fusion":
            w = float(np.load(os.path.join(CROSS_DIR, "late_fusion", f"seed_{seed}",
                                           "val_predictions.npz"))["text_weight"])
            if self.text_scorer is None or self.image_scorer is None:
                raise FileNotFoundError("late_fusion text/image models missing")
            self.final = LateScorer(self.text_scorer, self.image_scorer, w)
        else:
            self.final = CrossScorer(final, seed)

        self.mm, self.mm_T = None, 1.0
        try:
            ck = _load(MISMATCH_PATH)
            self.mm = MismatchHead(int(ck["d_in"]), int(ck["hidden"]))
            self.mm.load_state_dict(ck["state_dict"])
            self.mm.eval()
            self.mm_T = float(ck["T"])
        except Exception as e:
            print(f"note: mismatch model not loaded ({type(e).__name__}); mismatch_score will be null")
        os.makedirs(HEATMAP_DIR, exist_ok=True)

    @staticmethod
    def _try(fn):
        try:
            return fn()
        except Exception as e:
            print(f"warning: {type(e).__name__}: {e}")
            return None

    def _mismatch(self, feats):
        if self.mm is None:
            return None
        with torch.no_grad():
            z = self.mm(pair_feats(feats["clip_text"], feats["clip_image"]))[0].item()
        return float(1 / (1 + np.exp(-z / self.mm_T)))

    def _text_evidence(self, text, img, scorer, base_pf, topk=8):
        words = text.split()[:MAX_WORDS]
        if not words:
            return []
        if len(words) == 1:
            return [{"token": words[0], "weight": 1.0}]
        variants = [" ".join(words[:k] + words[k + 1:]) for k in range(len(words))]
        feats = self.fx.extract(variants, [img] * len(variants))
        w = normalise(support(base_pf, scorer.p_fake(feats)))
        order = np.argsort(-w)[:topk]
        return [{"token": words[i], "weight": round(float(w[i]), 4)} for i in order if w[i] > 0]

    def _image_evidence(self, text, img, scorer, base_pf):
        img224 = ImageOps.fit(img, (224, 224), Image.BICUBIC)
        fill = tuple(int(v) for v in np.asarray(img224).reshape(-1, 3).mean(0))
        cell = 224 // GRID
        masked = []
        for r in range(GRID):
            for c in range(GRID):
                m = img224.copy()
                m.paste(fill, (c * cell, r * cell,
                               (c + 1) * cell if c < GRID - 1 else 224,
                               (r + 1) * cell if r < GRID - 1 else 224))
                masked.append(m)
        feats = self.fx.extract([text] * len(masked), masked)
        heat = normalise(support(base_pf, scorer.p_fake(feats))).reshape(GRID, GRID)
        name = uuid.uuid4().hex[:12] + ".png"
        path = os.path.join(HEATMAP_DIR, name)
        overlay_heatmap(img224, heat, path)
        return "/" + path.replace(os.sep, "/")

    def predict(self, text, image_path, modalities=None, explain=True):
        text = (text or "").strip()
        mods = set(modalities) if modalities else {"text", "image"}
        if not mods or not mods <= {"text", "image"}:
            raise ValueError("modalities must be a non-empty subset of ['text', 'image']")
        if "text" in mods and not text:
            raise ValueError("text is empty")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as im:
            img = im.convert("RGB")
        both = mods == {"text", "image"}

        feats = self.fx.extract([text or " "], [img])
        p_text = float(self.text_scorer.p_fake(feats)[0]) if self.text_scorer else None
        p_image = float(self.image_scorer.p_fake(feats)[0]) if self.image_scorer else None
        if both:
            scorer, T, calibrated = self.final, self.T, True
        else:
            scorer = self.text_scorer if mods == {"text"} else self.image_scorer
            if scorer is None:
                raise FileNotFoundError("single-modality model missing for this ablation")
            T, calibrated = 1.0, False        # final-model calibration does not apply
        p_fusion = float(scorer.p_fake(feats)[0])
        label, conf = decide(p_fusion, T, self.tau)
        mismatch = self._mismatch(feats) if both else None

        tokens, heat_url = [], None
        if explain:
            if self.explain_fn is not None:
                ex = self.explain_fn(text, image_path)
                tokens, heat_url = ex.get("text_tokens", []), ex.get("heatmap_path")
            else:
                if "text" in mods:
                    tokens = self._text_evidence(text, img, scorer, p_fusion)
                if "image" in mods:
                    heat_url = self._image_evidence(text or " ", img, scorer, p_fusion)

        r4 = lambda v: None if v is None else round(float(v), 4)
        return {
            "prediction": label,
            "confidence": r4(conf),
            "manipulation_type": manipulation_type(label, mismatch, p_text, p_image),
            "modality_scores": {"text": r4(p_text), "image": r4(p_image), "fusion": r4(p_fusion)},
            "evidence": {"text_tokens": tokens, "image_heatmap_url": heat_url,
                         "mismatch_score": r4(mismatch)},
            "explanation": generate_explanation(label, conf, p_fusion, mismatch, tokens, self.tau),
            "p_fake": r4(p_fusion),
            "meta": {"modalities_used": sorted(mods), "calibrated": calibrated,
                     "temperature": T, "uncertain_threshold": self.tau,
                     "final_model": self.final_name},
        }


_DEFAULT = None


def predict(text, image_path, modalities=None, explain=True):
    """Convenience wrapper for the backend: models load once, on first call."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Predictor("auto")
    return _DEFAULT.predict(text, image_path, modalities, explain)


# --------------------------------------------------------------------- CLI
def run_demo(pred, csv_path):
    df = pd.read_csv(csv_path)
    real = df[df["label_name"] == "REAL"].iloc[0]
    fake = df[df["label_name"] == "FAKE"].iloc[0]
    cases = [("REAL pair (true label REAL)", real["text"], real["image_path"]),
             ("FAKE pair (true label FAKE)", fake["text"], fake["image_path"]),
             ("deliberately MISMATCHED (FAKE caption + REAL image)", fake["text"], real["image_path"])]
    out = {}
    for name, t, p in cases:
        r = pred.predict(t, p)
        out[name] = r
        print(f"\n{name}\n  text: {t[:80]}\n  -> {r['prediction']} (conf {r['confidence']}) | "
              f"P(fake) {r['p_fake']} | mismatch {r['evidence']['mismatch_score']} | "
              f"type: {r['manipulation_type']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--image")
    ap.add_argument("--modalities", nargs="+", choices=["text", "image"])
    ap.add_argument("--demo", help="path to data/test/test_clean.csv (3-case sanity check)")
    ap.add_argument("--final", default="auto", choices=["auto"] + FINAL_CHOICES)
    ap.add_argument("--no_explain", action="store_true")
    ap.add_argument("--out", help="save the JSON here")
    a = ap.parse_args()

    pred = Predictor(a.final)
    if a.demo:
        res = run_demo(pred, a.demo)
    elif a.text and a.image:
        res = pred.predict(a.text, a.image, a.modalities, explain=not a.no_explain)
        print(json.dumps(res, indent=2))
    else:
        raise SystemExit("Give --text and --image, or --demo <test_clean.csv>")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print("Saved", a.out)


if __name__ == "__main__":
    main()