"""
Part 1 (v2): Prepare a small, balanced Fakeddit subset from the OFFICIAL splits.

Input files (looked up inside --data_dir; override names with --train_tsv etc.):
    multimodal_train.tsv
    multimodal_validate.tsv
    multimodal_test_public.tsv

Required TSV columns : id, clean_title, image_url, 2_way_label, 6_way_label
Optional TSV columns : hasImage, subreddit   (used only if present)

Labels (as documented for Fakeddit):
    2_way_label : 0 = Fake, 1 = True/Real
    6_way_label : 0 True, 1 Satire, 2 Misleading, 3 Imposter,
                  4 False connection, 5 Manipulated

Output layout:
    <out>/train/train_clean.csv      + <out>/train/images/
    <out>/validate/validate_clean.csv + <out>/validate/images/
    <out>/test/test_clean.csv        + <out>/test/images/
    <out>/dataset_stats.json

Example:
    python part1_prepare_data.py --data_dir . --out data --per_class 200
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import imagehash
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

BINARY_NAMES = {0: "FAKE", 1: "REAL"}
SIX_WAY_NAMES = {0: "True", 1: "Satire", 2: "Misleading",
                 3: "Imposter", 4: "False connection", 5: "Manipulated"}
REQUIRED_COLS = ["id", "clean_title", "image_url", "2_way_label", "6_way_label"]
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP"}
HEADERS = {"User-Agent": "Mozilla/5.0 (course-project; research use)"}
MIN_SIDE = 64
MAX_BYTES = 10 * 1024 * 1024

# Processing order matters: test first, then validate, then train. Image hashes
# and texts seen in earlier splits are blocked from later ones, so train can
# never contain a copy of a validation/test sample (no leakage).
SPLIT_ORDER = ["test", "validate", "train"]


# ------------------------------------------------------------------ helpers
def safe_name(row_id):
    """Filesystem-safe filename stem derived from the id."""
    s = str(row_id).strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", s):
        return s
    return "id_" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:20]


def norm_text(t):
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def load_split_tsv(path):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: file not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, on_bad_lines="skip",
                     encoding="utf-8")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {os.path.basename(path)} is missing columns "
                 f"{missing}.\nColumns found: {list(df.columns)}")
    return df


def clean_split(raw, fail):
    """Row-level cleaning. `fail` is a Counter for removal reasons."""
    df = raw.copy()
    n0 = len(df)

    if "hasImage" in df.columns:
        has = df["hasImage"].astype(str).str.strip().str.lower()
        df = df[has.isin(["true", "1", "1.0"])]
        fail["no_image_flag"] += n0 - len(df)

    n = len(df)
    df = df.dropna(subset=["id", "clean_title", "image_url"])
    fail["missing_required_value"] += n - len(df)

    df["label"] = pd.to_numeric(df["2_way_label"], errors="coerce")
    df["label6"] = pd.to_numeric(df["6_way_label"], errors="coerce")
    n = len(df)
    df = df[df["label"].isin([0, 1]) & df["label6"].isin(range(6))]
    fail["invalid_label"] += n - len(df)
    df["label"] = df["label"].astype(int)
    df["label6"] = df["label6"].astype(int)

    df["text"] = df["clean_title"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    n = len(df)
    df = df[df["text"].str.len() >= 5]
    fail["text_too_short"] += n - len(df)

    df["id"] = df["id"].astype(str).str.strip()
    n = len(df)
    df = df.drop_duplicates(subset="id")
    fail["duplicate_id"] += n - len(df)

    df["_tn"] = df["text"].map(norm_text)
    n = len(df)
    df = df.drop_duplicates(subset="_tn")
    fail["duplicate_text_within_split"] += n - len(df)
    return df


# ---------------------------------------------------------------- download
def fetch_image(row_id, url, img_dir, timeout):
    """Download + validate one image. Never raises; returns a result dict."""
    res = {"id": row_id, "path": None, "phash": None, "status": "ok"}

    def fail(reason):
        res["status"] = reason
        return res

    url = str(url).strip()
    try:
        p = urlparse(url)
    except Exception:
        return fail("invalid_url")
    if p.scheme not in ("http", "https") or not p.netloc:
        return fail("invalid_url")

    stem = safe_name(row_id)
    final = os.path.join(img_dir, stem + ".jpg")
    part = final + ".part"
    out_tmp = final + ".tmp.jpg"

    try:
        # Resume support: reuse an already-valid image from a previous run.
        if os.path.exists(final):
            try:
                with Image.open(final) as im:
                    im.load()
                    res["path"] = final
                    res["phash"] = str(imagehash.phash(im.convert("RGB")))
                    return res
            except Exception:
                os.remove(final)

        try:
            with requests.get(url, headers=HEADERS, stream=True,
                              timeout=(5, timeout), allow_redirects=True) as r:
                if r.status_code != 200:
                    return fail("http_error")
                ctype = r.headers.get("Content-Type", "").lower()
                if ctype and not ctype.startswith("image/"):
                    return fail("not_image")
                size = 0
                with open(part, "wb") as f:
                    for chunk in r.iter_content(65536):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            return fail("too_large")
                        f.write(chunk)
                clen = r.headers.get("Content-Length")
                if (clen and clen.isdigit() and not r.headers.get("Content-Encoding")
                        and int(clen) != size):
                    return fail("incomplete_download")
        except requests.Timeout:
            return fail("timeout")
        except requests.RequestException:
            return fail("network_error")

        if size == 0:
            return fail("empty_file")

        try:
            with Image.open(part) as im:
                im.verify()                       # structural check
            with Image.open(part) as im:
                if im.format not in ALLOWED_FORMATS:
                    return fail("unsupported_format")
                if min(im.size) < MIN_SIDE:
                    return fail("image_too_small")
                rgb = im.convert("RGB")           # full decode: catches truncation
            res["phash"] = str(imagehash.phash(rgb))
            rgb.save(out_tmp, "JPEG", quality=95)
            os.replace(out_tmp, final)            # atomic: no half-written files
            res["path"] = final
            return res
        except Exception:
            return fail("corrupt_image")
    except Exception:
        return fail("other_error")
    finally:
        for tmp in (part, out_tmp):
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass


def build_split(name, df, target, img_dir, seen_hashes, workers, timeout, seed):
    """Balanced sampling with download-until-target.

    Per class we only ever request as many NEW candidates as are still missing
    (no blanket oversampling); failures and duplicates trigger another round.
    Returns (final_df, stats_dict)."""
    os.makedirs(img_dir, exist_ok=True)
    st = {"download_failures": Counter(), "attempted": 0,
          "image_duplicates_within_split": 0,
          "image_duplicates_vs_other_splits": 0}

    pools = {}
    for c in (0, 1):
        sub = df[df["label"] == c].sample(frac=1, random_state=seed)
        pools[c] = sub.to_dict("records")
    avail = min(len(pools[0]), len(pools[1]))
    if target <= 0 or target > avail:
        if target > avail:
            print(f"[{name}] only {avail} candidates in the smaller class; "
                  f"using {avail} per class instead of {target}.")
        target = avail
    st["target_per_class"] = target

    kept = {0: [], 1: []}
    pos = {0: 0, 1: 0}
    local_hashes, kept_paths = set(), set()

    while any(len(kept[c]) < target and pos[c] < len(pools[c]) for c in (0, 1)):
        jobs, meta = [], {}
        for c in (0, 1):
            need = target - len(kept[c])
            if need <= 0:
                continue
            batch = pools[c][pos[c]:pos[c] + need]
            pos[c] += len(batch)
            for rec in batch:
                jobs.append((rec["id"], rec["image_url"]))
                meta[rec["id"]] = rec
        if not jobs:
            break
        st["attempted"] += len(jobs)
        with ThreadPoolExecutor(workers) as ex:
            results = list(tqdm(
                ex.map(lambda j: fetch_image(j[0], j[1], img_dir, timeout), jobs),
                total=len(jobs), desc=f"{name}", leave=False))
        for r in results:
            rec = meta[r["id"]]
            if r["status"] != "ok":
                st["download_failures"][r["status"]] += 1
                continue
            h, path = r["phash"], r["path"]
            dup_other = h in seen_hashes and seen_hashes[h] != name
            dup_self = h in local_hashes
            if dup_other or dup_self:
                st["image_duplicates_vs_other_splits" if dup_other
                   else "image_duplicates_within_split"] += 1
                # Drop the ROW. Only delete the file if no kept row uses it.
                if path not in kept_paths and os.path.exists(path):
                    os.remove(path)
                continue
            local_hashes.add(h)
            kept_paths.add(path)
            rec["image_path"] = path
            kept[rec["label"]].append(rec)

    for h in local_hashes:
        seen_hashes[h] = name

    # Equalise counts exactly (a class can end up short if its pool ran out).
    n = min(len(kept[0]), len(kept[1]))
    rows = kept[0][:n] + kept[1][:n]
    for c in (0, 1):                       # remove files of trimmed rows
        for rec in kept[c][n:]:
            p = rec["image_path"]
            if p not in {x["image_path"] for x in rows} and os.path.exists(p):
                os.remove(p)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    st["download_failures"] = dict(st["download_failures"])
    return out, st


# -------------------------------------------------------------------- main
def dist(series, names):
    return {names.get(int(k), str(k)): int(v) for k, v in series.value_counts().sort_index().items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=".", help="folder containing the TSV files")
    ap.add_argument("--train_tsv", default="multimodal_train.tsv")
    ap.add_argument("--val_tsv", default="multimodal_validate.tsv")
    ap.add_argument("--test_tsv", default="multimodal_test_public.tsv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--per_class", type=int, default=200,
                    help="samples per binary class in EACH split (0 = max balanced)")
    ap.add_argument("--per_class_train", type=int, default=None)
    ap.add_argument("--per_class_val", type=int, default=None)
    ap.add_argument("--per_class_test", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=15, help="read timeout (s)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    files = {"train": a.train_tsv, "validate": a.val_tsv, "test": a.test_tsv}
    targets = {
        "train": a.per_class if a.per_class_train is None else a.per_class_train,
        "validate": a.per_class if a.per_class_val is None else a.per_class_val,
        "test": a.per_class if a.per_class_test is None else a.per_class_test,
    }

    raw, stats = {}, {
        "note": ("Fakeddit is a multimodal fake-news/misinformation dataset, "
                 "not a dedicated deepfake dataset."),
        "label_mapping": {"binary": BINARY_NAMES, "six_way": SIX_WAY_NAMES},
        "seed": a.seed, "splits": {},
    }
    for sp, fn in files.items():
        raw[sp] = load_split_tsv(os.path.join(a.data_dir, fn))
        print(f"[{sp}] read {len(raw[sp]):,} rows from {fn}")

    # Raw-text sets used to keep train/val/test text-disjoint.
    raw_text = {sp: set(raw[sp]["clean_title"].dropna().map(norm_text)) for sp in files}

    seen_hashes = {}
    for sp in SPLIT_ORDER:
        removal = Counter()
        cleaned = clean_split(raw[sp], removal)

        blocked = set()
        if sp == "validate":
            blocked = raw_text["test"]
        elif sp == "train":
            blocked = raw_text["test"] | raw_text["validate"]
        n = len(cleaned)
        cleaned = cleaned[~cleaned["_tn"].isin(blocked)]
        removal["text_overlap_with_other_splits"] += n - len(cleaned)

        # Original (pre-sampling) distributions are kept in the statistics.
        raw_lab = pd.to_numeric(raw[sp]["2_way_label"], errors="coerce").dropna().astype(int)
        raw_lab6 = pd.to_numeric(raw[sp]["6_way_label"], errors="coerce").dropna().astype(int)

        split_dir = os.path.join(a.out, sp)
        img_dir = os.path.join(split_dir, "images")
        final, st = build_split(sp, cleaned, targets[sp], img_dir, seen_hashes,
                                a.workers, a.timeout, a.seed)

        cols = ["id", "text", "image_path", "label", "label_name",
                "label6", "label6_name", "subreddit", "image_url"]
        if len(final):
            final["label_name"] = final["label"].map(BINARY_NAMES)
            final["label6_name"] = final["label6"].map(SIX_WAY_NAMES)
            if "subreddit" not in final.columns:
                final["subreddit"] = ""
            final = final[cols]
        else:
            final = pd.DataFrame(columns=cols)
        os.makedirs(split_dir, exist_ok=True)
        csv_path = os.path.join(split_dir, f"{sp}_clean.csv")
        final.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

        stats["splits"][sp] = {
            "source_file": files[sp],
            "raw_rows": int(len(raw[sp])),
            "raw_binary_distribution": dist(raw_lab, BINARY_NAMES),
            "raw_six_way_distribution": dist(raw_lab6, SIX_WAY_NAMES),
            "rows_after_cleaning": int(len(cleaned)),
            "cleaning_removed": dict(removal),
            "cleaned_binary_distribution": dist(cleaned["label"], BINARY_NAMES),
            "requested_per_class": targets[sp],
            "target_per_class_used": st["target_per_class"],
            "download_attempts": st["attempted"],
            "download_failures": st["download_failures"],
            "total_download_failures": int(sum(st["download_failures"].values())),
            "image_duplicates_removed_within_split": st["image_duplicates_within_split"],
            "image_duplicates_removed_vs_other_splits": st["image_duplicates_vs_other_splits"],
            "final_samples": int(len(final)),
            "final_binary_distribution": dist(final["label"], BINARY_NAMES) if len(final) else {},
            "final_six_way_distribution": dist(final["label6"], SIX_WAY_NAMES) if len(final) else {},
            "avg_text_length_words": float(final["text"].str.split().str.len().mean()) if len(final) else None,
            "csv": csv_path,
            "image_dir": img_dir,
        }
        print(f"[{sp}] final {len(final)} samples -> {csv_path}")

    stats_path = os.path.join(a.out, "dataset_stats.json")
    os.makedirs(a.out, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics written to {stats_path}")


if __name__ == "__main__":
    main()