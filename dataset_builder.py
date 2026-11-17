"""
Part 2 (v2): PyTorch Dataset / DataLoaders for the cleaned Fakeddit subset.

Uses the OFFICIAL splits produced by part1_prepare_data.py. There is NO random
splitting here, so there is no leakage and no splits.json.

Expected files (made by Part 1):
    data/train/train_clean.csv
    data/validate/validate_clean.csv
    data/test/test_clean.csv

Quick check (run from the same folder you ran Part 1 in):
    python part2_dataset.py --data_dir data

Use in your training code:
    from part2_dataset import make_loaders
    train_dl, val_dl, test_dl = make_loaders("data", tokenizer=tok, batch_size=16)

Install: pip install pandas pillow torch torchvision transformers
"""
import argparse
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SPLITS = ["train", "validate", "test"]
IMG_MEAN, IMG_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def split_csv(data_dir, split):
    return os.path.join(data_dir, split, f"{split}_clean.csv")


def load_split_df(data_dir, split):
    path = split_csv(data_dir, split)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found. Run part1_prepare_data.py first.")
    df = pd.read_csv(path, dtype={"id": str})
    need = ["id", "text", "image_path", "label", "label6"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return df


def image_transform(train: bool, size=224):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])


class FakedditDataset(Dataset):
    """Each item: dict with id, label (0=FAKE, 1=REAL), label6, and
    text and/or image depending on use_text / use_image (for baselines and
    modality ablation)."""

    def __init__(self, df, train=False, use_text=True, use_image=True, size=224):
        self.df = df.reset_index(drop=True)
        self.tf = image_transform(train, size)
        self.use_text, self.use_image = use_text, use_image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        item = {"id": str(r["id"]), "label": int(r["label"]),
                "label6": int(r["label6"])}
        if self.use_text:
            item["text"] = str(r["text"])
        if self.use_image:
            with Image.open(r["image_path"]) as im:
                item["image"] = self.tf(im.convert("RGB"))
        return item


def make_collate(tokenizer=None, max_len=64):
    """Batches items; tokenizes text here if a tokenizer is given."""
    def collate(batch):
        out = {
            "id": [b["id"] for b in batch],
            "label": torch.tensor([b["label"] for b in batch]),
            "label6": torch.tensor([b["label6"] for b in batch]),
        }
        if "image" in batch[0]:
            out["image"] = torch.stack([b["image"] for b in batch])
        if "text" in batch[0]:
            out["text"] = [b["text"] for b in batch]
            if tokenizer is not None:
                out["tokens"] = tokenizer(
                    out["text"], padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
        return out
    return collate


def make_loaders(data_dir="data", tokenizer=None, batch_size=16,
                 use_text=True, use_image=True, num_workers=0, max_len=64):
    """Returns (train_loader, val_loader, test_loader).
    num_workers=0 is the safest on macOS; raise it later if loading is slow."""
    loaders = []
    for sp in SPLITS:
        df = load_split_df(data_dir, sp)
        ds = FakedditDataset(df, train=(sp == "train"),
                             use_text=use_text, use_image=use_image)
        loaders.append(DataLoader(
            ds, batch_size=batch_size, shuffle=(sp == "train"),
            num_workers=num_workers,
            collate_fn=make_collate(tokenizer, max_len)))
    return tuple(loaders)


# ------------------------------------------------------------ verification
def verify(data_dir):
    print("Checking dataset in:", data_dir)
    all_ids = {}
    for sp in SPLITS:
        df = load_split_df(data_dir, sp)
        missing = [p for p in df["image_path"] if not os.path.isfile(p)]
        print(f"\n[{sp}] {len(df)} samples")
        print("  binary labels :", df["label"].value_counts().sort_index().to_dict(),
              "(0=FAKE, 1=REAL)")
        print("  six-way labels:", df["label6"].value_counts().sort_index().to_dict())
        print("  avg words/text:", round(df["text"].str.split().str.len().mean(), 1))
        print("  missing images:", len(missing))
        if missing:
            print("   e.g.", missing[:3])
        all_ids[sp] = set(df["id"])
    overlap = (all_ids["train"] & all_ids["validate"]) | \
              (all_ids["train"] & all_ids["test"]) | \
              (all_ids["validate"] & all_ids["test"])
    print("\nID overlap between splits:", len(overlap), "(must be 0)")

    # Load one real batch to prove the loaders work.
    train_dl, _, _ = make_loaders(data_dir, tokenizer=None, batch_size=4)
    b = next(iter(train_dl))
    print("\nOne batch OK -> image", tuple(b["image"].shape),
          "| labels", b["label"].tolist(), "| first text:", b["text"][0][:60])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    verify(ap.parse_args().data_dir)