
import csv
from pathlib import Path

# Dataset folder
DATASET_DIR = Path(__file__).parent
CSV_FILE = DATASET_DIR / "test_dataset.csv"
IMAGE_DIR = DATASET_DIR / "images"

# Required CSV columns
REQUIRED_COLUMNS = {
    "sample_id",
    "image_path",
    "caption",
    "label",
    "category"
}


def validate_dataset():
    print("Starting dataset validation...\n")

    # 1. Check whether CSV exists
    if not CSV_FILE.is_file():
        print("ERROR: CSV file not found.")
        return

    # 2. Read CSV
    with CSV_FILE.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            print("ERROR: Required CSV columns are missing.")
            return

        rows = list(reader)

    # 3. Check whether dataset is empty
    if not rows:
        print("ERROR: Dataset is empty.")
        return

    # 4. Check labels
    invalid_labels = [
        row["sample_id"]
        for row in rows
        if row["label"] not in ("0", "1")
    ]

    # 5. Check image paths
    missing_images = [
        row["image_path"]
        for row in rows
        if not (DATASET_DIR / row["image_path"]).is_file()
    ]

    # 6. Count REAL and FAKE samples
    real_count = sum(row["label"] == "1" for row in rows)
    fake_count = sum(row["label"] == "0" for row in rows)

    # 7. Check duplicate sample IDs
    sample_ids = [row["sample_id"] for row in rows]
    duplicate_ids = sorted({
        sample_id for sample_id in sample_ids
        if sample_ids.count(sample_id) > 1
    })

    # 8. Display results
    print("Total samples:", len(rows))
    print("REAL samples:", real_count)
    print("FAKE samples:", fake_count)
    print("Invalid labels:", invalid_labels)
    print("Missing images:", missing_images)
    print("Duplicate sample IDs:", duplicate_ids)

    # 9. Final result
    if not invalid_labels and not missing_images and not duplicate_ids:
        print("\nVALIDATION PASSED!")
    else:
        print("\nVALIDATION FAILED!")


if __name__ == "__main__":
    validate_dataset()
