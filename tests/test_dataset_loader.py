
import csv
import pytest
from PIL import Image

from data.small_test_dataset.dataset_loader import MultimodalDataset


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a temporary dataset for testing."""

    image_dir = tmp_path / "images"
    image_dir.mkdir()

    # Create two sample images
    Image.new("RGB", (100, 100), "blue").save(
        image_dir / "beach.png"
    )

    Image.new("RGB", (100, 100), "green").save(
        image_dir / "soccer.png"
    )

    csv_file = tmp_path / "test_dataset.csv"

    rows = [
        {
            "sample_id": "S01",
            "image_path": "images/beach.png",
            "caption": "A sunny beach.",
            "label": "1",
            "category": "beach"
        },
        {
            "sample_id": "S02",
            "image_path": "images/soccer.png",
            "caption": "A soccer match.",
            "label": "0",
            "category": "soccer"
        }
    ]

    with csv_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "image_path",
                "caption",
                "label",
                "category"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    return csv_file


def test_dataset_length(sample_dataset):
    dataset = MultimodalDataset(sample_dataset)

    assert len(dataset) == 2


def test_load_first_sample(sample_dataset):
    dataset = MultimodalDataset(sample_dataset)

    sample = dataset[0]

    assert sample["sample_id"] == "S01"
    assert sample["caption"] == "A sunny beach."
    assert sample["label"] == 1
    assert sample["category"] == "beach"


def test_image_is_rgb(sample_dataset):
    dataset = MultimodalDataset(sample_dataset)

    sample = dataset[0]

    assert sample["image"].mode == "RGB"
    assert sample["image"].size == (100, 100)


def test_missing_image_raises_error(tmp_path):
    csv_file = tmp_path / "test_dataset.csv"

    with csv_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "image_path",
                "caption",
                "label",
                "category"
            ]
        )
        writer.writeheader()
        writer.writerow({
            "sample_id": "S01",
            "image_path": "images/missing.png",
            "caption": "A missing image.",
            "label": "1",
            "category": "test"
        })

    with pytest.raises(FileNotFoundError):
        MultimodalDataset(csv_file)


def test_invalid_label_raises_error(sample_dataset):
    rows = list(csv.DictReader(
        sample_dataset.open(encoding="utf-8")
    ))

    rows[0]["label"] = "5"

    with sample_dataset.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Invalid label"):
        MultimodalDataset(sample_dataset)


def test_duplicate_sample_id_raises_error(sample_dataset):
    rows = list(csv.DictReader(
        sample_dataset.open(encoding="utf-8")
    ))

    rows[1]["sample_id"] = "S01"

    with sample_dataset.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Duplicate sample ID"):
        MultimodalDataset(sample_dataset)
