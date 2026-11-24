
import csv
from pathlib import Path
from PIL import Image


class MultimodalDataset:
    """
    Loads image-caption pairs for multimodal deepfake detection.

    Label mapping:
    0 = FAKE (image and caption do not match)
    1 = REAL (image and caption match)
    """

    def __init__(self, csv_file, transform=None):
        self.csv_file = Path(csv_file)
        self.dataset_dir = self.csv_file.parent
        self.transform = transform

        if not self.csv_file.is_file():
            raise FileNotFoundError(
                f"Dataset CSV not found: {self.csv_file}"
            )

        # Read dataset CSV
        with self.csv_file.open(
            "r", encoding="utf-8", newline=""
        ) as file:
            reader = csv.DictReader(file)

            required_columns = {
                "sample_id",
                "image_path",
                "caption",
                "label",
                "category"
            }

            if not reader.fieldnames or not required_columns.issubset(
                reader.fieldnames
            ):
                raise ValueError("Required CSV columns are missing.")

            self.samples = list(reader)

        if not self.samples:
            raise ValueError("Dataset is empty.")

        # Validate samples
        sample_ids = set()

        for sample in self.samples:
            if not sample["sample_id"]:
                raise ValueError("A sample has an empty sample ID.")

            if sample["sample_id"] in sample_ids:
                raise ValueError(
                    f"Duplicate sample ID: {sample['sample_id']}"
                )

            sample_ids.add(sample["sample_id"])

            if sample["label"] not in ("0", "1"):
                raise ValueError(
                    f"Invalid label in {sample['sample_id']}"
                )

            if not sample["caption"].strip():
                raise ValueError(
                    f"Empty caption in {sample['sample_id']}"
                )

            image_path = self.dataset_dir / sample["image_path"]

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, index):
        """Load and return one image-caption sample."""

        sample = self.samples[index]

        image_path = self.dataset_dir / sample["image_path"]

        # Open image and convert it to RGB
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        # Apply optional image preprocessing
        if self.transform is not None:
            image = self.transform(image)

        return {
            "sample_id": sample["sample_id"],
            "image": image,
            "caption": sample["caption"],
            "label": int(sample["label"]),
            "category": sample["category"]
        }


if __name__ == "__main__":

    csv_path = Path(__file__).parent / "test_dataset.csv"

    dataset = MultimodalDataset(csv_path)

    print("Dataset loaded successfully!")
    print("Total samples:", len(dataset))

    print("\nFirst sample:")

    sample = dataset[0]

    print("Sample ID:", sample["sample_id"])
    print("Image size:", sample["image"].size)
    print("Image mode:", sample["image"].mode)
    print("Caption:", sample["caption"])
    print("Label:", sample["label"])
    print("Category:", sample["category"])
