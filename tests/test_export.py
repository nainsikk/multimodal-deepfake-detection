
import json

from evaluation.export import save_evaluation_report


def test_save_evaluation_report(tmp_path):
    results = {
        "accuracy": 0.90,
        "precision": 0.88,
        "recall": 0.92,
        "f1_score": 0.90,
        "roc_auc": 0.95,
        "pr_auc": 0.94,
        "confusion_matrix": [
            [45, 5],
            [4, 46]
        ]
    }

    file_path = save_evaluation_report(
        results,
        output_dir=tmp_path
    )

    # Check that the JSON file was created.
    assert tmp_path.exists()

    # Read the saved JSON report.
    with open(file_path, "r", encoding="utf-8") as file:
        report = json.load(file)

    assert report["model_name"] == "Multimodal Deepfake Detection"
    assert report["metrics"]["accuracy"] == 0.90
    assert report["metrics"]["roc_auc"] == 0.95
    assert report["metrics"]["pr_auc"] == 0.94
    assert "evaluation_date" in report


def test_export_creates_output_folder(tmp_path):
    output_dir = tmp_path / "new_folder"

    save_evaluation_report(
        {"accuracy": 0.85},
        output_dir=output_dir
    )

    assert output_dir.exists()
    assert len(list(output_dir.glob("*.json"))) == 1
