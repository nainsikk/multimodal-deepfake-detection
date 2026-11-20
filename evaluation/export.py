
import json
from pathlib import Path
from datetime import datetime


def save_evaluation_report(
    results,
    output_dir="data/predictions",
    model_name="Multimodal Deepfake Detection"
):
    """
    Save model evaluation results as a JSON report.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = {
        "model_name": model_name,
        "evaluation_date": datetime.now().isoformat(
            timespec="seconds"
        ),
        "metrics": results
    }

    filename = (
        "evaluation_report_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )

    file_path = output_path / filename

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return str(file_path)
