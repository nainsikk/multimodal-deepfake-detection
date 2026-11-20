
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/sanity")
def run_sanity_checks(request: Request):

    predictor = request.app.state.predictor

    test_cases = [
        {
            "id": 1,
            "text": "A beautiful rainbow appears after rainfall.",
            "image_path": "sanity_real.jpg",
            "expected_label": "REAL"
        },
        {
            "id": 2,
            "text": "A celebrity has been digitally altered to appear as another person.",
            "image_path": "sanity_fake.jpg",
            "expected_label": "FAKE"
        },
        {
            "id": 3,
            "text": "Heavy flooding has affected several streets.",
            "image_path": "sanity_mismatch.jpg",
            "expected_label": "FAKE"
        }
    ]

    results = []

    for case in test_cases:

        prediction = predictor.predict(
            case["text"],
            case["image_path"]
        )

        results.append({
            "id": case["id"],
            "text": case["text"],
            "expected_label": case["expected_label"],
            "predicted_label": prediction["prediction"],
            "confidence": prediction["confidence"],
            "explanation": prediction["explanation"]
        })

    return {
        "total_cases": len(results),
        "results": results
    }
