
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/sanity")
def run_sanity_checks(request: Request):

    predictor = getattr(request.app.state, "predictor", None)

    predictor_loaded = (
        predictor is not None
        and getattr(predictor, "loaded", False)
    )

    predictor_type = (
        predictor.__class__.__name__
        if predictor is not None
        else None
    )

    test_cases = [
        {
            "id": 1,
            "text": "A beautiful rainbow appears after rainfall.",
            "expected_label": "REAL",
            "purpose": "Example of genuine visual content."
        },
        {
            "id": 2,
            "text": "A celebrity has been digitally altered to appear as another person.",
            "expected_label": "FAKE",
            "purpose": "Example of digitally manipulated content."
        },
        {
            "id": 3,
            "text": "Heavy flooding has affected several streets.",
            "expected_label": "FAKE",
            "purpose": "Example scenario for checking caption-image consistency."
        }
    ]

    return {
        "status": "healthy" if predictor_loaded else "not_ready",
        "model_loaded": predictor_loaded,
        "predictor_type": predictor_type,
        "mock_mode": predictor_type == "MockPredictor",
        "total_cases": len(test_cases),
        "results": test_cases,
        "note": (
            "These are illustrative test scenarios, not actual model "
            "predictions. Real evaluation requires the trained model "
            "and actual test images."
        )
    }
