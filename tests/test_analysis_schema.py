
import pytest
from pydantic import ValidationError

from app.schemas.analysis import AnalysisResponse


def create_valid_response(mismatch_score):
    return {
        "id": 1,
        "prediction": "REAL",
        "confidence": 0.94,
        "manipulation_type": "none",
        "modality_scores": {
            "text": 0.93,
            "image": 0.95,
            "fusion": 0.94
        },
        "evidence": {
            "text_tokens": [],
            "image_heatmap_url": None,
            "mismatch_score": mismatch_score
        },
        "explanation": "The caption and image appear consistent."
    }


def test_zero_mismatch_score_is_valid():
    response = AnalysisResponse(**create_valid_response(0))

    assert response.evidence.mismatch_score == 0


def test_maximum_mismatch_score_is_valid():
    response = AnalysisResponse(**create_valid_response(100))

    assert response.evidence.mismatch_score == 100


@pytest.mark.parametrize("score", [-1, 101])
def test_out_of_range_mismatch_score_is_rejected(score):
    with pytest.raises(ValidationError):
        AnalysisResponse(**create_valid_response(score))
