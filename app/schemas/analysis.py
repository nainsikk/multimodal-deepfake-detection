
from pydantic import BaseModel, Field
from typing import Literal


class TokenEvidence(BaseModel):
    token: str
    weight: float


class ModalityScores(BaseModel):
    text: float | None = None
    image: float | None = None
    fusion: float | None = None


class Evidence(BaseModel):
    text_tokens: list[TokenEvidence]
    image_heatmap_url: str | None

    mismatch_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Text-image mismatch score from 0 to 100. "
                    "Null when both modalities are not available."
    )


class AnalysisResponse(BaseModel):

    id: int | None = None

    prediction: Literal["REAL", "FAKE", "UNCERTAIN"]

    confidence: float

    manipulation_type: str

    modality_scores: ModalityScores

    evidence: Evidence

    explanation: str
