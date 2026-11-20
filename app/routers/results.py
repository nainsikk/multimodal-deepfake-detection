
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.analysis import Analysis

router = APIRouter()


@router.get("/result/{analysis_id}")
def get_result(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    analysis = (
        db.query(Analysis)
        .filter(Analysis.id == analysis_id)
        .first()
    )

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found."
        )

    result = json.loads(analysis.result_json)

    result["id"] = analysis.id

    return result
