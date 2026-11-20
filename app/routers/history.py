
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models.analysis import Analysis

router = APIRouter()


@router.get("/history")
def get_history(db: Session = Depends(get_db)):
    analyses = (
        db.query(Analysis)
        .order_by(desc(Analysis.id))
        .all()
    )

    return [
        {
            "id": analysis.id,
            "timestamp": analysis.timestamp,
            "prediction": analysis.prediction,
            "confidence": analysis.confidence
        }
        for analysis in analyses
    ]
