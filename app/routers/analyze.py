
import os
import json
import logging

from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    Depends,
    HTTPException,
    Request
)

from sqlalchemy.orm import Session

from app.database import get_db
from app.models.analysis import Analysis
from app.schemas.analysis import AnalysisResponse
from app.services.file_service import save_image


router = APIRouter()

logger = logging.getLogger(__name__)


@router.post(
    "/analyze",
    response_model=AnalysisResponse
)
async def analyze(

    request: Request,

    text: str = Form(...),

    image: UploadFile = File(...),

    modalities: str = Form("text,image"),

    db: Session = Depends(get_db)

):

    image_path = None

    try:

        # Validate text
        if not text or not text.strip():

            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty."
            )

        # Validate modalities
        allowed = {"text", "image"}

        selected = {
            item.strip().lower()
            for item in modalities.split(",")
            if item.strip()
        }

        if not selected or not selected.issubset(allowed):

            raise HTTPException(
                status_code=400,
                detail="Select valid modalities: text,image."
            )

        # Save uploaded image temporarily
        image_path = await save_image(image)

        # Get the already-loaded prediction model
        predictor = request.app.state.predictor

        # Generate prediction
        result = predictor.predict(
            text=text,
            image_path=image_path
        )

        # Save result in database
        analysis = Analysis(

            timestamp=datetime.now(timezone.utc).isoformat(),

            prediction=result["prediction"],

            confidence=result["confidence"],

            result_json=json.dumps(result)

        )

        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        # Attach database ID
        result["id"] = analysis.id

        logger.info(
            "Analysis ID %s completed: %s",
            analysis.id,
            result["prediction"]
        )

        return result

    except HTTPException:
        raise

    except Exception:

        db.rollback()

        logger.exception("Analysis failed")

        raise HTTPException(
            status_code=500,
            detail="Analysis failed due to an internal server error."
        )

    finally:

        # Delete uploaded media
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

        await image.close()
