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

    text: str | None = Form(None),

    image: UploadFile | None = File(None),

    modalities: str | None = Form(None),

    db: Session = Depends(get_db)
):

    image_path = None

    try:
        # Check whether text and image have been provided.
        has_text = bool(text and text.strip())
        has_image = image is not None

        # Validate and identify selected modalities.
        allowed = {"text", "image"}

        if modalities is not None:
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

        else:
            # Automatically detect modalities if not specified.
            selected = set()

            if has_text:
                selected.add("text")

            if has_image:
                selected.add("image")

        # At least one modality must be selected.
        if not selected:
            raise HTTPException(
                status_code=400,
                detail="Provide at least one input: text or image."
            )

        # Reject inputs that do not match selected modalities.
        if has_text and "text" not in selected:
            raise HTTPException(
                status_code=400,
                detail="Text was provided but text modality was not selected."
            )

        if has_image and "image" not in selected:
            raise HTTPException(
                status_code=400,
                detail="Image was provided but image modality was not selected."
            )

        # Check that the selected inputs are available.
        if "text" in selected and not has_text:
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty."
            )

        if "image" in selected and not has_image:
            raise HTTPException(
                status_code=400,
                detail="An image is required for image analysis."
            )

        # Save the image only when image analysis is selected.
        if "image" in selected:
            image_path = await save_image(image)

        # Prepare predictor inputs.
        predictor_text = text.strip() if "text" in selected else None

        predictor_image = image_path if "image" in selected else None

        # Get the already-loaded prediction model.
        predictor = request.app.state.predictor

        # Generate prediction.
        result = predictor.predict(
            text=predictor_text,
            image_path=predictor_image
        )

        # Save result in database.
        analysis = Analysis(
            timestamp=datetime.now(timezone.utc).isoformat(),
            prediction=result["prediction"],
            confidence=result["confidence"],
            result_json=json.dumps(result)
        )

        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        # Attach database ID.
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
        # Delete uploaded image after analysis.
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

        # Close uploaded file if one was provided.
        if image is not None:
            await image.close()