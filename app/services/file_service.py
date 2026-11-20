
import os
import tempfile
from io import BytesIO

from fastapi import UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError


MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_IMAGES = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG"
}


async def save_image(image: UploadFile) -> str:

    filename = image.filename or ""
    extension = os.path.splitext(filename)[1].lower()

    # Check file extension
    if extension not in ALLOWED_IMAGES:
        raise HTTPException(
            status_code=415,
            detail="Invalid image format. Upload JPG or PNG."
        )

    # Read uploaded file
    content = await image.read()

    # Check empty file
    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the maximum size of 10 MB."
        )

    # Verify actual image content
    try:
        img = Image.open(BytesIO(content))

        actual_format = img.format
        img.verify()

        if actual_format != ALLOWED_IMAGES[extension]:
            raise HTTPException(
                status_code=415,
                detail="File content does not match its extension."
            )

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError
    ):
        raise HTTPException(
            status_code=415,
            detail="Invalid or corrupted image file."
        )

    # Save verified image temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    ) as temp_file:

        temp_file.write(content)

        return temp_file.name
