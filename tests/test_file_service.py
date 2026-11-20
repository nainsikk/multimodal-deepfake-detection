
import pytest
from fastapi import HTTPException, UploadFile
from io import BytesIO

from app.services.file_service import save_image


@pytest.mark.asyncio
async def test_reject_invalid_extension():
    fake_file = UploadFile(
        filename="document.pdf",
        file=BytesIO(b"some content")
    )

    with pytest.raises(HTTPException) as error:
        await save_image(fake_file)

    assert error.value.status_code == 415
    assert error.value.detail == "Invalid image format. Upload JPG or PNG."


@pytest.mark.asyncio
async def test_reject_empty_file():
    fake_file = UploadFile(
        filename="empty.png",
        file=BytesIO(b"")
    )

    with pytest.raises(HTTPException) as error:
        await save_image(fake_file)

    assert error.value.status_code == 400
    assert error.value.detail == "Uploaded image is empty."


@pytest.mark.asyncio
async def test_reject_large_file():
    large_content = b"x" * (10 * 1024 * 1024 + 1)

    fake_file = UploadFile(
        filename="large.png",
        file=BytesIO(large_content)
    )

    with pytest.raises(HTTPException) as error:
        await save_image(fake_file)

    assert error.value.status_code == 413
    assert error.value.detail == "Image exceeds the maximum size of 10 MB."
