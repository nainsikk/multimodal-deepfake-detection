
import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models.analysis import Analysis
from app.routers.analyze import router
from app.services.mock_predictor import MockPredictor


@pytest.fixture
def client():
    # Create a separate temporary in-memory database.
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )

    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_engine
    )

    Base.metadata.create_all(bind=test_engine)

    # Provide the test database to the API.
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Create a test application without the production lifespan.
    test_app = FastAPI()
    test_app.state.predictor = MockPredictor()
    test_app.include_router(router)

    test_app.dependency_overrides[get_db] = override_get_db

    with TestClient(test_app) as test_client:
        yield test_client

    test_app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


def create_test_image():
    # Generate a valid PNG image in memory.
    image_bytes = io.BytesIO()

    image = Image.new("RGB", (100, 100), color="red")
    image.save(image_bytes, format="PNG")

    image_bytes.seek(0)

    return image_bytes


def test_analyze_success(client):
    # Test text and image analysis together.
    image = create_test_image()

    response = client.post(
        "/analyze",
        data={
            "text": "Heavy flooding has affected several streets.",
            "modalities": "text,image"
        },
        files={
            "image": (
                "test.png",
                image,
                "image/png"
            )
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert result["id"] is not None
    assert result["prediction"] in ["REAL", "FAKE", "UNCERTAIN"]
    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["evidence"]["mismatch_score"] <= 100


def test_analyze_text_only(client):
    # Test analysis using only text.
    response = client.post(
        "/analyze",
        data={
            "text": "A news report about flooding.",
            "modalities": "text"
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert result["id"] is not None
    assert result["modality_scores"]["text"] is not None
    assert result["modality_scores"]["image"] is None
    assert result["modality_scores"]["fusion"] is None
    assert result["evidence"]["mismatch_score"] is None
    assert result["manipulation_type"] == "text-only analysis"


def test_analyze_image_only(client):
    # Test analysis using only an image.
    image = create_test_image()

    response = client.post(
        "/analyze",
        data={
            "modalities": "image"
        },
        files={
            "image": (
                "test.png",
                image,
                "image/png"
            )
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert result["id"] is not None
    assert result["modality_scores"]["text"] is None
    assert result["modality_scores"]["image"] is not None
    assert result["modality_scores"]["fusion"] is None
    assert result["evidence"]["mismatch_score"] is None
    assert result["manipulation_type"] == "image-only analysis"


def test_analyze_creates_database_record(client):
    image = create_test_image()

    response = client.post(
        "/analyze",
        data={
            "text": "A news report about flooding.",
            "modalities": "text,image"
        },
        files={
            "image": (
                "test.png",
                image,
                "image/png"
            )
        }
    )

    assert response.status_code == 200

    analysis_id = response.json()["id"]

    # Check the database record using the API response ID.
    db = next(
        client.app.dependency_overrides[get_db]()
    )

    try:
        record = db.query(Analysis).filter(
            Analysis.id == analysis_id
        ).first()

        assert record is not None
        assert record.prediction in ["REAL", "FAKE", "UNCERTAIN"]

    finally:
        db.close()


def test_analyze_rejects_empty_text(client):
    image = create_test_image()

    response = client.post(
        "/analyze",
        data={
            "text": "   ",
            "modalities": "text,image"
        },
        files={
            "image": (
                "test.png",
                image,
                "image/png"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Text cannot be empty."


def test_analyze_rejects_invalid_image(client):
    response = client.post(
        "/analyze",
        data={
            "text": "Testing invalid image.",
            "modalities": "text,image"
        },
        files={
            "image": (
                "document.txt",
                b"This is not an image.",
                "text/plain"
            )
        }
    )

    assert response.status_code == 415
