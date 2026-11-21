
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models.analysis import Analysis
from app.routers.results import router


@pytest.fixture
def db_session():
    # Create a temporary in-memory database.
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

    db = TestingSessionLocal()

    yield db

    db.close()
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


@pytest.fixture
def client(db_session):
    # Create a test application.
    test_app = FastAPI()
    test_app.include_router(router)

    # Use the temporary test database.
    def override_get_db():
        yield db_session

    test_app.dependency_overrides[get_db] = override_get_db

    with TestClient(test_app) as test_client:
        yield test_client

    test_app.dependency_overrides.clear()


def create_analysis(db):
    # Prepare a sample saved analysis result.
    sample_result = {
        "prediction": "FAKE",
        "confidence": 0.85,
        "evidence": {
            "mismatch_score": 75
        }
    }

    analysis = Analysis(
        timestamp="2026-11-21 10:00:00",
        prediction="FAKE",
        confidence=0.85,
        result_json=json.dumps(sample_result)
    )

    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    return analysis, sample_result


def test_get_existing_result(client, db_session):
    analysis, sample_result = create_analysis(db_session)

    response = client.get(f"/result/{analysis.id}")

    assert response.status_code == 200

    result = response.json()

    assert result["id"] == analysis.id
    assert result["prediction"] == "FAKE"
    assert result["confidence"] == 0.85
    assert result["evidence"]["mismatch_score"] == 75


def test_get_nonexistent_result(client):
    response = client.get("/result/9999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Analysis not found."


def test_get_result_with_invalid_id(client):
    response = client.get("/result/abc")

    assert response.status_code == 422
