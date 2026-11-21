
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models.analysis import Analysis
from app.routers.history import router


@pytest.fixture
def db_session():
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


def create_analysis(db, prediction, confidence, timestamp):
    analysis = Analysis(
        timestamp=timestamp,
        prediction=prediction,
        confidence=confidence,
        result_json="{}"
    )

    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    return analysis


def test_history_returns_empty_list(client):
    response = client.get("/history")

    assert response.status_code == 200
    assert response.json() == []


def test_history_returns_saved_records(client, db_session):
    create_analysis(
        db_session,
        "REAL",
        0.90,
        "2026-11-20 10:00:00"
    )

    create_analysis(
        db_session,
        "FAKE",
        0.85,
        "2026-11-20 11:00:00"
    )

    response = client.get("/history")

    assert response.status_code == 200

    results = response.json()

    assert len(results) == 2


def test_history_returns_newest_record_first(client, db_session):
    older = create_analysis(
        db_session,
        "REAL",
        0.90,
        "2026-11-20 10:00:00"
    )

    newer = create_analysis(
        db_session,
        "FAKE",
        0.85,
        "2026-11-20 11:00:00"
    )

    response = client.get("/history")

    results = response.json()

    assert results[0]["id"] == newer.id
    assert results[1]["id"] == older.id


def test_history_returns_correct_fields(client, db_session):
    analysis = create_analysis(
        db_session,
        "FAKE",
        0.85,
        "2026-11-20 11:00:00"
    )

    response = client.get("/history")

    assert response.status_code == 200

    result = response.json()[0]

    assert result["id"] == analysis.id
    assert result["timestamp"] == "2026-11-20 11:00:00"
    assert result["prediction"] == "FAKE"
    assert result["confidence"] == 0.85

    assert set(result.keys()) == {
        "id",
        "timestamp",
        "prediction",
        "confidence"
    }

