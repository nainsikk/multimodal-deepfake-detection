
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.sanity import router


class FakePredictor:
    def __init__(self, loaded=True):
        self.loaded = loaded


@pytest.fixture
def client():
    test_app = FastAPI()
    test_app.include_router(router)

    fake_predictor = FakePredictor()
    test_app.state.predictor = fake_predictor

    with TestClient(test_app) as test_client:
        yield test_client, fake_predictor


def test_sanity_returns_three_results(client):
    test_client, _ = client

    response = test_client.get("/sanity")

    assert response.status_code == 200

    data = response.json()

    assert data["total_cases"] == 3
    assert len(data["results"]) == 3


def test_sanity_checks_predictor_status(client):
    test_client, _ = client

    response = test_client.get("/sanity")

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["predictor_type"] == "FakePredictor"
    assert data["mock_mode"] is False


def test_sanity_returns_illustrative_test_cases(client):
    test_client, _ = client

    response = test_client.get("/sanity")

    results = response.json()["results"]

    assert results[0]["expected_label"] == "REAL"
    assert results[1]["expected_label"] == "FAKE"
    assert results[2]["expected_label"] == "FAKE"

    for result in results:
        assert "id" in result
        assert "text" in result
        assert "expected_label" in result
        assert "purpose" in result

        assert "predicted_label" not in result
        assert "confidence" not in result
        assert "explanation" not in result


def test_sanity_identifies_mock_predictor():
    test_app = FastAPI()
    test_app.include_router(router)

    test_app.state.predictor = type(
        "MockPredictor",
        (),
        {"loaded": True}
    )()

    with TestClient(test_app) as test_client:
        response = test_client.get("/sanity")

    data = response.json()

    assert data["status"] == "healthy"
    assert data["mock_mode"] is True


def test_sanity_detects_predictor_not_ready():
    test_app = FastAPI()
    test_app.include_router(router)

    test_app.state.predictor = FakePredictor(loaded=False)

    with TestClient(test_app) as test_client:
        response = test_client.get("/sanity")

    data = response.json()

    assert data["status"] == "not_ready"
    assert data["model_loaded"] is False


def test_sanity_handles_missing_predictor():
    test_app = FastAPI()
    test_app.include_router(router)

    with TestClient(test_app) as test_client:
        response = test_client.get("/sanity")

    data = response.json()

    assert data["status"] == "not_ready"
    assert data["model_loaded"] is False
    assert data["predictor_type"] is None
    assert data["mock_mode"] is False
