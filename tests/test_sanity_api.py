
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.sanity import router


class FakePredictor:
    def __init__(self):
        self.calls = []

    def predict(self, text, image_path):
        self.calls.append((text, image_path))

        return {
            "prediction": "FAKE",
            "confidence": 0.85,
            "explanation": "Test prediction explanation."
        }


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


def test_sanity_passes_correct_inputs_to_predictor(client):
    test_client, predictor = client

    test_client.get("/sanity")

    assert predictor.calls == [
        (
            "A beautiful rainbow appears after rainfall.",
            "sanity_real.jpg"
        ),
        (
            "A celebrity has been digitally altered to appear as another person.",
            "sanity_fake.jpg"
        ),
        (
            "Heavy flooding has affected several streets.",
            "sanity_mismatch.jpg"
        )
    ]


def test_sanity_returns_prediction_details(client):
    test_client, _ = client

    response = test_client.get("/sanity")

    results = response.json()["results"]

    for result in results:
        assert "id" in result
        assert "text" in result
        assert "expected_label" in result
        assert "predicted_label" in result
        assert "confidence" in result
        assert "explanation" in result

        assert result["predicted_label"] == "FAKE"
        assert result["confidence"] == 0.85
        assert result["explanation"] == "Test prediction explanation."
