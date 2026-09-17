"""Tests for the FastAPI serving app (app.py), using FastAPI's TestClient."""

import pytest
from fastapi.testclient import TestClient

from app import app

VALID_PAYLOAD = {
    "gender": "male",
    "married": "yes",
    "dependents": "0",
    "education": "graduate",
    "self_employed": "no",
    "applicantincome": 5849,
    "coapplicantincome": 0.0,
    "loanamount": 128.0,
    "loan_amount_term": 360.0,
    "credit_history": 1.0,
    "property_area": "urban",
}


@pytest.fixture
def client():
    # Using the app as a context manager runs the startup event (model loading).
    with TestClient(app) as test_client:
        yield test_client


def test_root_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


class TestPredictEndpoint:
    def test_valid_payload_returns_a_prediction(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

        body = response.json()
        assert body["loan_status"] in ("Approved", "Rejected")
        assert isinstance(body["approved"], bool)
        assert 0.0 <= body["approval_probability"] <= 1.0

    def test_missing_required_fields_returns_422(self, client):
        # "education" and "property_area" are required fields with no default.
        response = client.post("/predict", json={"applicantincome": 5849})
        assert response.status_code == 422

    def test_wrong_field_type_returns_422(self, client):
        invalid_payload = dict(VALID_PAYLOAD)
        invalid_payload["applicantincome"] = "not-a-number"

        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422
