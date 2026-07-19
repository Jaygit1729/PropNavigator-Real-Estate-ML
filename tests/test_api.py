"""Tests for the FastAPI endpoints.

Uses FastAPI's TestClient, which drives the app in-process — no server to start,
no port to pick, and it still exercises the real routing, validation and
serialisation. Requires httpx.
"""

import pytest

pytest.importorskip("httpx", reason="httpx is required by fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(service):        # `service` first, so we skip cleanly if no model exists
    from api.main import app
    with TestClient(app) as c:
        yield c


# --- /health ------------------------------------------------------------------

def test_health_reports_the_loaded_model(client, service):
    r = client.get("/health")
    assert r.status_code == 200

    body = r.json()
    assert body["status"] == "ok"
    assert body["model_name"] == service.model_name
    assert body["n_features"] == 24


# --- /sectors -----------------------------------------------------------------

def test_sectors_lists_valid_values(client):
    body = client.get("/sectors").json()
    assert body["count"] == len(body["sectors"]) > 0
    assert "sector 49" in body["sectors"]


# --- /predict: happy paths ----------------------------------------------------

def test_predict_returns_price_and_interval(client, valid_request):
    r = client.post("/predict", json=valid_request)
    assert r.status_code == 200

    body = r.json()
    assert body["lower_bound_cr"] < body["predicted_price_cr"] < body["upper_bound_cr"]
    assert body["mape_percent"] > 0


def test_predict_works_with_only_required_fields(client):
    """The five essentials must be enough; everything else has a server-side default."""
    r = client.post("/predict", json={
        "property_type": "Flat", "sector": "sector 49",
        "area": 1500, "bedRoom": 3, "bathroom": 2,
    })
    assert r.status_code == 200
    assert r.json()["predicted_price_cr"] > 0


# --- /predict: the contract must reject bad input -----------------------------

@pytest.mark.parametrize("bad_field, value", [
    ("bedRoom", 99),                 # above ge/le bounds
    ("bedRoom", 0),                  # below minimum
    ("area", 50),                    # below gt=100
    ("area", 99_999),                # above le=27000
    ("bathroom", -1),
    ("property_type", "Houseboat"),  # not in the Literal
    ("furnishing", "gilded"),        # not in the Literal
    ("has_pool", 7),                 # not 0/1
])
def test_out_of_contract_values_are_rejected(client, valid_request, bad_field, value):
    r = client.post("/predict", json={**valid_request, bad_field: value})
    assert r.status_code == 422, f"{bad_field}={value!r} should have been rejected"


def test_missing_required_field_is_rejected(client, valid_request):
    payload = {k: v for k, v in valid_request.items() if k != "area"}
    assert client.post("/predict", json=payload).status_code == 422


def test_unknown_sector_returns_422_with_a_helpful_message(client, valid_request):
    r = client.post("/predict", json={**valid_request, "sector": "atlantis"})
    assert r.status_code == 422
    assert "Unknown sector" in r.json()["detail"]


# --- consistency --------------------------------------------------------------

def test_api_matches_direct_inference(client, service, valid_request):
    """The HTTP layer must not alter the answer the inference module gives."""
    via_http = client.post("/predict", json=valid_request).json()["predicted_price_cr"]
    direct = service.predict_price(valid_request)["predicted_price_cr"]
    assert via_http == direct


def test_same_input_gives_same_output(client, valid_request):
    """Predictions must be deterministic — no hidden randomness in the serving path."""
    first = client.post("/predict", json=valid_request).json()
    second = client.post("/predict", json=valid_request).json()
    assert first == second
