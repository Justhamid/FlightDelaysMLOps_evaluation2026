"""Tests de l'API FastAPI via TestClient (modele de test injecte, cf. conftest.py)."""

VALID_PAYLOAD = {
    "month": 7,
    "day_of_week": 3,
    "airline": "AA",
    "origin_airport": "JFK",
    "destination_airport": "LAX",
    "scheduled_departure": 1430,
    "scheduled_time": 360,
    "distance": 600,
}


def test_root_returns_api_info(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "Flight Delays API"


def test_health_reports_model_loaded(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_request_returns_prediction(api_client):
    response = api_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["delayed"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["anomaly"] is False


def test_predict_same_airport_returns_422(api_client):
    """Validateur metier : un vol ne peut pas partir et arriver au meme aeroport."""
    payload = {**VALID_PAYLOAD, "destination_airport": VALID_PAYLOAD["origin_airport"]}
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_hhmm_returns_422(api_client):
    """Validateur metier : 1499 est dans [0, 2359] mais invalide en HHMM (minutes=99)."""
    payload = {**VALID_PAYLOAD, "scheduled_departure": 1499}
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_distance_returns_422(api_client):
    payload = {**VALID_PAYLOAD, "distance": -10}
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_anomaly_returns_200_with_null_prediction(api_client):
    """Une valeur hors plage (mais syntaxiquement valide) ne renvoie pas d'erreur HTTP."""
    payload = {**VALID_PAYLOAD, "distance": 999_999}
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["anomaly"] is True
    assert body["delayed"] is None


def test_metrics_reflects_request_count(api_client):
    api_client.post("/predict", json=VALID_PAYLOAD)
    api_client.post("/predict", json=VALID_PAYLOAD)
    response = api_client.get("/metrics")
    assert response.status_code == 200
    assert response.json()["total_requests"] == 2
