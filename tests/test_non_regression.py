"""Tests de non-regression : detectent qu'une modification future casserait
silencieusement le pipeline ou le contrat de l'API, sans viser une performance precise."""

from src.pipeline import evaluate, prepare, train

FIXTURE_PATH = "tests/fixtures/flights_fixture.csv"
MIN_ACCEPTABLE_F1 = 0.05  # seuil bas : detecte un modele casse, pas une regression de qualite


def test_model_beats_minimal_f1_baseline():
    X_train, X_test, y_train, y_test, _ = prepare(FIXTURE_PATH, test_size=0.3)
    model = train(X_train, y_train, n_estimators=20, max_depth=5, class_weight="balanced")
    metrics = evaluate(model, X_test, y_test)
    assert metrics["f1"] >= MIN_ACCEPTABLE_F1


def test_model_predictions_are_deterministic_for_fixed_seed():
    X_train, X_test, y_train, _, _ = prepare(FIXTURE_PATH, test_size=0.3, random_state=42)
    model_a = train(X_train, y_train, n_estimators=20, max_depth=5, random_state=42)
    model_b = train(X_train, y_train, n_estimators=20, max_depth=5, random_state=42)
    assert (model_a.predict(X_test) == model_b.predict(X_test)).all()


def test_predict_response_schema_is_stable(api_client):
    """Garantit que le contrat de /predict (les cles de la reponse) ne change pas sans le vouloir."""
    response = api_client.post(
        "/predict",
        json={
            "month": 7,
            "day_of_week": 3,
            "airline": "AA",
            "origin_airport": "JFK",
            "destination_airport": "LAX",
            "scheduled_departure": 1430,
            "scheduled_time": 360,
            "distance": 600,
        },
    )
    assert set(response.json().keys()) == {"delayed", "probability", "anomaly", "message"}
