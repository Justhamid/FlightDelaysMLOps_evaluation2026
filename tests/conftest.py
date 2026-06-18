"""Fixtures pytest partagees : un petit modele entraine sur un echantillon fixe
(tests/fixtures/flights_fixture.csv), injecte directement dans l'API sans passer par
MLflow/le Model Registry. Garantit que les tests sont rapides et autonomes (pas de
serveur MLflow ni de dataset complet necessaires en CI).
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app, state
from src.pipeline import evaluate, prepare, train

FIXTURE_PATH = "tests/fixtures/flights_fixture.csv"


@pytest.fixture(scope="session")
def trained_pipeline():
    """Entraine un petit modele sur le fixture, reutilise par tous les tests de la session."""
    X_train, X_test, y_train, y_test, feature_columns = prepare(FIXTURE_PATH, test_size=0.3)
    model = train(X_train, y_train, n_estimators=20, max_depth=5, class_weight="balanced")
    metrics = evaluate(model, X_test, y_test)
    return {
        "model": model,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
    }


@pytest.fixture
def api_client(trained_pipeline):
    """TestClient avec le modele de test injecte directement dans l'etat de l'API.

    Le TestClient est instancie sans bloc `with` : le lifespan (qui tenterait de
    charger le modele depuis MLflow) n'est donc jamais declenche, et notre modele de
    test reste en place pour toute la duree du test.
    """
    reference = pd.read_csv(FIXTURE_PATH)[["DISTANCE", "SCHEDULED_TIME"]]
    state["model"] = trained_pipeline["model"]
    state["feature_columns"] = trained_pipeline["feature_columns"]
    state["reference"] = reference
    state["request_window"].clear()
    state["metrics"] = {
        "total_requests": 0,
        "total_predictions_delayed": 0,
        "total_predictions_on_time": 0,
        "total_anomalies": 0,
        "total_latency_ms": 0.0,
    }
    return TestClient(app)
