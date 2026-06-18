"""Tests unitaires des fonctions du pipeline (prepare / train / evaluate / save)."""

import json

import joblib
import pandas as pd

from src.pipeline import evaluate, prepare, save, train

FIXTURE_PATH = "tests/fixtures/flights_fixture.csv"


def test_prepare_returns_consistent_shapes():
    X_train, X_test, y_train, y_test, feature_columns = prepare(FIXTURE_PATH, test_size=0.3)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert list(X_train.columns) == feature_columns
    assert list(X_test.columns) == feature_columns


def test_prepare_handles_single_row_without_dropping_categories():
    """Verifie le correctif drop_first=False : une requete a une seule ligne ne doit
    pas perdre sa colonne categorielle one-hot (piege documente dans src/pipeline.py)."""
    _, _, _, _, feature_columns = prepare(FIXTURE_PATH, test_size=0.3)
    df = pd.read_csv(FIXTURE_PATH)
    one_row = df.iloc[[0]]

    encoded = pd.get_dummies(
        one_row, columns=["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], drop_first=False
    )
    reindexed = encoded.reindex(columns=feature_columns, fill_value=0)

    airline_col = f"AIRLINE_{one_row.iloc[0]['AIRLINE']}"
    assert reindexed[airline_col].iloc[0] == 1


def test_train_returns_fitted_model():
    X_train, _, y_train, _, _ = prepare(FIXTURE_PATH, test_size=0.3)
    model = train(X_train, y_train, n_estimators=10, max_depth=5)
    assert hasattr(model, "predict")
    preds = model.predict(X_train)
    assert len(preds) == len(X_train)


def test_evaluate_returns_expected_metric_keys():
    X_train, X_test, y_train, y_test, _ = prepare(FIXTURE_PATH, test_size=0.3)
    model = train(X_train, y_train, n_estimators=10, max_depth=5)
    metrics = evaluate(model, X_test, y_test)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_save_writes_reloadable_artifacts(tmp_path):
    X_train, X_test, y_train, y_test, feature_columns = prepare(FIXTURE_PATH, test_size=0.3)
    model = train(X_train, y_train, n_estimators=10, max_depth=5)
    metrics = evaluate(model, X_test, y_test)

    paths = save(model, metrics, feature_columns, artifacts_dir=str(tmp_path))

    reloaded_model = joblib.load(paths["model_path"])
    reloaded_columns = json.loads((tmp_path / "feature_columns.json").read_text())
    assert reloaded_columns == feature_columns
    assert reloaded_model.predict(X_test).shape[0] == len(X_test)
