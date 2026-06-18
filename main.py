"""Orchestrateur du pipeline d'entrainement, avec tracage MLflow (params, metriques,
artefacts) et enregistrement du modele dans le Model Registry.

Usage:
    python main.py --n-estimators 100 --max-depth 10
"""

import argparse

import mlflow
import mlflow.sklearn

from src.pipeline import evaluate, prepare, save, train

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "flight_delays"
REGISTERED_MODEL_NAME = "flight_delay_model"


def run(
    data_path: str,
    n_estimators: int,
    max_depth: int | None,
    class_weight: str | None = None,
    test_size: float = 0.2,
) -> dict:
    """Execute prepare -> train -> evaluate -> save et logge tout dans MLflow.

    Entrees: chemin des donnees, hyperparametres du modele, proportion de test.
    Sorties: dict des metriques calculees sur le jeu de test.
    Depend de: src.pipeline (prepare/train/evaluate/save), mlflow.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("test_size", test_size)

        X_train, X_test, y_train, y_test, feature_columns = prepare(data_path, test_size=test_size)
        model = train(
            X_train,
            y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
        )
        metrics = evaluate(model, X_test, y_test)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        paths = save(model, metrics, feature_columns)
        mlflow.log_artifact(paths["metrics_path"])
        mlflow.log_artifact(paths["feature_columns_path"])
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"Run termine. Metriques: {metrics}")
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/flights_sample.csv")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--class-weight", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    run(args.data, args.n_estimators, args.max_depth, args.class_weight, args.test_size)


if __name__ == "__main__":
    main()
