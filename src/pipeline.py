"""Pipeline ML pour la prediction de retard de vol, decoupe en etapes a contrat."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

CATEGORICAL_COLUMNS = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
NUMERIC_COLUMNS = ["MONTH", "DAY_OF_WEEK", "SCHEDULED_DEPARTURE", "SCHEDULED_TIME", "DISTANCE"]
TARGET_COLUMN = "delayed"


def prepare(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """Charge l'echantillon et produit des jeux d'entrainement/test encodes.

    Entrees:
        data_path: chemin du CSV (colonnes brutes definies dans CATEGORICAL_COLUMNS /
            NUMERIC_COLUMNS + la cible `delayed`).
        test_size: proportion reservee au test.
        random_state: graine pour la reproductibilite du split.
    Sorties:
        (X_train, X_test, y_train, y_test, feature_columns) ou feature_columns est la
        liste ordonnee des colonnes apres one-hot encoding, a reutiliser telle quelle
        a l'inference (cf. app.py) pour realigner une requete unique sur ces colonnes.
    Depend de: pandas, scikit-learn.
    """
    df = pd.read_csv(data_path)

    # drop_first=False : a l'inference une requete = une seule ligne, et drop_first=True
    # supprimerait alors systematiquement une categorie au hasard (bug deja rencontre).
    # On reindexe les nouvelles requetes sur `feature_columns` pour les realigner.
    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=False)

    feature_columns = [c for c in encoded.columns if c != TARGET_COLUMN]
    X = encoded[feature_columns]
    y = encoded[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_columns


def train(
    X_train,
    y_train,
    n_estimators: int = 100,
    max_depth: int | None = None,
    class_weight: str | None = None,
    random_state: int = 42,
):
    """Entraine un RandomForestClassifier sur le jeu d'entrainement encode.

    Entrees: X_train/y_train issus de `prepare`, hyperparametres du modele.
        class_weight="balanced" compense le desequilibre de classe (~18% de retards) ;
        sans cela, le modele tend a predire systematiquement "pas de retard".
    Sorties: modele scikit-learn entraine.
    Depend de: scikit-learn.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> dict:
    """Calcule les metriques de classification sur le jeu de test.

    Entrees: modele entraine (sortie de `train`), X_test/y_test issus de `prepare`.
    Sorties: dict {accuracy, precision, recall, f1}.
    Depend de: scikit-learn.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def save(model, metrics: dict, feature_columns: list, artifacts_dir: str = "artifacts") -> dict:
    """Ecrit le modele, les metriques et la liste des colonnes attendues sur disque.

    Entrees: modele entraine, metriques issues de `evaluate`, feature_columns issus de
        `prepare`, dossier de destination.
    Sorties: dict des chemins ecrits (model_path, metrics_path, feature_columns_path).
    Depend de: joblib, json, pathlib. Cree `artifacts_dir` s'il n'existe pas.
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_path / "model.pkl"
    metrics_path = artifacts_path / "metrics.txt"
    feature_columns_path = artifacts_path / "feature_columns.json"

    joblib.dump(model, model_path)
    metrics_path.write_text("\n".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    feature_columns_path.write_text(json.dumps(feature_columns))

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "feature_columns_path": str(feature_columns_path),
    }
