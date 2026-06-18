"""Pose un alias MLflow (ex: "champion") sur une version enregistree du modele.

Par defaut, pointe vers la derniere version enregistree. Permet a l'API de charger
toujours `models:/flight_delay_model@champion` sans connaitre le numero de version.

Usage:
    python scripts/set_model_alias.py --alias champion
"""

import argparse

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "flight_delay_model"


def set_alias(alias: str = "champion", version: str | None = None) -> str:
    """Pointe l'alias donne vers une version du modele enregistre.

    Entrees: nom de l'alias, numero de version (None = la plus recente).
    Sorties: le numero de version sur lequel l'alias a ete pose.
    Depend de: mlflow (tracking URI sqlite configuree dans main.py).
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    if version is None:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        version = max(versions, key=lambda v: int(v.version)).version

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, version)
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--version", default=None)
    args = parser.parse_args()
    v = set_alias(args.alias, args.version)
    print(f"Alias '{args.alias}' -> version {v}")
