"""DAG Airflow orchestrant le pipeline d'entrainement flight delays.

check_data >> prepare >> train >> evaluate >> save

Controles integres :
- check_data : le fichier de donnees existe et contient les colonnes attendues
- save : l'artefact modele a bien ete ecrit sur disque et est rechargeable

Isolation des echecs : chaque tache ecrit ses resultats intermediaires dans des
fichiers temporaires (`_prepared.pkl`, `_model_tmp.pkl`) ; le fichier final
`artifacts/model.pkl` n'est ecrase qu'a la toute derniere etape (save), donc un
echec dans train/evaluate ne laisse jamais de modele partiellement ecrit. Avec le
trigger_rule par defaut d'Airflow, un echec sur une tache arrete le DAG : les
taches suivantes ne s'executent pas (pas de propagation de donnees invalides).
"""

import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_DIR = "/opt/airflow/project"
sys.path.insert(0, PROJECT_DIR)

DATA_PATH = f"{PROJECT_DIR}/data/raw/flights_sample.csv"
ARTIFACTS_DIR = f"{PROJECT_DIR}/artifacts"
REQUIRED_COLUMNS = [
    "MONTH",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
    "delayed",
]


def check_data(**context):
    """Verifie que le fichier de donnees existe et contient les colonnes attendues."""
    import os

    import pandas as pd

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Fichier de donnees introuvable: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, nrows=5)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {DATA_PATH}: {missing}")


def prepare_task(**context):
    """Appelle prepare() du pipeline et sauvegarde le resultat pour les taches suivantes."""
    import joblib

    from src.pipeline import prepare

    result = prepare(DATA_PATH)
    joblib.dump(result, f"{ARTIFACTS_DIR}/_prepared.pkl")


def train_task(**context):
    """Appelle train() sur les donnees preparees par la tache precedente."""
    import joblib

    from src.pipeline import train

    X_train, _, y_train, _, _ = joblib.load(f"{ARTIFACTS_DIR}/_prepared.pkl")
    model = train(X_train, y_train, n_estimators=150, max_depth=15, class_weight="balanced")
    joblib.dump(model, f"{ARTIFACTS_DIR}/_model_tmp.pkl")


def evaluate_task(**context):
    """Appelle evaluate() et transmet les metriques a la tache save via XCom."""
    import joblib

    from src.pipeline import evaluate

    _, X_test, _, y_test, _ = joblib.load(f"{ARTIFACTS_DIR}/_prepared.pkl")
    model = joblib.load(f"{ARTIFACTS_DIR}/_model_tmp.pkl")
    metrics = evaluate(model, X_test, y_test)
    context["ti"].xcom_push(key="metrics", value=metrics)


def save_task(**context):
    """Appelle save() puis verifie que l'artefact modele est bien ecrit et lisible."""
    import os

    import joblib

    from src.pipeline import save

    _, _, _, _, feature_columns = joblib.load(f"{ARTIFACTS_DIR}/_prepared.pkl")
    model = joblib.load(f"{ARTIFACTS_DIR}/_model_tmp.pkl")
    metrics = context["ti"].xcom_pull(key="metrics", task_ids="evaluate")

    save(model, metrics, feature_columns, artifacts_dir=ARTIFACTS_DIR)

    model_path = f"{ARTIFACTS_DIR}/model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Le modele n'a pas ete sauvegarde correctement.")
    joblib.load(model_path)

    os.remove(f"{ARTIFACTS_DIR}/_prepared.pkl")
    os.remove(f"{ARTIFACTS_DIR}/_model_tmp.pkl")


with DAG(
    dag_id="flight_delay_pipeline",
    description="Pipeline d'entrainement flight delays (prepare/train/evaluate/save)",
    start_date=datetime(2026, 6, 1),
    schedule=None,
    catchup=False,
    tags=["flight-delays", "mlops"],
) as dag:
    t_check = PythonOperator(task_id="check_data", python_callable=check_data)
    t_prepare = PythonOperator(task_id="prepare", python_callable=prepare_task)
    t_train = PythonOperator(task_id="train", python_callable=train_task)
    t_evaluate = PythonOperator(task_id="evaluate", python_callable=evaluate_task)
    t_save = PythonOperator(task_id="save", python_callable=save_task)

    t_check >> t_prepare >> t_train >> t_evaluate >> t_save
