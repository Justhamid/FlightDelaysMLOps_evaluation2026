# Flight Delays MLOps

Système MLOps complet pour prédire le retard d'un vol (`delayed = 1` si `ARRIVAL_DELAY > 15 min`, sinon `0`), à partir du dataset Kaggle [usdot/flight-delays](https://www.kaggle.com/datasets/usdot/flight-delays).

Projet réalisé en équipe : **Hamid, Joan et [Ton prénom]**.

> Statut : en construction — ce README sera complété brique par brique (voir la liste ci-dessous).

## Architecture du projet

```
data/raw/         → échantillon du dataset brut (versionné avec DVC, pas Git)
data/processed/   → données préparées par le pipeline
src/              → pipeline ML (prepare, train, evaluate, save) en fonctions à contrats
scripts/          → scripts utilitaires (échantillonnage, alias MLflow, etc.)
artifacts/        → modèle entraîné, métriques (générés, gitignorés)
dags/             → DAG Airflow orchestrant le pipeline
tests/            → tests pytest (pipeline, API, non-régression)
report/           → éléments du rapport PDF (captures d'écran, schémas)
.github/workflows/→ CI GitHub Actions
```

## Données

Échantillon de 200 000 vols tiré du dataset Kaggle [usdot/flight-delays](https://www.kaggle.com/datasets/usdot/flight-delays), versionné avec **DVC** (pas Git — le CSV brut complet fait 565 Mo).

- Cible binaire `delayed` = 1 si `ARRIVAL_DELAY > 15 min`, sinon 0 (taux de retard sur l'échantillon : ~17.8%)
- Variables conservées : uniquement celles connues **avant le départ** du vol (`MONTH`, `DAY_OF_WEEK`, `AIRLINE`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`, `SCHEDULED_DEPARTURE`, `SCHEDULED_TIME`, `DISTANCE`) — les colonnes post-vol (`DEPARTURE_DELAY`, `TAXI_OUT`, `AIR_TIME`...) sont exclues pour éviter toute fuite d'information
- Récupérer l'échantillon : `dvc pull` (nécessite l'accès au remote DVC local `../dvcstore`, configuré pour ce TP — voir `.dvc/config`)
- Régénérer l'échantillon depuis zéro : télécharger `flights.csv` depuis Kaggle dans `data/raw/flights.csv`, puis `python scripts/build_sample.py --input data/raw/flights.csv`

## Pipeline & MLflow

Le pipeline (`src/pipeline.py`) est découpé en 4 fonctions à contrat : `prepare()`, `train()`, `evaluate()`, `save()`, chacune documentée (entrées/sorties/dépendances). `main.py` les enchaîne et logge tout dans **MLflow** (params, métriques accuracy/precision/recall/F1, artefacts) avec un backend SQLite (`mlflow.db`) qui supporte directement le Model Registry, sans serveur dédié à lancer.

```bash
python main.py --n-estimators 150 --max-depth 15 --class-weight balanced
mlflow ui --backend-store-uri sqlite:///mlflow.db   # puis http://localhost:5000
python scripts/set_model_alias.py --alias champion  # pointe l'alias vers la derniere version
```

## Orchestration Airflow (Docker)

Le DAG `flight_delay_pipeline` (`dags/flight_delay_dag.py`) exécute `check_data >> prepare >> train >> evaluate >> save` en réutilisant directement les fonctions de `src/pipeline.py`. Deux contrôles intégrés : existence + format des données en entrée (`check_data`), existence et relecture de l'artefact modèle en sortie (`save`). Un échec sur une tâche stoppe le DAG (les tâches suivantes passent en `upstream_failed`, jamais exécutées) et ne laisse jamais de modèle partiellement écrit, car `artifacts/model.pkl` n'est remplacé qu'à la toute dernière étape.

```bash
docker compose up -d                       # demarre Postgres + scheduler + webserver
# http://localhost:8080  (admin / admin)
docker compose exec airflow-scheduler airflow dags unpause flight_delay_pipeline
docker compose exec airflow-scheduler airflow dags trigger flight_delay_pipeline
docker compose down                        # pour arreter
```

## API d'inférence

`app.py` expose le modèle (chargé depuis le MLflow Model Registry via l'alias `champion`) en HTTP :

- `GET /` et `GET /health` : statut de l'API et du modèle
- `POST /predict` : prédiction à partir d'un `FlightRequest` (Pydantic), avec 2 validateurs métier au-delà du simple typage : l'heure programmée doit être un format HHMM valide (minutes < 60), et l'aéroport de départ doit différer de celui d'arrivée
- Erreurs HTTP explicites : `422` pour toute requête invalide (typage ou validateur métier), `500` pour les erreurs internes (modèle indisponible, erreur inattendue)
- Documentation interactive sur `/docs` (Swagger)

```bash
uvicorn app:app --reload
# puis http://127.0.0.1:8000/docs
```

## Briques MLOps couvertes

- [x] Pipeline de données versionné avec DVC
- [x] Traçabilité des entraînements avec MLflow (≥3 runs comparables)
- [x] Pipeline en fonctions à contrats (prepare / train / evaluate / save)
- [x] Orchestration via un DAG Airflow (Docker)
- [x] API d'inférence FastAPI (`/`, `/health`, `/predict`)
- [ ] Supervision et détection de dérive (`/metrics`, anomalies, dérive)
- [ ] Tests pytest + CI GitHub Actions
- [ ] Proposition d'architecture d'évolution (stockage SQL/NoSQL/Big Data)

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```
