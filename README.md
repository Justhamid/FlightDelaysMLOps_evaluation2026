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

## Briques MLOps couvertes

- [x] Pipeline de données versionné avec DVC
- [ ] Traçabilité des entraînements avec MLflow (≥3 runs comparables)
- [ ] Pipeline en fonctions à contrats (prepare / train / evaluate / save)
- [ ] Orchestration via un DAG Airflow (Docker)
- [ ] API d'inférence FastAPI (`/`, `/health`, `/predict`)
- [ ] Supervision et détection de dérive (`/metrics`, anomalies, dérive)
- [ ] Tests pytest + CI GitHub Actions
- [ ] Proposition d'architecture d'évolution (stockage SQL/NoSQL/Big Data)

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Exécution

_À compléter au fur et à mesure de l'avancement du projet._
