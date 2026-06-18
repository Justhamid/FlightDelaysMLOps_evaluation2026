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

## Briques MLOps couvertes

- [ ] Pipeline de données versionné avec DVC
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
