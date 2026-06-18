"""API d'inference FastAPI pour la prediction de retard de vol.

Charge le modele depuis le MLflow Model Registry (alias "champion", pose par
scripts/set_model_alias.py) au demarrage. Lancer avec:
    uvicorn app:app --reload
puis ouvrir http://127.0.0.1:8000/docs pour la documentation Swagger.
"""

import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.stats import ks_2samp

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/api.log")],
)
logger = logging.getLogger("flight_delay_api")

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_URI = "models:/flight_delay_model@champion"
FEATURE_COLUMNS_PATH = "artifacts/feature_columns.json"
REFERENCE_SAMPLE_PATH = "artifacts/reference_sample.csv"
CATEGORICAL_COLUMNS = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
ANOMALY_COLUMNS = ["DISTANCE", "SCHEDULED_TIME"]
DRIFT_WINDOW_SIZE = 200
DRIFT_MIN_SAMPLES = 30
DRIFT_P_VALUE_THRESHOLD = 0.05

state: dict = {
    "model": None,
    "feature_columns": None,
    "reference": None,
    "request_window": deque(maxlen=DRIFT_WINDOW_SIZE),
    "metrics": {
        "total_requests": 0,
        "total_predictions_delayed": 0,
        "total_predictions_on_time": 0,
        "total_anomalies": 0,
        "total_latency_ms": 0.0,
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modele, les colonnes attendues et l'echantillon de reference au demarrage."""
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        state["model"] = mlflow.sklearn.load_model(MODEL_URI)
        with open(FEATURE_COLUMNS_PATH) as f:
            state["feature_columns"] = json.load(f)
        state["reference"] = pd.read_csv(REFERENCE_SAMPLE_PATH)
        logger.info("Modele et echantillon de reference charges (%s)", MODEL_URI)
    except Exception:
        logger.error("Echec du chargement du modele au demarrage", exc_info=True)
        state["model"] = None
    yield


app = FastAPI(
    title="Flight Delays API",
    description="Predit si un vol aura plus de 15 minutes de retard a l'arrivee.",
    version="1.0.0",
    lifespan=lifespan,
)


class FlightRequest(BaseModel):
    month: int = Field(ge=1, le=12, description="Mois du vol (1-12)")
    day_of_week: int = Field(ge=1, le=7, description="Jour de la semaine (1=lundi .. 7=dimanche)")
    airline: str = Field(min_length=1, description="Code compagnie, ex: 'AA'")
    origin_airport: str = Field(min_length=1, description="Code aeroport de depart")
    destination_airport: str = Field(min_length=1, description="Code aeroport d'arrivee")
    scheduled_departure: int = Field(ge=0, le=2359, description="Heure programmee au format HHMM, ex: 1430")
    scheduled_time: float = Field(gt=0, description="Duree de vol programmee en minutes")
    distance: float = Field(gt=0, description="Distance du vol en miles")

    @field_validator("scheduled_departure")
    @classmethod
    def validate_hhmm_format(cls, v: int) -> int:
        """Verifie que l'entier represente bien une heure HHMM valide (minutes < 60).

        Un simple Field(ge=0, le=2359) laisserait passer des valeurs comme 1380 ou 999
        (minutes >= 60), syntaxiquement dans la plage mais invalides metier.
        """
        if v % 100 >= 60:
            raise ValueError("scheduled_departure doit etre au format HHMM valide (minutes < 60)")
        return v

    @model_validator(mode="after")
    def validate_different_airports(self) -> "FlightRequest":
        """Un vol ne peut pas avoir le meme aeroport de depart et d'arrivee."""
        if self.origin_airport == self.destination_airport:
            raise ValueError("origin_airport et destination_airport doivent etre differents")
        return self


class PredictionResponse(BaseModel):
    delayed: int | None
    probability: float | None
    anomaly: bool = False
    message: str | None = None


def encode_request(payload: FlightRequest, feature_columns: list) -> pd.DataFrame:
    """Encode une requete unique sur les memes colonnes que celles vues a l'entrainement.

    Entrees: requete validee par Pydantic, feature_columns issus de `prepare()`
        (sauvegardes dans artifacts/feature_columns.json par `save()`).
    Sorties: DataFrame a une ligne, reindexee sur feature_columns (0 pour les colonnes
        absentes de cette requete), directement utilisable par `model.predict`.
    Depend de: pandas. `get_dummies(drop_first=False)` puis `reindex` est necessaire car
    une requete unique ne contient qu'une seule valeur par colonne categorielle : sans
    cette etape, les colonnes generees ne correspondraient pas a celles de l'entrainement.
    """
    raw = pd.DataFrame(
        [
            {
                "MONTH": payload.month,
                "DAY_OF_WEEK": payload.day_of_week,
                "AIRLINE": payload.airline,
                "ORIGIN_AIRPORT": payload.origin_airport,
                "DESTINATION_AIRPORT": payload.destination_airport,
                "SCHEDULED_DEPARTURE": payload.scheduled_departure,
                "SCHEDULED_TIME": payload.scheduled_time,
                "DISTANCE": payload.distance,
            }
        ]
    )
    encoded = pd.get_dummies(raw, columns=CATEGORICAL_COLUMNS, drop_first=False)
    return encoded.reindex(columns=feature_columns, fill_value=0)


def detect_anomaly(payload: FlightRequest) -> str | None:
    """Detecte une anomalie ponctuelle : une valeur numerique hors de la plage observee
    a l'entrainement (artifacts/reference_sample.csv).

    Differe des contraintes Pydantic (qui valident un format/type) : ici on compare la
    valeur a la distribution reelle vue pendant l'entrainement (ex: une distance de
    50000 miles est un entier positif valide, mais aucun vol d'entrainement n'en approche).
    Entrees: requete validee. Sorties: message d'anomalie, ou None si rien d'anormal.
    """
    reference = state["reference"]
    if reference is None:
        return None

    values = {"DISTANCE": payload.distance, "SCHEDULED_TIME": payload.scheduled_time}
    for column in ANOMALY_COLUMNS:
        col_min, col_max = reference[column].min(), reference[column].max()
        value = values[column]
        if not (col_min <= value <= col_max):
            return f"{column} = {value} hors de la plage observee a l'entrainement [{col_min:.0f}, {col_max:.0f}]"
    return None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Renvoie un 422 explicite (typage Pydantic ou validateur metier) avec le detail."""
    logger.warning("Requete invalide sur %s: %s", request.url.path, exc.errors())
    return JSONResponse(status_code=422, content=jsonable_encoder({"detail": exc.errors()}))


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Filet de securite : toute erreur non prevue renvoie un 500 propre (pas de stacktrace cote client)."""
    logger.error("Erreur non geree sur %s", request.url.path, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Erreur interne du serveur"})


@app.get("/")
def root():
    return {"name": "Flight Delays API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: FlightRequest):
    """Predit si le vol decrit par `payload` aura plus de 15 minutes de retard.

    Une anomalie ponctuelle (valeur hors plage) bloque la prediction (HTTP 200,
    `delayed`/`probability` a null) sans lever d'erreur HTTP : la requete est valide
    syntaxiquement, juste statistiquement suspecte. Une derive (cf. /drift_report) est
    un signal different : elle n'empeche jamais une prediction individuelle.
    """
    if state["model"] is None:
        logger.error("Predict appele alors que le modele n'est pas charge")
        raise HTTPException(status_code=500, detail="Modele indisponible")

    start = time.perf_counter()
    state["metrics"]["total_requests"] += 1

    anomaly_reason = detect_anomaly(payload)
    if anomaly_reason:
        state["metrics"]["total_anomalies"] += 1
        logger.warning("Anomalie detectee sur /predict: %s", anomaly_reason)
        return PredictionResponse(delayed=None, probability=None, anomaly=True, message=anomaly_reason)

    state["request_window"].append({"DISTANCE": payload.distance, "SCHEDULED_TIME": payload.scheduled_time})

    X = encode_request(payload, state["feature_columns"])
    proba = float(state["model"].predict_proba(X)[0][1])
    delayed = int(proba >= 0.5)

    state["metrics"]["total_predictions_delayed" if delayed else "total_predictions_on_time"] += 1
    state["metrics"]["total_latency_ms"] += (time.perf_counter() - start) * 1000

    logger.info("Prediction effectuee: delayed=%s proba=%.3f", delayed, proba)
    return PredictionResponse(delayed=delayed, probability=round(proba, 4))


@app.get("/metrics")
def metrics():
    """Expose les indicateurs de supervision de l'API (compteurs en memoire, reinitialises au redemarrage)."""
    m = state["metrics"]
    avg_latency = m["total_latency_ms"] / m["total_requests"] if m["total_requests"] else 0.0
    return {
        "total_requests": m["total_requests"],
        "total_predictions_delayed": m["total_predictions_delayed"],
        "total_predictions_on_time": m["total_predictions_on_time"],
        "total_anomalies_detected": m["total_anomalies"],
        "average_latency_ms": round(avg_latency, 2),
        "drift_window_size": len(state["request_window"]),
    }


@app.get("/drift_report")
def drift_report():
    """Detecte une derive de distribution sur la fenetre glissante des dernieres requetes.

    Compare (test de Kolmogorov-Smirnov) la distribution recente des entrees de /predict
    a l'echantillon de reference issu de l'entrainement, colonne par colonne. p-value <
    0.05 => les deux distributions sont jugees significativement differentes (derive).
    """
    window = state["request_window"]
    if len(window) < DRIFT_MIN_SAMPLES:
        return {"status": "not_enough_data", "window_size": len(window), "required": DRIFT_MIN_SAMPLES}

    window_df = pd.DataFrame(window)
    reference = state["reference"]
    details = {}
    drift_detected = False
    for column in ANOMALY_COLUMNS:
        statistic, p_value = ks_2samp(reference[column], window_df[column])
        column_drift = bool(p_value < DRIFT_P_VALUE_THRESHOLD)
        drift_detected = drift_detected or column_drift
        details[column] = {
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": column_drift,
        }

    if drift_detected:
        logger.warning("Derive detectee sur la fenetre glissante: %s", details)
    else:
        logger.info("Pas de derive detectee sur la fenetre glissante (%s requetes)", len(window))

    return {"window_size": len(window), "drift_detected": drift_detected, "details": details}
