"""API d'inference FastAPI pour la prediction de retard de vol.

Charge le modele depuis le MLflow Model Registry (alias "champion", pose par
scripts/set_model_alias.py) au demarrage. Lancer avec:
    uvicorn app:app --reload
puis ouvrir http://127.0.0.1:8000/docs pour la documentation Swagger.
"""

import json
import logging
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("flight_delay_api")

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_URI = "models:/flight_delay_model@champion"
FEATURE_COLUMNS_PATH = "artifacts/feature_columns.json"
CATEGORICAL_COLUMNS = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]

state: dict = {"model": None, "feature_columns": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modele et la liste des colonnes attendues au demarrage de l'API."""
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        state["model"] = mlflow.sklearn.load_model(MODEL_URI)
        with open(FEATURE_COLUMNS_PATH) as f:
            state["feature_columns"] = json.load(f)
        logger.info("Modele charge depuis %s", MODEL_URI)
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
    delayed: int
    probability: float


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
    """Predit si le vol decrit par `payload` aura plus de 15 minutes de retard."""
    if state["model"] is None:
        logger.error("Predict appele alors que le modele n'est pas charge")
        raise HTTPException(status_code=500, detail="Modele indisponible")

    X = encode_request(payload, state["feature_columns"])
    proba = float(state["model"].predict_proba(X)[0][1])
    delayed = int(proba >= 0.5)
    logger.info("Prediction effectuee: delayed=%s proba=%.3f", delayed, proba)
    return PredictionResponse(delayed=delayed, probability=round(proba, 4))
