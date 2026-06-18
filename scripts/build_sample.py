"""Construit l'échantillon d'entraînement à partir du flights.csv complet de Kaggle.

Usage:
    python scripts/build_sample.py
    python scripts/build_sample.py --input data/raw/flights_full.csv --n-rows 200000

Ne garde que les variables connues AVANT le départ du vol (pas de fuite d'information
type DEPARTURE_DELAY, TAXI_OUT, AIR_TIME... qui ne sont disponibles qu'après le vol).
"""

import argparse

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "MONTH",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
]


def build_sample(input_path: str, output_path: str, n_rows: int, seed: int) -> pd.DataFrame:
    """Charge le CSV complet, construit la cible et échantillonne n_rows lignes.

    Entrées: chemin du flights.csv complet, taille d'échantillon, graine aléatoire.
    Sorties: DataFrame échantillonné (features + colonne `delayed`), écrit aussi en CSV.
    Dépendances: pandas, numpy. Le fichier d'entrée doit contenir les colonnes du
    dataset Kaggle usdot/flight-delays (ARRIVAL_DELAY, CANCELLED, DIVERTED, etc.).
    """
    usecols = FEATURE_COLUMNS + ["ARRIVAL_DELAY", "CANCELLED", "DIVERTED"]
    df = pd.read_csv(input_path, usecols=usecols, low_memory=False)

    # Un vol annulé ou détourné n'a pas de retard d'arrivée exploitable.
    df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]
    df = df.dropna(subset=["ARRIVAL_DELAY"] + FEATURE_COLUMNS)

    df["delayed"] = (df["ARRIVAL_DELAY"] > 15).astype(int)
    df = df[FEATURE_COLUMNS + ["delayed"]]

    n_rows = min(n_rows, len(df))
    sample = df.sample(n=n_rows, random_state=seed).reset_index(drop=True)

    sample.to_csv(output_path, index=False)
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/flights_full.csv")
    parser.add_argument("--output", default="data/raw/flights_sample.csv")
    parser.add_argument("--n-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample = build_sample(args.input, args.output, args.n_rows, args.seed)
    print(f"Échantillon écrit : {args.output} ({len(sample)} lignes)")
    print(f"Taux de retard (delayed=1) : {sample['delayed'].mean():.2%}")


if __name__ == "__main__":
    main()
