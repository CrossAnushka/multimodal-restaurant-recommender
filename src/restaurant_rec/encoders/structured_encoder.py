"""Structured-attribute encoder.

Turns cuisine / price band / city / location into a fixed numeric vector:

    [ cuisine one-hot (len CUISINES) | price scaled (1) |
      city one-hot (len CITIES) | lat_norm, lon_norm (2) ]

Layout length == config.STRUCTURED_DIM. Latitude/longitude are standardized
with statistics fitted on the restaurant table so the same transform applies to
brand-new (cold) restaurants at recommendation time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CITIES, CUISINES, STRUCTURED_DIM
from ..data import schema as S

CUISINE_INDEX = {c: i for i, c in enumerate(CUISINES)}
CITY_INDEX = {c: i for i, c in enumerate(CITIES.keys())}
_N_CUIS = len(CUISINES)
_N_CITY = len(CITIES)


def fit_normalizer(restaurants: pd.DataFrame) -> dict:
    return {
        "lat_mean": float(restaurants[S.REST_LAT].mean()),
        "lat_std": float(restaurants[S.REST_LAT].std() + 1e-9),
        "lon_mean": float(restaurants[S.REST_LON].mean()),
        "lon_std": float(restaurants[S.REST_LON].std() + 1e-9),
    }


def encode_one(cuisine: str, price_band: int, city: str, lat: float, lon: float,
               norm: dict) -> np.ndarray:
    vec = np.zeros(STRUCTURED_DIM, dtype=np.float32)
    if cuisine in CUISINE_INDEX:
        vec[CUISINE_INDEX[cuisine]] = 1.0
    vec[_N_CUIS] = (float(price_band) - 1.0) / 3.0  # -> [0, 1]
    if city in CITY_INDEX:
        vec[_N_CUIS + 1 + CITY_INDEX[city]] = 1.0
    vec[_N_CUIS + 1 + _N_CITY] = (lat - norm["lat_mean"]) / norm["lat_std"]
    vec[_N_CUIS + 1 + _N_CITY + 1] = (lon - norm["lon_mean"]) / norm["lon_std"]
    return vec


def encode_df(restaurants: pd.DataFrame, norm: dict | None = None) -> np.ndarray:
    """Return (n_restaurants, STRUCTURED_DIM) ordered by restaurant_id."""

    restaurants = restaurants.sort_values(S.REST_ID).reset_index(drop=True)
    norm = norm or fit_normalizer(restaurants)
    out = np.zeros((len(restaurants), STRUCTURED_DIM), dtype=np.float32)
    for i, row in restaurants.iterrows():
        out[i] = encode_one(
            str(row[S.REST_CUISINE]), int(row[S.REST_PRICE]), str(row[S.REST_CITY]),
            float(row[S.REST_LAT]), float(row[S.REST_LON]), norm,
        )
    return out
