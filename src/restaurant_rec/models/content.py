"""Content-based filtering.

A user's taste profile is the rating-weighted average of the content vectors of
the restaurants they liked. Scoring a candidate is then cosine similarity
between the profile and the candidate's content vector. The same profiles feed
the unified ranker, and a cuisine-based profile builder bootstraps brand-new
(cold) users from their onboarding preferences.

This module also builds the pairwise **cross features** (cuisine match, price
fit, location proximity) that give the ranker explicit structured signal.
"""

from __future__ import annotations

import numpy as np

from ..config import CONFIG, Config
from ..data import schema as S


# --------------------------------------------------------------------------- #
# User taste profiles
# --------------------------------------------------------------------------- #
def build_user_profiles(content: np.ndarray, ratings_train, n_users: int,
                        cfg: Config | None = None) -> np.ndarray:
    """Return (n_users, content_dim) L2-normalized taste profiles."""

    cfg = cfg or CONFIG
    dim = content.shape[1]
    u = ratings_train[S.RATING_USER_ID].to_numpy()
    i = ratings_train[S.RATING_REST_ID].to_numpy()
    r = ratings_train[S.RATING_VALUE].to_numpy().astype(np.float32)

    # Emphasize liked places; ignore clearly disliked ones.
    w = np.clip(r - 2.5, 0.0, None)

    prof = np.zeros((n_users, dim), dtype=np.float64)
    wsum = np.zeros(n_users, dtype=np.float64)
    np.add.at(prof, u, w[:, None] * content[i])
    np.add.at(wsum, u, w)
    nz = wsum > 0
    prof[nz] /= wsum[nz][:, None]

    norm = np.linalg.norm(prof, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (prof / norm).astype(np.float32)


def profile_from_cuisines(content: np.ndarray, restaurants, cuisines: list[str],
                          price_pref: int | None = None) -> np.ndarray:
    """Cold-start user profile: mean content vector of restaurants matching the
    user's preferred cuisines (optionally nudged toward their price band)."""

    mask = restaurants[S.REST_CUISINE].isin(cuisines).to_numpy()
    if price_pref is not None:
        same_price = (restaurants[S.REST_PRICE] == price_pref).to_numpy()
        if (mask & same_price).sum() >= 3:
            mask = mask & same_price
    ids = restaurants.loc[mask, S.REST_ID].to_numpy()
    if len(ids) == 0:
        return np.zeros(content.shape[1], dtype=np.float32)
    prof = content[ids].mean(0)
    norm = np.linalg.norm(prof) + 1e-9
    return (prof / norm).astype(np.float32)


def cosine_scores(profile: np.ndarray, content: np.ndarray) -> np.ndarray:
    """Cosine similarity of one profile against every item (n_items,)."""

    cn = content / (np.linalg.norm(content, axis=1, keepdims=True) + 1e-9)
    p = profile / (np.linalg.norm(profile) + 1e-9)
    return cn @ p


# --------------------------------------------------------------------------- #
# Pairwise cross features
# --------------------------------------------------------------------------- #
class CrossFeatureBuilder:
    """Builds (cuisine_match, price_fit, proximity) for (user, item) pairs."""

    N_FEATURES = 3

    def __init__(self, users, restaurants):
        n_u = int(users[S.USER_ID].max()) + 1
        n_i = int(restaurants[S.REST_ID].max()) + 1

        self.user_price = np.zeros(n_u, dtype=np.float32)
        self.user_lat = np.zeros(n_u, dtype=np.float32)
        self.user_lon = np.zeros(n_u, dtype=np.float32)
        self.user_pref: list[set[str]] = [set() for _ in range(n_u)]
        for _, row in users.iterrows():
            uid = int(row[S.USER_ID])
            self.user_price[uid] = float(row[S.USER_PRICE_PREF])
            self.user_lat[uid] = float(row[S.USER_LAT])
            self.user_lon[uid] = float(row[S.USER_LON])
            prefs = str(row[S.USER_PREF_CUISINES])
            self.user_pref[uid] = set(prefs.split("|")) if prefs else set()

        self.item_cuisine = np.empty(n_i, dtype=object)
        self.item_price = np.zeros(n_i, dtype=np.float32)
        self.item_lat = np.zeros(n_i, dtype=np.float32)
        self.item_lon = np.zeros(n_i, dtype=np.float32)
        for _, row in restaurants.iterrows():
            iid = int(row[S.REST_ID])
            self.item_cuisine[iid] = str(row[S.REST_CUISINE])
            self.item_price[iid] = float(row[S.REST_PRICE])
            self.item_lat[iid] = float(row[S.REST_LAT])
            self.item_lon[iid] = float(row[S.REST_LON])

    def pair_features(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        users = np.asarray(users)
        items = np.asarray(items)
        cuisine_match = np.array([
            1.0 if self.item_cuisine[i] in self.user_pref[u] else 0.0
            for u, i in zip(users, items)
        ], dtype=np.float32)
        price_fit = 1.0 - np.abs(self.item_price[items] - self.user_price[users]) / 3.0
        dist = np.sqrt((self.user_lat[users] - self.item_lat[items]) ** 2
                       + (self.user_lon[users] - self.item_lon[items]) ** 2)
        proximity = np.exp(-dist / 1.5).astype(np.float32)
        return np.stack([cuisine_match, price_fit, proximity], axis=1)

    def features_for_attrs(self, pref_cuisines: set[str], price_pref: int,
                           lat: float, lon: float, items: np.ndarray) -> np.ndarray:
        """Cross features for an ad-hoc (cold) user against many items."""

        items = np.asarray(items)
        cuisine_match = np.array(
            [1.0 if self.item_cuisine[i] in pref_cuisines else 0.0 for i in items],
            dtype=np.float32)
        price_fit = 1.0 - np.abs(self.item_price[items] - float(price_pref)) / 3.0
        dist = np.sqrt((lat - self.item_lat[items]) ** 2
                       + (lon - self.item_lon[items]) ** 2)
        proximity = np.exp(-dist / 1.5).astype(np.float32)
        return np.stack([cuisine_match, price_fit, proximity], axis=1)
