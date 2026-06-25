"""Inference: produce a ranked restaurant list for a user.

Two entry points:
  * ``recommend_for_user`` -- an existing (warm or cold) user already in the
    dataset, scored with the full hybrid ranker.
  * ``recommend_cold_user`` -- a brand-new user described only by onboarding
    preferences (cuisines / price / city). We build a preference profile, predict
    a CF vector from it via the cold-start regressor, and rank with the same
    ranker -- demonstrating new-user cold start end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CITIES, CONFIG, Config
from .data import schema as S
from .models import coldstart as cs
from .models.content import profile_from_cuisines
from .models.ranker import build_features
from .models.ranker import predict as ranker_predict
from .pipeline import RecState, assemble_state


def _attach_info(state: RecState, item_ids: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rest = state.data["restaurants"].set_index(S.REST_ID)
    rows = []
    for iid, sc in zip(item_ids, scores):
        r = rest.loc[int(iid)]
        rows.append({
            "restaurant_id": int(iid),
            "score": round(float(sc), 3),
            "name": r[S.REST_NAME],
            "cuisine": r[S.REST_CUISINE],
            "price": "$" * int(r[S.REST_PRICE]),
            "city": r[S.REST_CITY],
            "cold_item": bool(r[S.REST_IS_COLD]),
        })
    return pd.DataFrame(rows)


def _seen_items(state: RecState, user_id: int) -> set:
    r = state.data["ratings"]
    return set(r.loc[r[S.RATING_USER_ID] == user_id, S.RATING_REST_ID].astype(int))


def recommend_for_user(user_id: int, k: int = 10, exclude_seen: bool = True,
                       state: RecState | None = None,
                       cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or CONFIG
    state = state or assemble_state(cfg)

    seen = _seen_items(state, user_id) if exclude_seen else set()
    cands = np.array([i for i in range(state.n_items) if i not in seen])

    cr = state.cross.pair_features(np.full(len(cands), user_id), cands)
    feats = build_features(np.full(len(cands), user_id), cands, user_cf=state.user_cf,
                           item_cf=state.item_cf, content=state.content,
                           user_profiles=state.user_profiles, cross=cr)
    scores = ranker_predict(state.ranker, feats)
    top = np.argsort(-scores)[:k]
    return _attach_info(state, cands[top], scores[top])


def recommend_cold_user(cuisines: list[str], price: int, city: str | None = None,
                        lat: float | None = None, lon: float | None = None,
                        k: int = 10, state: RecState | None = None,
                        cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or CONFIG
    state = state or assemble_state(cfg)

    if lat is None or lon is None:
        if city in CITIES:
            lat, lon = CITIES[city]
        else:  # fall back to dataset centroid
            lat = float(state.data["restaurants"][S.REST_LAT].mean())
            lon = float(state.data["restaurants"][S.REST_LON].mean())

    # Onboarding profile + predicted CF vector for the new user.
    profile = profile_from_cuisines(state.content, state.data["restaurants"],
                                    cuisines, price)
    _item_reg, user_reg = cs.load_coldstart(cfg)
    user_cf_vec = cs.predict(user_reg, profile[None, :])[0]

    cands = np.arange(state.n_items)
    cross = state.cross.features_for_attrs(set(cuisines), price, lat, lon, cands)
    feats = np.concatenate([
        np.tile(user_cf_vec, (len(cands), 1)),
        state.item_cf[cands],
        state.content[cands],
        np.tile(profile, (len(cands), 1)),
        cross,
    ], axis=1).astype(np.float32)
    scores = ranker_predict(state.ranker, feats)
    top = np.argsort(-scores)[:k]
    return _attach_info(state, cands[top], scores[top])
