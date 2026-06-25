"""Shared plumbing used by train / evaluate / recommend.

Centralizes the assembly of the per-user / per-item feature arrays so the three
entry points stay DRY and consistent: modality matrices -> fused content ->
CF vectors (with cold-start substitutions) -> taste profiles -> cross features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CONFIG, Config
from .data import schema as S
from .data.generate import load_dataset
from .encoders.image_encoder import build_restaurant_image_embeddings, load_image_embeddings
from .encoders.structured_encoder import encode_df
from .encoders.text_encoder import build_restaurant_text_embeddings, load_text_embeddings
from .fusion.late_fusion import LateFusion
from .models import coldstart as cs
from .models import collaborative as cf
from .models.content import CrossFeatureBuilder, build_user_profiles, profile_from_cuisines
from .models.ranker import load_ranker


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def build_and_cache_embeddings(data: dict | None = None, cfg: Config | None = None) -> None:
    """Run the frozen encoders and cache text + image embeddings to disk."""

    cfg = cfg or CONFIG
    data = data or load_dataset()
    build_restaurant_text_embeddings(data, cfg)
    build_restaurant_image_embeddings(data, cfg)


def get_modality_matrices(data: dict, cfg: Config | None = None) -> dict:
    """Return {'text','image','structured'} matrices ordered by restaurant_id."""

    cfg = cfg or CONFIG
    return {
        "text": load_text_embeddings(),
        "image": load_image_embeddings(),
        "structured": encode_df(data["restaurants"]),
    }


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #
def warm_masks(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (over id index) of warm users / warm items."""

    rest, users = data["restaurants"], data["users"]
    n_i = int(rest[S.REST_ID].max()) + 1
    n_u = int(users[S.USER_ID].max()) + 1
    warm_item = np.ones(n_i, dtype=bool)
    warm_user = np.ones(n_u, dtype=bool)
    warm_item[rest.loc[rest[S.REST_IS_COLD], S.REST_ID].to_numpy()] = False
    warm_user[users.loc[users[S.USER_IS_COLD], S.USER_ID].to_numpy()] = False
    return warm_user, warm_item


def train_ratings(data: dict) -> pd.DataFrame:
    r = data["ratings"]
    return r[r[S.RATING_SPLIT] == "train"]


# --------------------------------------------------------------------------- #
# Inference state
# --------------------------------------------------------------------------- #
@dataclass
class RecState:
    cfg: Config
    data: dict
    mats: dict
    fusion: LateFusion
    content: np.ndarray          # (n_items, content_dim)
    user_cf: np.ndarray          # (n_users, cf_dim) cold users substituted
    item_cf: np.ndarray          # (n_items, cf_dim) cold items substituted
    user_bias: np.ndarray        # (n_users,)
    item_bias: np.ndarray        # (n_items,)
    global_bias: float
    user_profiles: np.ndarray    # (n_users, content_dim) cold users substituted
    cross: CrossFeatureBuilder
    ranker: object
    warm_user: np.ndarray
    warm_item: np.ndarray
    active_modalities: set = field(default_factory=lambda: {"text", "image", "structured"})

    @property
    def n_users(self) -> int:
        return self.user_cf.shape[0]

    @property
    def n_items(self) -> int:
        return self.item_cf.shape[0]


def assemble_state(cfg: Config | None = None, apply_coldstart: bool = True,
                   active_modalities: set | None = None) -> RecState:
    """Load every saved artifact and build the arrays needed for scoring."""

    cfg = cfg or CONFIG
    data = load_dataset()
    mats = get_modality_matrices(data, cfg)
    warm_user, warm_item = warm_masks(data)
    tr = train_ratings(data)

    fusion = LateFusion.load()
    use = active_modalities or {"text", "image", "structured"}
    content = fusion.transform(mats, use=use)

    cf_model = cf.load_cf()
    emb = cf.get_embeddings(cf_model)
    user_cf = emb["user_emb"].copy()
    item_cf = emb["item_emb"].copy()

    user_profiles = build_user_profiles(content, tr, user_cf.shape[0], cfg)

    # Cold-start substitutions: predict CF vectors / profiles for cold entities.
    if apply_coldstart:
        item_model, user_model = cs.load_coldstart(cfg)
        rest = data["restaurants"]
        cold_items = np.where(~warm_item)[0]
        if len(cold_items):
            item_cf[cold_items] = cs.predict(item_model, content[cold_items])

        users = data["users"]
        cold_users = np.where(~warm_user)[0]
        for u in cold_users:
            prefs = str(users.loc[users[S.USER_ID] == u, S.USER_PREF_CUISINES].iloc[0])
            price = int(users.loc[users[S.USER_ID] == u, S.USER_PRICE_PREF].iloc[0])
            prof = profile_from_cuisines(content, rest, prefs.split("|"), price)
            user_profiles[u] = prof
        if len(cold_users):
            user_cf[cold_users] = cs.predict(user_model, user_profiles[cold_users])

    cross = CrossFeatureBuilder(data["users"], data["restaurants"])
    ranker = load_ranker(cfg)

    return RecState(
        cfg=cfg, data=data, mats=mats, fusion=fusion, content=content,
        user_cf=user_cf, item_cf=item_cf,
        user_bias=emb["user_bias"].copy(), item_bias=emb["item_bias"].copy(),
        global_bias=emb["global_bias"],
        user_profiles=user_profiles,
        cross=cross, ranker=ranker, warm_user=warm_user, warm_item=warm_item,
        active_modalities=use,
    )
