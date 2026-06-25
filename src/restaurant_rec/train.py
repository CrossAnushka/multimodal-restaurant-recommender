"""Training orchestration.

Order of operations (frozen encoders are assumed already cached):

  1. Fit the late-fusion statistics on the **training** restaurants and build the
     item content matrix.
  2. Train collaborative-filtering matrix factorization on training interactions.
  3. Build user taste profiles from training interactions.
  4. Train the unified ranker on training interactions (CF + content + cross).
  5. Train the cold-start regressors (content -> CF) on warm entities only.

All artifacts are written under ``artifacts/`` for evaluate / recommend to load.
"""

from __future__ import annotations

import numpy as np

from .config import CONFIG, Config
from .data import schema as S
from .data.generate import load_dataset
from .fusion.late_fusion import LateFusion
from .models import coldstart as cs
from .models import collaborative as cf
from .models.content import CrossFeatureBuilder, build_user_profiles
from .models.ranker import build_features, save_ranker, train_ranker
from .pipeline import get_modality_matrices, train_ratings, warm_masks


def train_all(cfg: Config | None = None, verbose: bool = True) -> dict:
    cfg = cfg or CONFIG
    data = load_dataset()
    mats = get_modality_matrices(data, cfg)
    tr = train_ratings(data)
    warm_user, warm_item = warm_masks(data)

    n_users = int(data["users"][S.USER_ID].max()) + 1
    n_items = int(data["restaurants"][S.REST_ID].max()) + 1

    # 1. Late fusion (fit on warm items so cold items never leak into the stats).
    if verbose:
        print("[1/5] fitting late fusion ...")
    fusion = LateFusion().fit(mats, np.where(warm_item)[0], cfg)
    content = fusion.transform(mats)
    fusion.save()
    if verbose:
        print(f"      content vector dim = {content.shape[1]} "
              f"(blocks: {fusion.block_dims})")

    # 2. Collaborative filtering.
    if verbose:
        print("[2/5] training collaborative filtering ...")
    cf_model = cf.train_cf(tr, n_users, n_items, cfg, verbose=verbose)
    cf.save_cf(cf_model, n_users, n_items, cfg)
    emb = cf.get_embeddings(cf_model)
    user_cf, item_cf = emb["user_emb"], emb["item_emb"]

    # 3. User taste profiles.
    if verbose:
        print("[3/5] building user taste profiles ...")
    user_profiles = build_user_profiles(content, tr, n_users, cfg)

    # 4. Unified ranker.
    if verbose:
        print("[4/5] training unified ranker ...")
    cross_builder = CrossFeatureBuilder(data["users"], data["restaurants"])
    u = tr[S.RATING_USER_ID].to_numpy()
    i = tr[S.RATING_REST_ID].to_numpy()
    y = tr[S.RATING_VALUE].to_numpy().astype(np.float32)
    cross = cross_builder.pair_features(u, i)
    feats = build_features(u, i, user_cf=user_cf, item_cf=item_cf, content=content,
                           user_profiles=user_profiles, cross=cross)
    ranker = train_ranker(feats, y, cfg, verbose=verbose)
    save_ranker(ranker, feats.shape[1])

    # 5. Cold-start regressors (warm entities only).
    if verbose:
        print("[5/5] training cold-start regressors ...")
    wi = np.where(warm_item)[0]
    wu = np.where(warm_user)[0]
    item_reg = cs.train_regressor(content[wi], item_cf[wi], cfg, name="item",
                                  verbose=verbose)
    user_reg = cs.train_regressor(user_profiles[wu], user_cf[wu], cfg, name="user",
                                  verbose=verbose)
    cs.save_coldstart(item_reg, user_reg, content.shape[1],
                      user_profiles.shape[1], cfg.model.cf_dim)

    if verbose:
        print("done. artifacts written to artifacts/")
    return {"content_dim": content.shape[1], "ranker_input_dim": feats.shape[1]}
