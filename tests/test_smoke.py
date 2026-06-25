"""Fast smoke tests on a tiny config.

These exercise the data generator and the modeling stack (fusion, CF, ranker,
content profiles, cold-start, ranking metrics) using small in-memory data and
random embeddings -- so they run in seconds without downloading DistilBERT /
ResNet. Full integration with the real encoders is covered by the CLI `all`
run documented in the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from restaurant_rec import config
from restaurant_rec.config import tiny_config
from restaurant_rec.data import schema as S


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def test_generate_dataset(monkeypatch, tmp_path):
    # Redirect CSV writes to a temp dir (these paths are imported at call time).
    for name, fname in [("RESTAURANTS_CSV", "r.csv"), ("USERS_CSV", "u.csv"),
                        ("REVIEWS_CSV", "rev.csv"), ("RATINGS_CSV", "rat.csv")]:
        monkeypatch.setattr(config, name, tmp_path / fname)

    from restaurant_rec.data.generate import generate_dataset

    cfg = tiny_config()
    data = generate_dataset(cfg, write_images=False)
    rest, users, ratings = data["restaurants"], data["users"], data["ratings"]

    assert len(rest) == cfg.data.n_restaurants
    assert len(users) == cfg.data.n_users
    assert list(rest.columns) == S.RESTAURANT_COLUMNS
    assert ratings[S.RATING_VALUE].between(1, 5).all()

    # Cold-start invariant: any interaction touching a cold user/item is test-only.
    cold_u = set(users.loc[users[S.USER_IS_COLD], S.USER_ID])
    cold_i = set(rest.loc[rest[S.REST_IS_COLD], S.REST_ID])
    touches_cold = ratings[S.RATING_USER_ID].isin(cold_u) | ratings[S.RATING_REST_ID].isin(cold_i)
    assert (ratings.loc[touches_cold, S.RATING_SPLIT] == "test").all()
    # ...and cold entities never appear in training.
    train = ratings[ratings[S.RATING_SPLIT] == "train"]
    assert not train[S.RATING_USER_ID].isin(cold_u).any()
    assert not train[S.RATING_REST_ID].isin(cold_i).any()


# --------------------------------------------------------------------------- #
# Late fusion
# --------------------------------------------------------------------------- #
def test_late_fusion_roundtrip(tmp_path):
    from restaurant_rec.fusion.late_fusion import LateFusion

    cfg = tiny_config()
    n = 30
    mats = {
        "text": np.random.randn(n, 64).astype(np.float32),
        "image": np.random.randn(n, 48).astype(np.float32),
        "structured": np.random.randn(n, config.STRUCTURED_DIM).astype(np.float32),
    }
    train_idx = np.arange(20)
    fusion = LateFusion().fit(mats, train_idx, cfg)
    content = fusion.transform(mats)
    assert content.shape == (n, fusion.output_dim)

    # ablation zeroes a modality block
    dropped = fusion.transform(mats, use={"image", "structured"})
    assert np.allclose(dropped[:, : fusion.block_dims["text"]], 0.0)

    path = tmp_path / "fusion.npz"
    fusion.save(path)
    reloaded = LateFusion.load(path)
    assert np.allclose(reloaded.transform(mats), content, atol=1e-5)


# --------------------------------------------------------------------------- #
# Modeling stack
# --------------------------------------------------------------------------- #
def _toy_frames(n_users, n_items):
    rng = np.random.default_rng(0)
    restaurants = pd.DataFrame({
        S.REST_ID: np.arange(n_items),
        S.REST_CUISINE: rng.choice(config.CUISINES, n_items),
        S.REST_PRICE: rng.integers(1, 5, n_items),
        S.REST_CITY: rng.choice(list(config.CITIES), n_items),
        S.REST_LAT: rng.normal(40, 1, n_items),
        S.REST_LON: rng.normal(-100, 1, n_items),
        S.REST_IS_COLD: False,
    })
    users = pd.DataFrame({
        S.USER_ID: np.arange(n_users),
        S.USER_PRICE_PREF: rng.integers(1, 5, n_users),
        S.USER_PREF_CUISINES: ["Italian|Thai"] * n_users,
        S.USER_LAT: rng.normal(40, 1, n_users),
        S.USER_LON: rng.normal(-100, 1, n_users),
    })
    return restaurants, users


def test_model_stack_trains_and_scores():
    from restaurant_rec.models import coldstart as cs
    from restaurant_rec.models import collaborative as cf
    from restaurant_rec.models.content import CrossFeatureBuilder, build_user_profiles
    from restaurant_rec.models.ranker import build_features, predict, train_ranker

    cfg = tiny_config()
    n_users, n_items, content_dim = 25, 18, 12
    rng = np.random.default_rng(0)

    # toy interactions
    n_rat = 200
    ratings = pd.DataFrame({
        S.RATING_USER_ID: rng.integers(0, n_users, n_rat),
        S.RATING_REST_ID: rng.integers(0, n_items, n_rat),
        S.RATING_VALUE: rng.integers(1, 6, n_rat).astype(float),
    })
    cf_model = cf.train_cf(ratings, n_users, n_items, cfg, verbose=False)
    emb = cf.get_embeddings(cf_model)
    assert emb["user_emb"].shape == (n_users, cfg.model.cf_dim)

    content = rng.standard_normal((n_items, content_dim)).astype(np.float32)
    profiles = build_user_profiles(content, ratings, n_users, cfg)
    assert profiles.shape == (n_users, content_dim)

    restaurants, users = _toy_frames(n_users, n_items)
    cross_b = CrossFeatureBuilder(users, restaurants)
    u = ratings[S.RATING_USER_ID].to_numpy()
    i = ratings[S.RATING_REST_ID].to_numpy()
    cross = cross_b.pair_features(u, i)
    feats = build_features(u, i, user_cf=emb["user_emb"], item_cf=emb["item_emb"],
                           content=content, user_profiles=profiles, cross=cross)
    ranker = train_ranker(feats, ratings[S.RATING_VALUE].to_numpy().astype(np.float32),
                          cfg, verbose=False)
    preds = predict(ranker, feats)
    assert preds.shape == (n_rat,) and np.isfinite(preds).all()

    # cold-start regressor maps content -> item CF
    reg = cs.train_regressor(content, emb["item_emb"], cfg, name="item", verbose=False)
    pred_cf = cs.predict(reg, content)
    assert pred_cf.shape == emb["item_emb"].shape


# --------------------------------------------------------------------------- #
# Ranking metrics
# --------------------------------------------------------------------------- #
def test_ranking_metrics_sanity():
    from restaurant_rec.evaluate import ranking_metrics

    n_items, k = 50, 10
    test_pos = {0: {1, 2, 3}}
    train_by_user = {0: set()}

    # perfect scorer ranks the relevant items first
    def perfect(u, cands):
        return np.array([10.0 if c in test_pos[u] else 0.0 for c in cands])

    good = ranking_metrics(perfect, [0], train_by_user, test_pos, n_items, k)
    assert good[f"P@{k}"] == pytest.approx(3 / k)
    assert good[f"R@{k}"] == pytest.approx(1.0)
    assert good[f"NDCG@{k}"] == pytest.approx(1.0)

    # adversarial scorer ranks relevant items last -> zero hits in top-k
    def worst(u, cands):
        return np.array([-10.0 if c in test_pos[u] else 0.0 for c in cands])

    bad = ranking_metrics(worst, [0], train_by_user, test_pos, n_items, k)
    assert bad[f"P@{k}"] == 0.0
