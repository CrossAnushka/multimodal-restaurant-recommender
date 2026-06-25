"""Evaluation: rating accuracy, ranking quality, ablations, cold-start.

Produces the portfolio narrative:
  * Hybrid vs CF-only vs Content-only           -> fusion helps
  * modality ablations (drop text/image/struct) -> each modality adds lift
  * cold-start scenario                          -> graceful degradation

Relevance convention: a held-out (test) interaction with rating >= like_threshold
is a "relevant" item for that user. Candidates for a user are all restaurants
except the ones in their training history.
"""

from __future__ import annotations

import numpy as np

from .config import CONFIG, Config
from .data import schema as S
from .models import collaborative as cf
from .models.content import CrossFeatureBuilder, build_user_profiles, cosine_scores
from .models.ranker import build_features
from .models.ranker import predict as ranker_predict
from .pipeline import RecState, assemble_state, get_modality_matrices, train_ratings, warm_masks


# --------------------------------------------------------------------------- #
# Ground-truth dictionaries
# --------------------------------------------------------------------------- #
def _interaction_dicts(data: dict, cfg: Config):
    r = data["ratings"]
    train = r[r[S.RATING_SPLIT] == "train"]
    test = r[r[S.RATING_SPLIT] == "test"]
    thr = cfg.model.like_threshold

    train_by_user: dict[int, set] = {}
    for u, i in zip(train[S.RATING_USER_ID], train[S.RATING_REST_ID]):
        train_by_user.setdefault(int(u), set()).add(int(i))

    test_pos_by_user: dict[int, set] = {}
    for u, i, v in zip(test[S.RATING_USER_ID], test[S.RATING_REST_ID],
                       test[S.RATING_VALUE]):
        if v >= thr:
            test_pos_by_user.setdefault(int(u), set()).add(int(i))

    return train_by_user, test_pos_by_user, test


# --------------------------------------------------------------------------- #
# Ranking metrics
# --------------------------------------------------------------------------- #
def ranking_metrics(score_fn, users_subset, train_by_user, test_pos_by_user,
                    n_items: int, k: int, relevant_filter=None) -> dict:
    precisions, recalls, ndcgs, aps = [], [], [], []
    rec_union: set = set()

    idcg_cache = {n: sum(1.0 / np.log2(r + 2) for r in range(min(k, n)))
                  for n in range(1, k + 1)}

    for u in users_subset:
        rel = test_pos_by_user.get(u, set())
        if relevant_filter is not None:
            rel = {i for i in rel if relevant_filter(i)}
        if not rel:
            continue
        seen = train_by_user.get(u, set())
        cands = np.array([it for it in range(n_items) if it not in seen])
        scores = score_fn(u, cands)
        topk = cands[np.argsort(-scores)[:k]]
        rec_union.update(int(x) for x in topk)

        hits = np.array([1.0 if int(it) in rel else 0.0 for it in topk])
        n_hit = hits.sum()
        precisions.append(n_hit / k)
        recalls.append(n_hit / len(rel))
        dcg = float(np.sum(hits / np.log2(np.arange(len(hits)) + 2)))
        ndcgs.append(dcg / idcg_cache.get(min(len(rel), k), 1.0))
        # average precision @ k
        ncorrect, ap = 0, 0.0
        for rank, h in enumerate(hits):
            if h:
                ncorrect += 1
                ap += ncorrect / (rank + 1)
        aps.append(ap / min(len(rel), k))

    n = max(1, len(precisions))
    return {
        f"P@{k}": float(np.sum(precisions) / n),
        f"R@{k}": float(np.sum(recalls) / n),
        f"NDCG@{k}": float(np.sum(ndcgs) / n),
        f"MAP@{k}": float(np.sum(aps) / n),
        "coverage": len(rec_union) / n_items,
        "n_users": len(precisions),
    }


# --------------------------------------------------------------------------- #
# Scoring functions
# --------------------------------------------------------------------------- #
def make_hybrid_score(user_cf, item_cf, content, user_profiles, cross_builder, ranker):
    def f(u, cands):
        cr = cross_builder.pair_features(np.full(len(cands), u), cands)
        feats = build_features(np.full(len(cands), u), cands, user_cf=user_cf,
                               item_cf=item_cf, content=content,
                               user_profiles=user_profiles, cross=cr)
        return ranker_predict(ranker, feats)
    return f


def make_cf_score(state: RecState):
    def f(u, cands):
        return (state.global_bias + state.user_bias[u] + state.item_bias[cands]
                + state.item_cf[cands] @ state.user_cf[u])
    return f


def make_content_score(state: RecState):
    def f(u, cands):
        return cosine_scores(state.user_profiles[u], state.content[cands])
    return f


# --------------------------------------------------------------------------- #
# Rating-prediction metrics (pointwise)
# --------------------------------------------------------------------------- #
def rating_prediction_metrics(state: RecState, test) -> dict:
    u = test[S.RATING_USER_ID].to_numpy()
    i = test[S.RATING_REST_ID].to_numpy()
    y = test[S.RATING_VALUE].to_numpy().astype(np.float32)

    cr = state.cross.pair_features(u, i)
    feats = build_features(u, i, user_cf=state.user_cf, item_cf=state.item_cf,
                           content=state.content, user_profiles=state.user_profiles,
                           cross=cr)
    pred = ranker_predict(state.ranker, feats)
    err = pred - y
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(np.abs(err)))}


# --------------------------------------------------------------------------- #
# High-level evaluations
# --------------------------------------------------------------------------- #
def evaluate_models(state: RecState | None = None, cfg: Config | None = None) -> dict:
    """Compare Hybrid / CF-only / Content-only on warm users' held-out items."""

    cfg = cfg or CONFIG
    state = state or assemble_state(cfg)
    k = cfg.model.eval_k
    train_by_user, test_pos_by_user, test = _interaction_dicts(state.data, cfg)

    warm_users = [u for u in test_pos_by_user if state.warm_user[u]]

    hybrid = make_hybrid_score(state.user_cf, state.item_cf, state.content,
                               state.user_profiles, state.cross, state.ranker)
    results = {
        "Hybrid (fusion)": ranking_metrics(hybrid, warm_users, train_by_user,
                                            test_pos_by_user, state.n_items, k),
        "CF only": ranking_metrics(make_cf_score(state), warm_users, train_by_user,
                                   test_pos_by_user, state.n_items, k),
        "Content only": ranking_metrics(make_content_score(state), warm_users,
                                        train_by_user, test_pos_by_user,
                                        state.n_items, k),
    }
    results["Hybrid (fusion)"].update(rating_prediction_metrics(state, test))
    return results


def evaluate_coldstart(state: RecState | None = None, cfg: Config | None = None) -> dict:
    """Ranking quality for cold users, plus cold-item retrieval for warm users."""

    cfg = cfg or CONFIG
    state = state or assemble_state(cfg)
    k = cfg.model.eval_k
    train_by_user, test_pos_by_user, _test = _interaction_dicts(state.data, cfg)
    hybrid = make_hybrid_score(state.user_cf, state.item_cf, state.content,
                               state.user_profiles, state.cross, state.ranker)

    cold_users = [u for u in test_pos_by_user if not state.warm_user[u]]
    warm_users = [u for u in test_pos_by_user if state.warm_user[u]]

    out = {
        "Cold users (hybrid)": ranking_metrics(
            hybrid, cold_users, train_by_user, test_pos_by_user, state.n_items, k),
        "Cold items -> warm users": ranking_metrics(
            hybrid, warm_users, train_by_user, test_pos_by_user, state.n_items, k,
            relevant_filter=lambda i: not state.warm_item[i]),
    }
    return out


def evaluate_ablations(cfg: Config | None = None) -> dict:
    """Retrain the ranker on each modality subset and report ranking quality.

    CF embeddings are modality-independent and reused; only the content matrix,
    user profiles, and ranker are rebuilt per subset.
    """

    from .fusion.late_fusion import LateFusion
    from .models.ranker import train_ranker

    cfg = cfg or CONFIG
    from .data.generate import load_dataset
    data = load_dataset()
    mats = get_modality_matrices(data, cfg)
    tr = train_ratings(data)
    _, warm_item = warm_masks(data)
    train_by_user, test_pos_by_user, _ = _interaction_dicts(data, cfg)
    warm_users = [u for u in test_pos_by_user if u in train_by_user]  # warm users
    k = cfg.model.eval_k

    cf_model = cf.load_cf()
    emb = cf.get_embeddings(cf_model)
    user_cf, item_cf = emb["user_emb"], emb["item_emb"]
    cross_builder = CrossFeatureBuilder(data["users"], data["restaurants"])
    n_users = user_cf.shape[0]

    fusion = LateFusion().fit(mats, np.where(warm_item)[0], cfg)

    subsets = {
        "All modalities": {"text", "image", "structured"},
        "no text": {"image", "structured"},
        "no image": {"text", "structured"},
        "no structured": {"text", "image"},
        "text only": {"text"},
        "image only": {"image"},
        "structured only": {"structured"},
    }

    u = tr[S.RATING_USER_ID].to_numpy()
    i = tr[S.RATING_REST_ID].to_numpy()
    y = tr[S.RATING_VALUE].to_numpy().astype(np.float32)
    cross_tr = cross_builder.pair_features(u, i)

    results = {}
    for name, use in subsets.items():
        content = fusion.transform(mats, use=use)
        profiles = build_user_profiles(content, tr, n_users, cfg)
        feats = build_features(u, i, user_cf=user_cf, item_cf=item_cf,
                               content=content, user_profiles=profiles, cross=cross_tr)
        ranker = train_ranker(feats, y, cfg, verbose=False)
        score_fn = make_hybrid_score(user_cf, item_cf, content, profiles,
                                     cross_builder, ranker)
        results[name] = ranking_metrics(score_fn, warm_users, train_by_user,
                                        test_pos_by_user, item_cf.shape[0], k)
    return results


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def format_table(title: str, results: dict, columns: list[str]) -> str:
    name_w = max(len(k) for k in results) + 2
    header = " " * name_w + "".join(f"{c:>10}" for c in columns)
    lines = [title, "-" * len(header), header]
    for name, metrics in results.items():
        row = f"{name:<{name_w}}"
        for c in columns:
            v = metrics.get(c)
            row += f"{v:>10.4f}" if isinstance(v, float) else f"{'-':>10}"
        lines.append(row)
    return "\n".join(lines)


def run_full_evaluation(cfg: Config | None = None) -> None:
    cfg = cfg or CONFIG
    k = cfg.model.eval_k
    rank_cols = [f"P@{k}", f"R@{k}", f"NDCG@{k}", f"MAP@{k}", "coverage"]

    state = assemble_state(cfg)

    print()
    print(format_table("Model comparison (warm users)  [+ RMSE/MAE for hybrid]",
                        evaluate_models(state, cfg), rank_cols + ["RMSE", "MAE"]))
    print()
    print(format_table("Cold-start scenario", evaluate_coldstart(state, cfg), rank_cols))
    print()
    print("Modality ablations (retrained ranker per subset) ...")
    print(format_table("Modality ablations (warm users)",
                        evaluate_ablations(cfg), rank_cols))
    print()
