"""DistilBERT text encoder (frozen).

Produces one 768-d embedding per restaurant by mean-pooling the token
embeddings of its description plus its listing reviews. Mean-pooling over both
tokens and texts is robust and needs no fine-tuning. Cold restaurants (few/no
reviews) still get an embedding from their description alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import CONFIG, TEXT_EMB_NPY, Config
from ..utils import get_device
from ..data import schema as S


def _mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def _encode_texts(texts: list[str], cfg: Config, batch_size: int = 32) -> np.ndarray:
    """Encode a list of strings into (len(texts), text_dim) mean-pooled vectors."""

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = get_device()
    tok = AutoTokenizer.from_pretrained(cfg.model.text_model_name)
    model = AutoModel.from_pretrained(cfg.model.text_model_name).to(device).eval()

    out = np.zeros((len(texts), cfg.model.text_dim), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="text", leave=False):
            batch = texts[start:start + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            pooled = _mean_pool(hidden, enc["attention_mask"])
            out[start:start + len(batch)] = pooled.cpu().numpy()
    return out


def build_restaurant_text_embeddings(data: dict, cfg: Config | None = None,
                                     save: bool = True) -> np.ndarray:
    """Return (n_restaurants, text_dim) embeddings ordered by restaurant_id."""

    cfg = cfg or CONFIG
    restaurants = data["restaurants"].sort_values(S.REST_ID).reset_index(drop=True)
    reviews = data["reviews"]
    cap = cfg.data.reviews_per_restaurant_cap

    # Collect (restaurant_id, text) for description + capped reviews.
    rid_list: list[int] = []
    text_list: list[str] = []
    for _, row in restaurants.iterrows():
        rid_list.append(int(row[S.REST_ID]))
        text_list.append(str(row[S.REST_DESCRIPTION]))
    rev_by_rest = reviews.groupby(S.REVIEW_REST_ID)[S.REVIEW_TEXT].apply(list).to_dict()
    for rid, texts in rev_by_rest.items():
        for t in texts[:cap]:
            rid_list.append(int(rid))
            text_list.append(str(t))

    per_text = _encode_texts(text_list, cfg)

    # Average per restaurant.
    n_rest = len(restaurants)
    dim = cfg.model.text_dim
    sums = np.zeros((n_rest, dim), dtype=np.float64)
    counts = np.zeros(n_rest, dtype=np.float64)
    for rid, vec in zip(rid_list, per_text):
        sums[rid] += vec
        counts[rid] += 1
    counts = np.clip(counts, 1.0, None)
    emb = (sums / counts[:, None]).astype(np.float32)

    if save:
        TEXT_EMB_NPY.parent.mkdir(parents=True, exist_ok=True)
        np.save(TEXT_EMB_NPY, emb)
    return emb


def load_text_embeddings() -> np.ndarray:
    from ..utils import require
    require(TEXT_EMB_NPY, "run `build-embeddings` first")
    return np.load(TEXT_EMB_NPY)
