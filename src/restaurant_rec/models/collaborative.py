"""Collaborative filtering via matrix factorization (PyTorch).

Learns a latent vector + bias per user and per restaurant from the observed
interaction ratings, predicting rating ~ global + user_bias + item_bias +
<user_vec, item_vec>. The learned user/item vectors are reused as features by
the unified ranker; the item vectors are also the target the cold-start
regressor learns to predict from content.

Restaurants/users held out as "cold" still occupy rows in the embedding tables
(so indexing by id always works) but their rows stay near initialization -- we
never trust them directly; cold-start handling replaces them.
"""

from __future__ import annotations

import numpy as np

from ..config import CF_MODEL_PT, CONFIG, Config
from ..utils import get_device, set_seed


def _build_module(n_users: int, n_items: int, dim: int, global_mean: float):
    import torch
    import torch.nn as nn

    class MatrixFactorization(nn.Module):
        def __init__(self):
            super().__init__()
            self.user_emb = nn.Embedding(n_users, dim)
            self.item_emb = nn.Embedding(n_items, dim)
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            self.global_bias = nn.Parameter(torch.tensor(float(global_mean)))
            nn.init.normal_(self.user_emb.weight, std=0.05)
            nn.init.normal_(self.item_emb.weight, std=0.05)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)

        def forward(self, u, i):
            dot = (self.user_emb(u) * self.item_emb(i)).sum(-1)
            return self.global_bias + self.user_bias(u).squeeze(-1) \
                + self.item_bias(i).squeeze(-1) + dot

    return MatrixFactorization()


def train_cf(ratings_train, n_users: int, n_items: int,
             cfg: Config | None = None, verbose: bool = True):
    """Train matrix factorization on the training interactions.

    ``ratings_train`` is a DataFrame with user_id / restaurant_id / rating.
    Returns the trained torch module (on CPU).
    """

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    cfg = cfg or CONFIG
    mc = cfg.model
    set_seed(mc.seed)
    device = get_device()

    from ..data import schema as S
    u = torch.tensor(ratings_train[S.RATING_USER_ID].to_numpy(), dtype=torch.long)
    i = torch.tensor(ratings_train[S.RATING_REST_ID].to_numpy(), dtype=torch.long)
    r = torch.tensor(ratings_train[S.RATING_VALUE].to_numpy(), dtype=torch.float32)

    model = _build_module(n_users, n_items, mc.cf_dim, float(r.mean())).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=mc.cf_lr,
                           weight_decay=mc.cf_weight_decay)
    loss_fn = torch.nn.MSELoss()

    loader = DataLoader(TensorDataset(u, i, r), batch_size=mc.cf_batch_size,
                        shuffle=True)
    model.train()
    for epoch in range(mc.cf_epochs):
        total = 0.0
        for bu, bi, br in loader:
            bu, bi, br = bu.to(device), bi.to(device), br.to(device)
            opt.zero_grad()
            pred = model(bu, bi)
            loss = loss_fn(pred, br)
            loss.backward()
            opt.step()
            total += loss.item() * len(br)
        if verbose and (epoch % 5 == 0 or epoch == mc.cf_epochs - 1):
            rmse = (total / len(r)) ** 0.5
            print(f"  [cf] epoch {epoch:2d}  train RMSE {rmse:.4f}")

    return model.cpu().eval()


def get_embeddings(model) -> dict:
    """Extract embedding tables + biases as numpy arrays."""

    return {
        "user_emb": model.user_emb.weight.detach().numpy(),
        "item_emb": model.item_emb.weight.detach().numpy(),
        "user_bias": model.user_bias.weight.detach().numpy().squeeze(-1),
        "item_bias": model.item_bias.weight.detach().numpy().squeeze(-1),
        "global_bias": float(model.global_bias.detach()),
    }


def save_cf(model, n_users: int, n_items: int, cfg: Config | None = None,
            path=CF_MODEL_PT) -> None:
    import torch

    cfg = cfg or CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "n_users": n_users,
        "n_items": n_items,
        "dim": cfg.model.cf_dim,
        "global_mean": float(model.global_bias.detach()),
    }, path)


def load_cf(path=CF_MODEL_PT):
    import torch

    from ..utils import require
    require(path, "run `train` first")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = _build_module(ckpt["n_users"], ckpt["n_items"], ckpt["dim"],
                          ckpt["global_mean"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
