"""Unified hybrid ranking model.

This is the model-level **late fusion**: an MLP that consumes, per (user, item)
pair, the collaborative-filtering vectors *and* the fused multi-modal content
vectors *and* the structured cross features, and predicts a relevance score
(pointwise, regressing the rating). It is deliberately agnostic to where its
inputs come from -- at cold-start time the caller simply swaps in a predicted
CF vector or a preference-based profile, and the same ranker applies.

    input = [ user_cf | item_cf | item_content | user_profile | cross(3) ]
"""

from __future__ import annotations

import numpy as np

from ..config import CONFIG, RANKER_MODEL_PT, Config
from ..utils import get_device, set_seed
from ..data import schema as S


def build_features(users: np.ndarray, items: np.ndarray, *,
                   user_cf: np.ndarray, item_cf: np.ndarray,
                   content: np.ndarray, user_profiles: np.ndarray,
                   cross: np.ndarray) -> np.ndarray:
    """Assemble the ranker input matrix for arrays of (user, item) pairs."""

    return np.concatenate([
        user_cf[users],
        item_cf[items],
        content[items],
        user_profiles[users],
        cross,
    ], axis=1).astype(np.float32)


def _build_module(input_dim: int, cfg: Config):
    import torch.nn as nn

    mc = cfg.model
    layers: list = []
    d = input_dim
    for h in mc.ranker_hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(mc.ranker_dropout)]
        d = h
    layers += [nn.Linear(d, 1)]

    class Ranker(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return Ranker()


def train_ranker(features: np.ndarray, targets: np.ndarray,
                 cfg: Config | None = None, verbose: bool = True):
    """Train the ranker MLP on assembled (features, rating) pairs."""

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    cfg = cfg or CONFIG
    mc = cfg.model
    set_seed(mc.seed)
    device = get_device()

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    model = _build_module(X.shape[1], cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=mc.ranker_lr,
                           weight_decay=mc.ranker_weight_decay)
    loss_fn = torch.nn.MSELoss()

    loader = DataLoader(TensorDataset(X, y), batch_size=mc.ranker_batch_size,
                        shuffle=True)
    model.train()
    for epoch in range(mc.ranker_epochs):
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
            total += loss.item() * len(by)
        if verbose and (epoch % 5 == 0 or epoch == mc.ranker_epochs - 1):
            print(f"  [ranker] epoch {epoch:2d}  train RMSE {(total/len(y))**0.5:.4f}")

    return model.cpu().eval()


def predict(model, features: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return model(torch.tensor(features, dtype=torch.float32)).numpy()


def save_ranker(model, input_dim: int, path=RANKER_MODEL_PT) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim}, path)


def load_ranker(cfg: Config | None = None, path=RANKER_MODEL_PT):
    import torch

    from ..utils import require
    cfg = cfg or CONFIG
    require(path, "run `train` first")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = _build_module(ckpt["input_dim"], cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
