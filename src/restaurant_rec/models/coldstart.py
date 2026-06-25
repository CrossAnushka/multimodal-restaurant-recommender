"""Cold-start handling.

Collaborative filtering has nothing to say about a restaurant or user with no
interactions. We bridge the gap by learning to *predict* the CF latent vector
from content:

  * item regressor: restaurant content vector -> item CF vector
  * user regressor: user taste profile      -> user CF vector

Both are trained only on "warm" entities (those that appear in the training
interactions, where a real CF vector exists). At inference a cold restaurant or
user gets a predicted CF vector that drops straight into the unified ranker, so
the full hybrid model still applies -- gracefully degrading toward content when
that is all we have.
"""

from __future__ import annotations

import numpy as np

from ..config import COLDSTART_MODEL_PT, CONFIG, Config
from ..utils import get_device, set_seed


def _build_regressor(in_dim: int, out_dim: int, hidden):
    import torch.nn as nn

    layers: list = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def train_regressor(X: np.ndarray, Y: np.ndarray, cfg: Config | None = None,
                    name: str = "regressor", verbose: bool = True):
    """Fit a small MLP mapping content (X) -> CF vectors (Y)."""

    import torch

    cfg = cfg or CONFIG
    mc = cfg.model
    set_seed(mc.seed)
    device = get_device()

    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    Yt = torch.tensor(Y, dtype=torch.float32).to(device)
    model = _build_regressor(X.shape[1], Y.shape[1], mc.coldstart_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=mc.coldstart_lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(mc.coldstart_epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), Yt)
        loss.backward()
        opt.step()
        if verbose and (epoch % 20 == 0 or epoch == mc.coldstart_epochs - 1):
            print(f"  [coldstart:{name}] epoch {epoch:2d}  MSE {loss.item():.4f}")

    return model.cpu().eval()


def predict(model, X: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def save_coldstart(item_model, user_model, content_dim: int, profile_dim: int,
                   cf_dim: int, path=COLDSTART_MODEL_PT) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "item_state": item_model.state_dict(),
        "user_state": user_model.state_dict(),
        "content_dim": content_dim,
        "profile_dim": profile_dim,
        "cf_dim": cf_dim,
    }, path)


def load_coldstart(cfg: Config | None = None, path=COLDSTART_MODEL_PT):
    import torch

    from ..utils import require
    cfg = cfg or CONFIG
    require(path, "run `train` first")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    item_model = _build_regressor(ckpt["content_dim"], ckpt["cf_dim"],
                                  cfg.model.coldstart_hidden)
    user_model = _build_regressor(ckpt["profile_dim"], ckpt["cf_dim"],
                                  cfg.model.coldstart_hidden)
    item_model.load_state_dict(ckpt["item_state"])
    user_model.load_state_dict(ckpt["user_state"])
    return item_model.eval(), user_model.eval()
