"""ResNet18 image encoder (frozen).

Produces one 512-d embedding per restaurant from its food photo, using the
ImageNet-pretrained ResNet18 with the classification head removed (penultimate
global-average-pooled features). The encoder is frozen -- no training here.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..config import CONFIG, DATA_DIR, IMAGE_EMB_NPY, Config
from ..utils import get_device
from ..data import schema as S


def _build_backbone(device):
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()  # keep 512-d penultimate features
    model = model.to(device).eval()
    return model, weights.transforms()


def build_restaurant_image_embeddings(data: dict, cfg: Config | None = None,
                                      save: bool = True) -> np.ndarray:
    """Return (n_restaurants, image_dim) embeddings ordered by restaurant_id."""

    import torch

    cfg = cfg or CONFIG
    restaurants = data["restaurants"].sort_values(S.REST_ID).reset_index(drop=True)
    device = get_device()
    model, preprocess = _build_backbone(device)

    n_rest = len(restaurants)
    emb = np.zeros((n_rest, cfg.model.image_dim), dtype=np.float32)

    batch_imgs, batch_ids = [], []

    def flush():
        if not batch_imgs:
            return
        with torch.no_grad():
            x = torch.stack(batch_imgs).to(device)
            feats = model(x).cpu().numpy()
        for rid, f in zip(batch_ids, feats):
            emb[rid] = f
        batch_imgs.clear()
        batch_ids.clear()

    for _, row in tqdm(restaurants.iterrows(), total=n_rest, desc="image", leave=False):
        path = DATA_DIR / str(row[S.REST_IMAGE])
        img = Image.open(path).convert("RGB")
        batch_imgs.append(preprocess(img))
        batch_ids.append(int(row[S.REST_ID]))
        if len(batch_imgs) >= 64:
            flush()
    flush()

    if save:
        IMAGE_EMB_NPY.parent.mkdir(parents=True, exist_ok=True)
        np.save(IMAGE_EMB_NPY, emb)
    return emb


def load_image_embeddings() -> np.ndarray:
    from ..utils import require
    require(IMAGE_EMB_NPY, "run `build-embeddings` first")
    return np.load(IMAGE_EMB_NPY)
