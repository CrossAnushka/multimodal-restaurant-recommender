"""Late fusion of the three modality embeddings into one item content vector.

Late fusion = encode each modality independently, then combine at the feature
stage. For each modality we:

  1. standardize (z-score, stats from training restaurants),
  2. reduce to a common-ish size with PCA (so a 768-d text block doesn't swamp
     a 20-d structured block),
  3. L2-normalize the block and apply a per-modality weight.

The blocks are concatenated into the **item content vector**. Because each block
is unit-norm, cosine similarity in this space is just the (weighted) average of
the per-modality cosine similarities -- which makes content-based filtering and
the modality ablations clean and interpretable.

The fitted statistics are stored as plain arrays in an ``.npz`` so loading never
depends on a pickled scikit-learn object.
"""

from __future__ import annotations

import numpy as np

from ..config import CONFIG, FUSION_STATS_NPZ, Config

MODALITIES = ("text", "image", "structured")


class LateFusion:
    """Fit/transform the per-modality standardize -> PCA -> L2 pipeline."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {m: 1.0 for m in MODALITIES}
        self.stats: dict[str, dict] = {}
        self.block_dims: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def fit(self, mats: dict[str, np.ndarray], train_idx: np.ndarray,
            cfg: Config | None = None) -> "LateFusion":
        from sklearn.decomposition import PCA

        cfg = cfg or CONFIG
        proj = cfg.model.fusion_proj_dim
        for m in MODALITIES:
            X = mats[m]
            Xtr = X[train_idx]
            mean = Xtr.mean(0)
            std = Xtr.std(0) + 1e-8
            Ztr = (Xtr - mean) / std
            k = int(min(proj, Ztr.shape[1], max(1, Ztr.shape[0] - 1)))
            pca = PCA(n_components=k, random_state=cfg.model.seed)
            pca.fit(Ztr)
            self.stats[m] = {
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
                "pca_mean": pca.mean_.astype(np.float32),
                "components": pca.components_.astype(np.float32),  # (k, d)
            }
            self.block_dims[m] = k
        return self

    # ------------------------------------------------------------------ #
    def _block(self, m: str, X: np.ndarray) -> np.ndarray:
        st = self.stats[m]
        Z = (X - st["mean"]) / st["std"]
        P = (Z - st["pca_mean"]) @ st["components"].T
        norm = np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
        return (P / norm) * self.weights.get(m, 1.0)

    def transform(self, mats: dict[str, np.ndarray],
                  use: set[str] | None = None) -> np.ndarray:
        """Build the content matrix. ``use`` selects which modalities are active
        (others are zeroed) -- this is how the ablation study drops a modality."""

        use = use or set(MODALITIES)
        blocks = []
        for m in MODALITIES:
            B = self._block(m, mats[m])
            if m not in use:
                B = np.zeros_like(B)
            blocks.append(B)
        return np.concatenate(blocks, axis=1).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return sum(self.block_dims[m] for m in MODALITIES)

    # ------------------------------------------------------------------ #
    def save(self, path=FUSION_STATS_NPZ) -> None:
        flat = {"__modalities__": np.array(MODALITIES)}
        flat["__weights__"] = np.array([self.weights[m] for m in MODALITIES],
                                       dtype=np.float32)
        for m in MODALITIES:
            for key, arr in self.stats[m].items():
                flat[f"{m}__{key}"] = arr
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **flat)

    @classmethod
    def load(cls, path=FUSION_STATS_NPZ) -> "LateFusion":
        from ..utils import require
        require(path, "run `train` first (fusion stats are fit during training)")
        data = np.load(path, allow_pickle=True)
        mods = list(data["__modalities__"])
        weights = {m: float(w) for m, w in zip(mods, data["__weights__"])}
        obj = cls(weights=weights)
        for m in mods:
            obj.stats[m] = {
                "mean": data[f"{m}__mean"],
                "std": data[f"{m}__std"],
                "pca_mean": data[f"{m}__pca_mean"],
                "components": data[f"{m}__components"],
            }
            obj.block_dims[m] = obj.stats[m]["components"].shape[0]
        return obj
