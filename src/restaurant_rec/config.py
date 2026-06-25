"""Central configuration: paths, dimensions, seeds, and hyperparameters.

Everything tunable lives here so the rest of the package never hard-codes a
magic number. The defaults are sized to run end-to-end on a laptop CPU/MPS in
a few minutes while still producing meaningful evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

# Generated tabular data
RESTAURANTS_CSV = DATA_DIR / "restaurants.csv"
USERS_CSV = DATA_DIR / "users.csv"
REVIEWS_CSV = DATA_DIR / "reviews.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"

# Cached embeddings (one-time cost from the frozen encoders).
# Restaurant-level: text = description + listing reviews, image = food photo(s).
# User content profiles are derived from item content vectors (see models/content.py),
# so no user-authored text pass is needed.
TEXT_EMB_NPY = ARTIFACT_DIR / "text_embeddings.npy"
IMAGE_EMB_NPY = ARTIFACT_DIR / "image_embeddings.npy"

# Trained model weights
CF_MODEL_PT = ARTIFACT_DIR / "cf_model.pt"
RANKER_MODEL_PT = ARTIFACT_DIR / "ranker_model.pt"
COLDSTART_MODEL_PT = ARTIFACT_DIR / "coldstart_model.pt"
FUSION_STATS_NPZ = ARTIFACT_DIR / "fusion_stats.npz"


# --------------------------------------------------------------------------- #
# Vocabulary of structured attributes
# --------------------------------------------------------------------------- #
CUISINES = [
    "Italian", "Chinese", "Indian", "Mexican", "Japanese", "Thai",
    "American", "French", "Korean", "Mediterranean", "Vietnamese", "Greek",
]

# Each city has a centre (lat, lon) used to place restaurants & users nearby.
CITIES = {
    "Metropolis": (40.71, -74.01),
    "Riverside":  (34.05, -118.24),
    "Lakeview":   (41.88, -87.63),
    "Bayport":    (37.77, -122.42),
    "Hillcrest":  (47.61, -122.33),
}

PRICE_BANDS = [1, 2, 3, 4]  # $ .. $$$$


@dataclass(frozen=True)
class DataConfig:
    """Knobs controlling synthetic dataset generation."""

    n_restaurants: int = 300
    n_users: int = 1500
    n_ratings: int = 20000
    reviews_per_restaurant_cap: int = 40  # text embedding aggregates up to this many
    images_per_restaurant: int = 1
    latent_dim: int = 32                  # hidden taste / style factors
    image_size: int = 96                  # generated food image side length (px)
    # Fraction of restaurants / users held out as "cold" (no training interactions).
    cold_restaurant_frac: float = 0.08
    cold_user_frac: float = 0.08
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """Architecture + training hyperparameters."""

    # Frozen encoders
    text_model_name: str = "distilbert-base-uncased"
    text_dim: int = 768
    image_dim: int = 512  # ResNet18 penultimate features

    # Late fusion: each modality is projected to `fusion_proj_dim` then concatenated.
    fusion_proj_dim: int = 64

    # Collaborative filtering (matrix factorization)
    cf_dim: int = 24
    cf_epochs: int = 20
    cf_lr: float = 5e-3
    cf_weight_decay: float = 2e-4
    cf_batch_size: int = 1024

    # Unified ranker (MLP)
    ranker_hidden: tuple[int, ...] = (256, 128, 64)
    ranker_dropout: float = 0.2
    ranker_epochs: int = 30
    ranker_lr: float = 1e-3
    ranker_weight_decay: float = 1e-5
    ranker_batch_size: int = 1024

    # Cold-start content->CF regressor
    coldstart_hidden: tuple[int, ...] = (128,)
    coldstart_epochs: int = 60
    coldstart_lr: float = 1e-3

    # Evaluation
    eval_k: int = 10
    test_frac: float = 0.2
    like_threshold: float = 4.0  # rating >= this counts as a positive/"relevant" item

    seed: int = 42


# Structured feature layout (built by encoders.structured_encoder):
#   one-hot cuisine (len CUISINES) + price (1) + city one-hot (len CITIES) + lat/lon (2)
STRUCTURED_DIM = len(CUISINES) + 1 + len(CITIES) + 2


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


# Default singleton used across the package and CLI.
CONFIG = Config()


def tiny_config() -> Config:
    """A miniature config for fast smoke tests (no heavy model downloads needed
    when encoders are mocked)."""

    return Config(
        data=DataConfig(
            n_restaurants=24,
            n_users=40,
            n_ratings=400,
            reviews_per_restaurant_cap=4,
            images_per_restaurant=1,
            latent_dim=8,
            image_size=48,
            cold_restaurant_frac=0.15,
            cold_user_frac=0.15,
            seed=0,
        ),
        model=ModelConfig(
            fusion_proj_dim=8,
            cf_dim=8,
            cf_epochs=3,
            ranker_hidden=(32, 16),
            ranker_epochs=3,
            coldstart_epochs=5,
            eval_k=5,
            seed=0,
        ),
    )


def ensure_dirs() -> None:
    """Create the data/ and artifacts/ trees if missing."""

    for d in (DATA_DIR, IMAGE_DIR, ARTIFACT_DIR):
        d.mkdir(parents=True, exist_ok=True)
