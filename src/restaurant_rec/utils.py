"""Small shared helpers: seeding, device selection, and IO."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch (if importable) for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch optional for pure-data steps
        pass


def get_device() -> "Any":
    """Return the best available torch device (mps > cuda > cpu)."""

    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def require(path: Path, hint: str) -> Path:
    """Fail fast with a helpful message if an expected artifact is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Expected file not found: {path}\n  -> {hint}"
        )
    return path
