"""FastAPI service + static frontend for the recommender.

The heavy inference state (artifacts + frozen-encoder feature matrices) is built
**once** at startup via :func:`pipeline.assemble_state` and reused across every
request -- never rebuild it per call.

Endpoints
---------
* ``GET  /api/options``          -- form vocabulary (cuisines, cities, price bands)
* ``POST /api/recommend/cold``   -- onboarding-preferences -> cold-start recs
* ``POST /api/recommend/user``   -- existing user id -> warm recs
* ``GET  /images/...``           -- restaurant food photos
* ``GET  /``                     -- the single-page frontend

Each recommendation also carries a per-modality **"why" breakdown** (text /
image / structured), computed by slicing the fused content vector into its
modality blocks and taking the cosine similarity of each block against the
user's taste profile -- a direct, visual demonstration of the late-fusion idea.

Run with::

    uvicorn restaurant_rec.api:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import CONFIG, IMAGE_DIR, PRICE_BANDS
from .data import schema as S
from .fusion.late_fusion import MODALITIES
from .models.content import profile_from_cuisines
from .pipeline import RecState, assemble_state
from .recommend import recommend_cold_user, recommend_for_user

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


# --------------------------------------------------------------------------- #
# Startup state (built once)
# --------------------------------------------------------------------------- #
class _Ctx:
    """Holds everything derived once at startup for fast request handling."""

    state: RecState
    city_centroids: dict[str, tuple[float, float]]
    block_slices: dict[str, tuple[int, int]]
    cuisines: list[dict]   # [{name, count}] sorted by frequency desc
    cities: list[str]


CTX = _Ctx()


def _build_ctx() -> None:
    state = assemble_state(CONFIG)
    rest = state.data["restaurants"]

    # Per-city centroid so the cold-start city selection actually moves the
    # proximity signal (recommend_cold_user accepts explicit lat/lon).
    cent = rest.groupby(S.REST_CITY)[[S.REST_LAT, S.REST_LON]].mean()
    CTX.city_centroids = {c: (float(r[S.REST_LAT]), float(r[S.REST_LON]))
                          for c, r in cent.iterrows()}

    # Modality block boundaries inside the fused content vector.
    slices, start = {}, 0
    for m in MODALITIES:
        d = int(state.fusion.block_dims[m])
        slices[m] = (start, start + d)
        start += d
    CTX.block_slices = slices

    vc = rest[S.REST_CUISINE].value_counts()
    CTX.cuisines = [{"name": str(c), "count": int(n)} for c, n in vc.items()]
    CTX.cities = sorted(str(c) for c in rest[S.REST_CITY].unique())
    CTX.state = state


@asynccontextmanager
async def lifespan(app: FastAPI):
    _build_ctx()
    yield


app = FastAPI(title="Multi-Modal Restaurant Recommender", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# "Why" breakdown
# --------------------------------------------------------------------------- #
def _modality_breakdown(profile: np.ndarray, item_ids: np.ndarray) -> list[dict]:
    """Per-modality cosine similarity between the taste profile and each item.

    The fused content vector is ``[text | image | structured]`` with each block
    L2-normalized; slicing it back out and taking a per-block cosine recovers
    how much each modality supports the recommendation. Clamped to [0, 1].
    """

    content = CTX.state.content[item_ids]
    out = []
    per_block = {}
    for m in MODALITIES:
        a, b = CTX.block_slices[m]
        p = profile[a:b]
        q = content[:, a:b]
        pn = p / (np.linalg.norm(p) + 1e-9)
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        per_block[m] = np.clip(qn @ pn, 0.0, 1.0)
    for idx in range(len(item_ids)):
        out.append({m: round(float(per_block[m][idx]), 3) for m in MODALITIES})
    return out


def _results(df: pd.DataFrame, breakdown: list[dict]) -> list[dict]:
    rest = CTX.state.data["restaurants"].set_index(S.REST_ID)
    items = []
    for (_, row), why in zip(df.iterrows(), breakdown):
        rid = int(row["restaurant_id"])
        img = str(rest.loc[rid, S.REST_IMAGE]).lstrip("/")
        items.append({
            "restaurant_id": rid,
            "name": row["name"],
            "cuisine": row["cuisine"],
            "price": row["price"],                       # "$".."$$$$"
            "price_band": len(str(row["price"])),
            "city": row["city"],
            "score": float(row["score"]),
            "cold_item": bool(row["cold_item"]),
            "image_url": f"/{img}",
            "why": why,
        })
    return items


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class ColdRequest(BaseModel):
    cuisines: list[str] = Field(default_factory=list)
    price: int = 2
    city: str | None = None
    k: int = 9


class UserRequest(BaseModel):
    user_id: int
    k: int = 9


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/options")
def options() -> dict:
    return {
        "cuisines": CTX.cuisines,
        "cities": CTX.cities,
        "price_bands": list(PRICE_BANDS),
        "n_users": int(CTX.state.n_users),
    }


@app.post("/api/recommend/cold")
def recommend_cold(req: ColdRequest) -> dict:
    cuisines = [c.strip() for c in req.cuisines if c.strip()]
    if not cuisines:
        raise HTTPException(400, "Pick at least one cuisine.")
    lat, lon = CTX.city_centroids.get(req.city, (None, None)) if req.city else (None, None)

    df = recommend_cold_user(cuisines, req.price, req.city, lat=lat, lon=lon,
                             k=req.k, state=CTX.state)
    profile = profile_from_cuisines(CTX.state.content,
                                    CTX.state.data["restaurants"], cuisines, req.price)
    breakdown = _modality_breakdown(profile, df["restaurant_id"].to_numpy())
    return {"mode": "cold", "results": _results(df, breakdown)}


@app.post("/api/recommend/user")
def recommend_user(req: UserRequest) -> dict:
    if not (0 <= req.user_id < CTX.state.n_users):
        raise HTTPException(404, f"user_id must be 0..{CTX.state.n_users - 1}")

    df = recommend_for_user(req.user_id, k=req.k, state=CTX.state)
    profile = CTX.state.user_profiles[req.user_id]
    breakdown = _modality_breakdown(profile, df["restaurant_id"].to_numpy())
    cold_user = not bool(CTX.state.warm_user[req.user_id])
    return {"mode": "user", "cold_user": cold_user, "results": _results(df, breakdown)}


# Static assets: food images, then the SPA (mounted last so /api wins).
app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
