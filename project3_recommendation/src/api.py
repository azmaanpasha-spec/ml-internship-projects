"""Step 6 of the approach: a recommendation API.

Serves the models fitted by `evaluate.py` from a saved artefact, so startup is instant
rather than refitting ALS on every boot.

    uvicorn src.api:app --reload --port 8000

Then:
    http://127.0.0.1:8000/docs              interactive OpenAPI docs
    /recommend/42?k=10                      personalised list for user 42
    /recommend/42?k=10&model=als_mf         the same user from a different model
    /similar/305?k=10                       item-to-item similarity
    /users/42/history                       what the model knows about that user
    /segments                               recommendation behaviour by user segment

Cold users -- no interaction history at all -- cannot be served by collaborative models.
The API detects that and falls back to time-decayed popularity, reporting the fallback in
the response rather than silently returning something arbitrary.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

from common.io_utils import load_model  # noqa: E402

MODEL_PATH = Path(__file__).absolute().parent.parent / "models" / "recommenders.joblib"
DEFAULT_MODEL = "hybrid"
FALLBACK_MODEL = "popularity_decay_21d"

STATE: Dict[str, object] = {}


# --------------------------------------------------------------------------------------
# Response schemas
# --------------------------------------------------------------------------------------


class ItemOut(BaseModel):
    item_id: int
    title: str
    category: str
    subcategory: str
    brand: str
    price: float
    avg_rating: float
    tags: str
    is_cold_start: bool
    rank: Optional[int] = None
    similarity: Optional[float] = None


class RecommendationsOut(BaseModel):
    user_id: int
    model: str = Field(description="model that produced the list")
    fallback_used: bool = Field(description="true when the user had no history to personalise on")
    k: int
    user_segment: str
    history_size: int
    items: List[ItemOut]


class HistoryOut(BaseModel):
    user_id: int
    user_segment: str
    n_interactions: int
    top_categories: Dict[str, int]
    items: List[ItemOut]


class SimilarOut(BaseModel):
    item_id: int
    model: str
    items: List[ItemOut]


# --------------------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"no fitted models at {MODEL_PATH}. Run `python -m src.evaluate` first."
        )
    artefact = load_model(MODEL_PATH)
    STATE["models"] = artefact["models"]
    STATE["dataset"] = artefact["dataset"]
    STATE["items"] = artefact["dataset"].items.set_index("item_id", drop=False)
    yield
    STATE.clear()


app = FastAPI(
    title="Product Recommendation API",
    description="Personalised recommendations from collaborative, content-based and hybrid models.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _items_payload(item_ids: List[int], ranked: bool = True) -> List[ItemOut]:
    items = STATE["items"]
    payload = []
    for position, item_id in enumerate(item_ids, start=1):
        if item_id not in items.index:
            continue
        row = items.loc[item_id]
        payload.append(
            ItemOut(
                item_id=int(row["item_id"]),
                title=str(row["title"]),
                category=str(row["category"]),
                subcategory=str(row["subcategory"]),
                brand=str(row["brand"]),
                price=float(row["price"]),
                avg_rating=float(row["avg_rating"]),
                tags=str(row["tags"]),
                is_cold_start=bool(row["is_cold_start"]),
                rank=position if ranked else None,
            )
        )
    return payload


def _require_model(name: str):
    models = STATE["models"]
    if name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"unknown model '{name}'. Available: {sorted(models)}",
        )
    return models[name]


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    dataset = STATE.get("dataset")
    return {
        "status": "ok",
        "models": sorted(STATE["models"]),
        "default_model": DEFAULT_MODEL,
        "n_users": dataset.n_users if dataset else 0,
        "n_items": dataset.n_items if dataset else 0,
        "training_cutoff": str(dataset.cutoff) if dataset else None,
    }


@app.get("/models")
def list_models() -> dict:
    """Available models and a one-line description of each."""
    return {
        "models": sorted(STATE["models"]),
        "notes": {
            "popularity": "global interaction counts; the baseline to beat",
            "popularity_decay_21d": "popularity with a 21-day half-life",
            "item_item_cf": "item-item cosine collaborative filtering (best accuracy)",
            "als_mf": "implicit-feedback matrix factorisation (best catalogue coverage)",
            "content_based": "TF-IDF over item metadata; the only model that can serve new items",
            "hybrid": "collaborative filtering, with content substituted for cold items",
        },
    }


@app.get("/recommend/{user_id}", response_model=RecommendationsOut)
def recommend(
    user_id: int,
    k: int = Query(10, ge=1, le=100),
    model: str = Query(DEFAULT_MODEL),
) -> RecommendationsOut:
    from src.data_prep import user_activity_segment

    dataset = STATE["dataset"]
    if not 0 <= user_id < dataset.n_users:
        raise HTTPException(status_code=404, detail=f"user {user_id} not in the dataset")

    history_size = len(dataset.seen_by_user.get(user_id, set()))
    fallback_used = history_size == 0
    # A collaborative model has nothing to work with for a user who has never interacted.
    model_name = FALLBACK_MODEL if fallback_used and model != "content_based" else model
    recommender = _require_model(model_name)

    item_ids = recommender.recommend(user_id, k=k)
    return RecommendationsOut(
        user_id=user_id,
        model=model_name,
        fallback_used=fallback_used,
        k=k,
        user_segment=user_activity_segment(dataset, user_id),
        history_size=history_size,
        items=_items_payload(item_ids),
    )


@app.get("/users/{user_id}/history", response_model=HistoryOut)
def user_history(user_id: int, limit: int = Query(20, ge=1, le=200)) -> HistoryOut:
    from src.data_prep import user_activity_segment

    dataset = STATE["dataset"]
    if not 0 <= user_id < dataset.n_users:
        raise HTTPException(status_code=404, detail=f"user {user_id} not in the dataset")

    interactions = dataset.train_interactions
    user_rows = interactions[interactions["user_id"] == user_id]
    recent = (
        user_rows.sort_values("timestamp", ascending=False)
        .drop_duplicates("item_id")
        .head(limit)
    )
    merged = recent.merge(dataset.items, on="item_id", how="left")

    return HistoryOut(
        user_id=user_id,
        user_segment=user_activity_segment(dataset, user_id),
        n_interactions=int(len(user_rows)),
        top_categories=merged["category"].value_counts().head(5).to_dict(),
        items=_items_payload(recent["item_id"].tolist(), ranked=False),
    )


@app.get("/similar/{item_id}", response_model=SimilarOut)
def similar_items(
    item_id: int,
    k: int = Query(10, ge=1, le=50),
    model: str = Query("als_mf"),
) -> SimilarOut:
    dataset = STATE["dataset"]
    if not 0 <= item_id < dataset.n_items:
        raise HTTPException(status_code=404, detail=f"item {item_id} not in the catalogue")

    recommender = _require_model(model)
    if not hasattr(recommender, "similar_items"):
        raise HTTPException(
            status_code=400,
            detail=f"model '{model}' does not expose item-to-item similarity; "
                   "use als_mf or content_based",
        )

    neighbours = recommender.similar_items(item_id, k=k)
    payload = _items_payload([item for item, _ in neighbours])
    for entry, (_, score) in zip(payload, neighbours):
        entry.similarity = round(float(score), 4)

    return SimilarOut(item_id=item_id, model=model, items=payload)


@app.get("/segments")
def segments(k: int = Query(10, ge=1, le=50), model: str = Query(DEFAULT_MODEL)) -> dict:
    """How recommendations differ across user segments.

    Step 6 of the brief asks for an analysis of recommendations by segment; this returns
    the average price, rating and novelty of what each segment is shown, which is where
    popularity bias becomes visible.
    """
    from src.data_prep import user_activity_segment

    dataset = STATE["dataset"]
    recommender = _require_model(model)
    items = dataset.items.set_index("item_id")
    popularity = dataset.item_popularity()

    by_segment: Dict[str, List[int]] = {}
    for user_id in dataset.eval_users[:2000]:
        by_segment.setdefault(user_activity_segment(dataset, user_id), []).append(user_id)

    out = {}
    for segment, user_ids in sorted(by_segment.items()):
        recommendations = recommender.recommend_batch(user_ids[:400], k=k)
        recommended = [item for ids in recommendations.values() for item in ids]
        if not recommended:
            continue
        subset = items.loc[recommended]
        out[segment] = {
            "users_sampled": min(len(user_ids), 400),
            "mean_price": round(float(subset["price"].mean()), 2),
            "mean_rating": round(float(subset["avg_rating"].mean()), 3),
            "distinct_items": int(len(set(recommended))),
            "mean_train_interactions_of_recommended": round(
                float(popularity[recommended].mean()), 2
            ),
            "top_categories": subset["category"].value_counts().head(3).to_dict(),
        }
    return {"model": model, "k": k, "segments": out}
