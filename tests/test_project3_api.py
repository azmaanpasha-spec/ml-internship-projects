"""Smoke tests for the recommendation API.

Requires the fitted artefact, so these skip cleanly if `python -m src.evaluate` has not
been run in project 3 yet.
"""

from __future__ import annotations

import pytest

from conftest import ROOT, use_project

ARTEFACT = ROOT / "project3_recommendation" / "models" / "recommenders.joblib"

pytestmark = pytest.mark.skipif(
    not ARTEFACT.exists(),
    reason="run `python -m src.evaluate` in project3_recommendation first",
)


@pytest.fixture(scope="module")
def client():
    use_project(3)
    from fastapi.testclient import TestClient

    from src.api import app

    with TestClient(app) as test_client:      # the context manager runs the lifespan
        yield test_client


def test_health_reports_the_loaded_models(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert "hybrid" in body["models"]
    assert body["n_items"] > 0
    assert body["training_cutoff"] is not None


def test_models_endpoint_documents_every_loaded_model(client):
    body = client.get("/models").json()
    for name in body["models"]:
        assert name in body["notes"], f"{name} is served but undocumented"


def test_recommend_returns_a_ranked_list(client):
    body = client.get("/recommend/42", params={"k": 10}).json()

    assert body["user_id"] == 42
    assert len(body["items"]) == 10
    # Ranks must be 1..k in order.
    assert [item["rank"] for item in body["items"]] == list(range(1, 11))
    for item in body["items"]:
        assert item["title"]
        assert item["price"] > 0


def test_recommend_honours_k(client):
    for k in (1, 5, 25):
        body = client.get("/recommend/7", params={"k": k}).json()
        assert len(body["items"]) == k


def test_recommend_rejects_an_unknown_user(client):
    assert client.get("/recommend/99999999").status_code == 404


def test_recommend_rejects_an_unknown_model(client):
    response = client.get("/recommend/42", params={"model": "not_a_model"})
    assert response.status_code == 404
    assert "unknown model" in response.json()["detail"]


def test_recommend_validates_k(client):
    assert client.get("/recommend/42", params={"k": 0}).status_code == 422
    assert client.get("/recommend/42", params={"k": 5000}).status_code == 422


def test_different_models_give_different_lists(client):
    cf = client.get("/recommend/42", params={"model": "item_item_cf"}).json()
    popularity = client.get("/recommend/42", params={"model": "popularity"}).json()

    cf_items = [item["item_id"] for item in cf["items"]]
    popular_items = [item["item_id"] for item in popularity["items"]]
    assert cf_items != popular_items, "personalisation had no effect"


def test_user_history_is_consistent_with_the_segment(client):
    body = client.get("/users/42/history").json()
    assert body["user_id"] == 42
    assert body["n_interactions"] >= 0
    if body["n_interactions"] > 0:
        assert body["top_categories"]


def test_similar_items_returns_scored_neighbours(client):
    body = client.get("/similar/100", params={"k": 5, "model": "als_mf"}).json()

    assert body["item_id"] == 100
    assert len(body["items"]) == 5
    assert 100 not in [item["item_id"] for item in body["items"]], "item returned itself"

    similarities = [item["similarity"] for item in body["items"]]
    assert all(score is not None for score in similarities)
    assert similarities == sorted(similarities, reverse=True), "neighbours not ranked"


def test_similar_items_rejects_a_model_without_item_similarity(client):
    response = client.get("/similar/100", params={"model": "popularity"})
    assert response.status_code == 400


def test_segments_endpoint_breaks_down_by_user_history(client):
    body = client.get("/segments", params={"k": 10, "model": "item_item_cf"}).json()
    assert body["segments"], "no segments returned"
    for stats in body["segments"].values():
        assert stats["mean_price"] > 0
        assert stats["distinct_items"] > 0
        assert stats["top_categories"]
