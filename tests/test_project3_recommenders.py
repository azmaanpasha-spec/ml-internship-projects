"""Correctness tests for the recommendation split, models and cold-start fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import use_project


@pytest.fixture(scope="module")
def dataset():
    """A small but structurally complete dataset: cold items, cold users, a time split."""
    use_project(3)
    from src.data_prep import build_dataset
    from src.generate_data import generate

    users, items, interactions = generate(n_users=400, n_items=120, n_days=90, seed=11)
    return build_dataset(interactions, items, users, test_fraction=0.2)


def test_split_is_strictly_chronological(dataset):
    """No training interaction may postdate a test interaction."""
    assert dataset.train_interactions["timestamp"].max() <= dataset.cutoff
    assert dataset.test_interactions["timestamp"].min() > dataset.cutoff


def test_relevance_excludes_items_already_seen_in_training(dataset):
    """Re-recommending a known item is not a win, so it cannot count as relevant."""
    for user_id, relevant in dataset.relevant_by_user.items():
        seen = dataset.seen_by_user.get(user_id, set())
        assert not (relevant & seen), f"user {user_id} has a relevant item already seen"


def test_confidence_weighting_ranks_purchases_above_views(dataset):
    """A purchase must produce a higher confidence than a single view."""
    matrix = dataset.train_matrix
    assert matrix.data.min() > 1.0          # 1 + alpha*log1p(w) with w >= 1
    # The strongest entries should exceed the weakest by a wide margin.
    assert matrix.data.max() > matrix.data.min() * 1.5


def test_cold_start_items_have_no_training_interactions(dataset):
    """The cold-start set is defined by absence, so verify the absence."""
    popularity = dataset.item_popularity()
    for item_id in dataset.cold_start_items:
        assert popularity[item_id] == 0


@pytest.fixture(scope="module")
def models(dataset):
    use_project(3)
    from src.recommenders import build_all

    return build_all(dataset)


def test_every_model_returns_the_requested_number_of_items(dataset, models):
    user_id = dataset.eval_users[0]
    for name, model in models.items():
        recommended = model.recommend(user_id, k=10)
        assert len(recommended) == 10, f"{name} returned {len(recommended)} items"
        assert len(set(recommended)) == 10, f"{name} returned duplicates"


def test_no_model_recommends_an_already_seen_item(dataset, models):
    """`exclude_seen` is the default and must actually hold."""
    for user_id in dataset.eval_users[:25]:
        seen = dataset.seen_by_user.get(user_id, set())
        for name, model in models.items():
            recommended = set(model.recommend(user_id, k=10))
            assert not (recommended & seen), f"{name} re-recommended a seen item"


def test_popularity_recommends_the_most_popular_items(dataset, models):
    popularity = dataset.item_popularity()
    top_by_count = set(np.argsort(-popularity)[:25])
    recommended = models["popularity"].recommend(dataset.eval_users[0], k=5)
    # Allowing for seen-item exclusion, most of the list should come from the head.
    assert len(set(recommended) & top_by_count) >= 3


def test_als_loss_decreases(dataset, models):
    """ALS must monotonically reduce its objective; otherwise the solver is wrong."""
    history = models["als_mf"].loss_history_
    assert len(history) > 2
    assert history[-1] < history[0], "ALS objective did not improve"
    # Allow tiny numerical wobble but no real increase.
    for earlier, later in zip(history, history[1:]):
        assert later <= earlier * 1.01


def test_only_content_based_can_score_cold_items(dataset, models):
    """Collaborative models give cold items no signal; the content model does."""
    cold = sorted(dataset.cold_start_items)
    if not cold:
        pytest.skip("this sample produced no cold-start items")

    user_ids = dataset.eval_users[:50]
    content_scores = np.asarray(models["content_based"].score_users(user_ids))[:, cold]
    cf_scores = np.asarray(models["item_item_cf"].score_users(user_ids))[:, cold]

    assert np.abs(content_scores).sum() > 0, "content model gave cold items nothing"
    assert np.abs(cf_scores).sum() == pytest.approx(0.0, abs=1e-9), (
        "item-item CF produced a non-zero score for an item with no interactions"
    )


def test_hybrid_surfaces_cold_items_that_collaborative_filtering_cannot(dataset, models):
    """The cold-item substitution is the hybrid's reason to exist."""
    cold = set(dataset.cold_start_items)
    if not cold:
        pytest.skip("this sample produced no cold-start items")

    user_ids = dataset.eval_users[:200]
    hybrid = models["hybrid"].recommend_batch(user_ids, k=20)
    cf = models["item_item_cf"].recommend_batch(user_ids, k=20)

    hybrid_cold = {item for items in hybrid.values() for item in items} & cold
    cf_cold = {item for items in cf.values() for item in items} & cold

    assert len(cf_cold) == 0
    assert len(hybrid_cold) > 0, "hybrid never surfaced a cold-start item"


def test_percentile_ranks_give_tied_scores_the_same_rank():
    """Tie handling decides whether a rank blend is meaningful at all.

    Breaking ties by array position would hand an uninformative item a high percentile.
    """
    use_project(3)
    from src.recommenders import HybridRecommender

    scores = np.array([[0.0, 0.0, 0.0, 0.0, 5.0, 9.0]])
    ranks = HybridRecommender._to_percentile_ranks(scores)

    # The four zeros must share one rank.
    assert len(set(np.round(ranks[0, :4], 10))) == 1
    # And it must be well below the informative items' ranks.
    assert ranks[0, 0] < ranks[0, 4] < ranks[0, 5]
    assert ranks[0, 5] == pytest.approx(1.0)


def test_ranking_metrics_order_the_models_sensibly(dataset, models):
    """A sanity check on the whole stack: CF should beat the popularity baseline."""
    use_project(3)
    from src.evaluate import score_recommendations

    scored = {}
    for name in ("popularity", "item_item_cf"):
        recommendations = models[name].recommend_batch(dataset.eval_users, k=10)
        scored[name] = score_recommendations(
            recommendations, dataset.relevant_by_user, dataset.eval_users, k_values=(10,)
        )
    assert scored["item_item_cf"]["ndcg@10"] > scored["popularity"]["ndcg@10"]
