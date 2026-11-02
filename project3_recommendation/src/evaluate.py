"""Step 5 of the approach: Precision@K, Recall@K and NDCG@K on a time-based split.

Beyond the headline ranking metrics, this module reports the things that decide whether a
recommender is deployable:

* **Per-segment accuracy.** One average hides everything. A model can look fine overall
  while being useless for users with five interactions, who are most of the base.
* **Catalogue coverage and novelty.** A recommender that always returns the same 50
  best-sellers scores respectably and adds no value over a merchandising rule. Coverage
  is the share of the catalogue that ever gets recommended; novelty is the mean
  self-information of recommended items.
* **Cold-start recall.** Measured specifically on the items that have no training
  interactions, which is the only honest way to show the content-based fallback works.
* **Hybrid weights tuned on a validation split** carved out of the *training* period, so
  the test set is never used to make a modelling decision.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

from common.evaluation import (  # noqa: E402
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from common.io_utils import save_json, save_model  # noqa: E402
from common.plotting import PALETTE, plot_metric_bars  # noqa: E402
from src.data_prep import Dataset, build_dataset, load_dataset, user_activity_segment  # noqa: E402
from src.recommenders import (  # noqa: E402
    ALSMatrixFactorization,
    ContentBasedRecommender,
    HybridRecommender,
    ItemItemCF,
    PopularityRecommender,
    Recommender,
    build_all,
)

warnings.filterwarnings("ignore")

K_VALUES = (5, 10, 20)
MAX_K = max(K_VALUES)

WEIGHT_CANDIDATES: List[Dict[str, float]] = [
    # Single-model references, so the blend has to earn its place.
    {"als_mf": 1.0, "item_item_cf": 0.0, "content_based": 0.0, "popularity_decay_21d": 0.0},
    {"als_mf": 0.0, "item_item_cf": 1.0, "content_based": 0.0, "popularity_decay_21d": 0.0},
    # Blends, weighted towards collaborative filtering, which is the strongest component.
    {"als_mf": 0.50, "item_item_cf": 0.50, "content_based": 0.00, "popularity_decay_21d": 0.00},
    {"als_mf": 0.30, "item_item_cf": 0.70, "content_based": 0.00, "popularity_decay_21d": 0.00},
    {"als_mf": 0.25, "item_item_cf": 0.65, "content_based": 0.10, "popularity_decay_21d": 0.00},
    {"als_mf": 0.30, "item_item_cf": 0.55, "content_based": 0.10, "popularity_decay_21d": 0.05},
    {"als_mf": 0.40, "item_item_cf": 0.35, "content_based": 0.15, "popularity_decay_21d": 0.10},
    {"als_mf": 0.20, "item_item_cf": 0.80, "content_based": 0.00, "popularity_decay_21d": 0.00},
]


def score_recommendations(
    recommendations: Dict[int, List[int]],
    relevant_by_user: Dict[int, set],
    user_ids: Sequence[int],
    k_values: Sequence[int] = K_VALUES,
) -> Dict[str, float]:
    """Mean ranking metrics over `user_ids`."""
    accumulator: Dict[str, List[float]] = {}

    for user_id in user_ids:
        recommended = recommendations.get(user_id, [])
        relevant = relevant_by_user.get(user_id, set())
        if not relevant:
            continue
        for k in k_values:
            accumulator.setdefault(f"precision@{k}", []).append(
                precision_at_k(recommended, relevant, k)
            )
            accumulator.setdefault(f"recall@{k}", []).append(recall_at_k(recommended, relevant, k))
            accumulator.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(recommended, relevant, k))
            accumulator.setdefault(f"map@{k}", []).append(
                average_precision_at_k(recommended, relevant, k)
            )
            accumulator.setdefault(f"hitrate@{k}", []).append(
                hit_rate_at_k(recommended, relevant, k)
            )

    return {metric: round(float(np.mean(values)), 5) for metric, values in accumulator.items()}


def catalogue_metrics(
    recommendations: Dict[int, List[int]], dataset: Dataset, k: int = 10
) -> Dict[str, float]:
    """Coverage, novelty and popularity bias of the recommendation lists."""
    popularity = dataset.item_popularity()
    total = popularity.sum()
    # Self-information of an item: rare items carry more.
    with np.errstate(divide="ignore"):
        novelty_per_item = -np.log2(np.where(popularity > 0, popularity / max(total, 1), 1e-9))

    recommended_items: List[int] = []
    for items in recommendations.values():
        recommended_items.extend(items[:k])
    if not recommended_items:
        return {"coverage": 0.0, "novelty": 0.0, "mean_popularity_rank": 0.0, "cold_items_shown": 0}

    recommended_array = np.asarray(recommended_items)
    popularity_rank = pd.Series(popularity).rank(ascending=False, method="min").to_numpy()

    return {
        "coverage": round(len(set(recommended_items)) / dataset.n_items, 4),
        "novelty": round(float(novelty_per_item[recommended_array].mean()), 3),
        "mean_popularity_rank": round(float(popularity_rank[recommended_array].mean()), 1),
        "cold_items_shown": int(
            sum(1 for item in set(recommended_items) if item in dataset.cold_start_items)
        ),
    }


def cold_start_recall(
    recommendations: Dict[int, List[int]], dataset: Dataset, k: int = 20
) -> Dict[str, float]:
    """Recall restricted to items with no training interactions.

    This is the question the content-based fallback exists to answer. Users whose held-out
    items are all warm are excluded, otherwise the metric is diluted to meaninglessness.
    """
    hits, total, users_counted = 0, 0, 0
    for user_id, relevant in dataset.relevant_by_user.items():
        cold_relevant = relevant & dataset.cold_start_items
        if not cold_relevant:
            continue
        recommended = set(recommendations.get(user_id, [])[:k])
        hits += len(cold_relevant & recommended)
        total += len(cold_relevant)
        users_counted += 1

    return {
        f"cold_item_recall@{k}": round(hits / total, 5) if total else 0.0,
        "cold_item_opportunities": total,
        "users_with_cold_relevant_items": users_counted,
    }


def segment_metrics(
    recommendations: Dict[int, List[int]], dataset: Dataset, k: int = 10
) -> Dict[str, Dict[str, float]]:
    """Accuracy broken out by how much training history the user has."""
    by_segment: Dict[str, List[int]] = {}
    for user_id in dataset.eval_users:
        by_segment.setdefault(user_activity_segment(dataset, user_id), []).append(user_id)

    results = {}
    for segment, user_ids in sorted(by_segment.items()):
        scored = score_recommendations(
            recommendations, dataset.relevant_by_user, user_ids, k_values=(k,)
        )
        scored["users"] = len(user_ids)
        results[segment] = scored
    return results


def tune_hybrid_weights(dataset: Dataset, k: int = 10) -> tuple[Dict[str, float], List[dict]]:
    """Select blend weights on a validation split taken from inside the training period.

    The test set plays no part in this choice -- that is the whole point.
    """
    print("tuning hybrid weights on an inner validation split...")
    inner = build_dataset(
        dataset.train_interactions, dataset.items, dataset.users, test_fraction=0.20
    )

    components = {
        "item_item_cf": ItemItemCF().fit(inner),
        "als_mf": ALSMatrixFactorization().fit(inner),
        "content_based": ContentBasedRecommender().fit(inner),
        "popularity_decay_21d": PopularityRecommender(half_life_days=21.0).fit(inner),
    }

    trials = []
    for weights in WEIGHT_CANDIDATES:
        hybrid = HybridRecommender(components, weights).fit(inner)
        recommendations = hybrid.recommend_batch(inner.eval_users, k=k)
        scored = score_recommendations(
            recommendations, inner.relevant_by_user, inner.eval_users, k_values=(k,)
        )
        trials.append({"weights": weights, f"ndcg@{k}": scored[f"ndcg@{k}"]})
        label = " ".join(f"{name.split('_')[0]}={value:.2f}" for name, value in weights.items())
        print(f"  {label:<58} ndcg@{k}={scored[f'ndcg@{k}']:.4f}")

    best = max(trials, key=lambda trial: trial[f"ndcg@{k}"])
    print(f"  -> selected {best['weights']}  (validation ndcg@{k}={best[f'ndcg@{k}']:.4f})")
    return best["weights"], trials


def plot_precision_curve(
    per_model: Dict[str, Dict[str, float]], path: Path, metric: str = "ndcg"
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for index, (model, metrics) in enumerate(per_model.items()):
        values = [metrics.get(f"{metric}@{k}", 0.0) for k in K_VALUES]
        ax.plot(K_VALUES, values, "o-", lw=2, color=PALETTE[index % len(PALETTE)], label=model)
    ax.set_xlabel("K")
    ax.set_ylabel(f"{metric.upper()}@K")
    ax.set_title(f"{metric.upper()}@K vs. list length")
    ax.set_xticks(list(K_VALUES))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_coverage_vs_accuracy(
    accuracy: Dict[str, Dict[str, float]], catalogue: Dict[str, Dict[str, float]], path: Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 5))
    for index, model in enumerate(accuracy):
        x = catalogue[model]["coverage"]
        y = accuracy[model]["ndcg@10"]
        ax.scatter(x, y, s=140, color=PALETTE[index % len(PALETTE)], zorder=3)
        ax.annotate(model, (x, y), textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.set_xlabel("Catalogue coverage (share of items ever recommended, top-10)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Accuracy vs. catalogue coverage")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_segment_bars(
    segments: Dict[str, Dict[str, Dict[str, float]]], path: Path, k: int = 10
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(segments)
    segment_names = sorted(segments[models[0]])
    x = np.arange(len(segment_names))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(1.9 * len(segment_names) + 4, 4.8))
    for index, model in enumerate(models):
        values = [segments[model][name].get(f"ndcg@{k}", 0.0) for name in segment_names]
        ax.bar(x + index * width, values, width, label=model, color=PALETTE[index % len(PALETTE)])
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(segment_names)
    ax.set_ylabel(f"NDCG@{k}")
    ax.set_title(f"NDCG@{k} by user history length")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_als_loss(model: ALSMatrixFactorization, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(range(1, len(model.loss_history_) + 1), model.loss_history_, "o-", color=PALETTE[0], lw=2)
    ax.set_xlabel("ALS iteration")
    ax.set_ylabel("Weighted squared error + regularisation")
    ax.set_title("ALS convergence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def evaluate(
    raw_dir: Path, reports_dir: Path, tune: bool = True, models_dir: Path | None = None
) -> dict:
    dataset = load_dataset(raw_dir)
    figures = reports_dir / "figures"

    weights, weight_trials = (
        tune_hybrid_weights(dataset) if tune else (None, [])
    )

    print("\nfitting models on the full training period...")
    started = time.time()
    models: Dict[str, Recommender] = build_all(dataset, weights=weights)
    print(f"  fitted {len(models)} models in {time.time() - started:.1f}s")

    accuracy, catalogue, cold_start, segments = {}, {}, {}, {}
    for name, model in models.items():
        recommendations = model.recommend_batch(dataset.eval_users, k=MAX_K)
        accuracy[name] = score_recommendations(
            recommendations, dataset.relevant_by_user, dataset.eval_users
        )
        catalogue[name] = catalogue_metrics(recommendations, dataset)
        cold_start[name] = cold_start_recall(recommendations, dataset)
        segments[name] = segment_metrics(recommendations, dataset)

    # --- figures ----------------------------------------------------------------------
    plot_metric_bars(
        accuracy,
        ["precision@10", "recall@10", "ndcg@10", "map@10"],
        figures / "model_comparison.png",
        "Recommender comparison (K = 10, time-based split)",
    )
    plot_precision_curve(accuracy, figures / "ndcg_vs_k.png", metric="ndcg")
    plot_precision_curve(accuracy, figures / "precision_vs_k.png", metric="precision")
    plot_coverage_vs_accuracy(accuracy, catalogue, figures / "coverage_vs_accuracy.png")
    plot_segment_bars(segments, figures / "segment_performance.png")
    plot_als_loss(models["als_mf"], figures / "als_convergence.png")

    results = {
        "dataset": dataset.summary(),
        "hybrid_weights": weights or "default",
        "hybrid_weight_trials": weight_trials,
        "accuracy": accuracy,
        "catalogue": catalogue,
        "cold_start": cold_start,
        "segments": segments,
        "k_values": list(K_VALUES),
    }
    save_json(results, reports_dir / "evaluation.json")

    # --- console report ----------------------------------------------------------------
    print("\n" + "=" * 96)
    print("RANKING QUALITY (time-based split, K = 10)")
    print("=" * 96)
    header = f"{'model':<24}{'P@10':>8}{'R@10':>8}{'NDCG@10':>10}{'MAP@10':>9}{'Hit@10':>9}{'Cover':>8}{'Novelty':>9}"
    print(header)
    for name in accuracy:
        row = accuracy[name]
        print(
            f"{name:<24}{row['precision@10']:>8.4f}{row['recall@10']:>8.4f}"
            f"{row['ndcg@10']:>10.4f}{row['map@10']:>9.4f}{row['hitrate@10']:>9.4f}"
            f"{catalogue[name]['coverage']:>8.3f}{catalogue[name]['novelty']:>9.2f}"
        )

    best = max(accuracy, key=lambda name: accuracy[name]["ndcg@10"])
    baseline = accuracy["popularity"]["ndcg@10"]
    uplift = (accuracy[best]["ndcg@10"] / baseline - 1.0) * 100 if baseline else 0.0
    print(f"\nbest model: {best}  (NDCG@10 {accuracy[best]['ndcg@10']:.4f}, "
          f"{uplift:+.1f}% vs. popularity baseline)")

    print("\nCOLD-START ITEMS (no training interactions)")
    for name in cold_start:
        info = cold_start[name]
        print(f"  {name:<24}recall@20={info['cold_item_recall@20']:.4f}  "
              f"cold items surfaced in top-10: {catalogue[name]['cold_items_shown']}")

    print("\nNDCG@10 BY USER HISTORY")
    segment_names = sorted(segments[best])
    print(f"  {'model':<24}" + "".join(f"{name:>16}" for name in segment_names))
    for name in segments:
        print(
            f"  {name:<24}"
            + "".join(f"{segments[name][s]['ndcg@10']:>16.4f}" for s in segment_names)
        )

    # Persist the fitted models so the API and the Streamlit app start instantly instead
    # of refitting ALS on every launch.
    if models_dir is not None:
        save_model({"models": models, "dataset": dataset}, models_dir / "recommenders.joblib")
        save_json(
            {"best_model": best, "hybrid_weights": weights or "default", "k_values": list(K_VALUES)},
            models_dir / "model_card.json",
        )
        print(f"models  -> {models_dir / 'recommenders.joblib'}")

    print(f"\nfigures -> {figures}")
    print(f"metrics -> {reports_dir / 'evaluation.json'}")
    return results


def main() -> None:
    here = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate recommenders.")
    parser.add_argument("--raw", type=Path, default=here / "data" / "raw")
    parser.add_argument("--reports", type=Path, default=here / "reports")
    parser.add_argument("--models", type=Path, default=here / "models")
    parser.add_argument("--no-tune", action="store_true", help="skip hybrid weight tuning")
    args = parser.parse_args()
    evaluate(args.raw, args.reports, tune=not args.no_tune, models_dir=args.models)


if __name__ == "__main__":
    main()
