"""Step 1 of the approach: prepare user-item interaction data and split it by time.

Two choices here decide whether the evaluation means anything.

**The split is chronological, not random.** A random split lets the model train on a
user's March purchase and be tested on their February one. Recommenders are used to
predict forward, so the last 20% of the timeline is held out and everything before it
trains. This is also what creates the cold-start items the content model has to handle.

**Feedback is implicit and confidence-weighted.** There are no ratings, only events. A
view is weak evidence of preference and a purchase is strong, so each user-item pair gets
a confidence `1 + alpha * log1p(total_event_weight)`, following Hu, Koren & Volinsky
(2008). The binary preference is "did they interact at all"; the confidence says how much
to trust it.

Relevance for evaluation is the set of items a user interacted with *in the test period
that they had not already interacted with during training*. Re-recommending last month's
purchase is not a win, so those items are excluded from both the ground truth and the
candidate list.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DEFAULT_TEST_FRACTION = 0.20
CONFIDENCE_ALPHA = 12.0
MIN_TRAIN_INTERACTIONS = 1
MIN_TEST_RELEVANT = 1


@dataclass
class Dataset:
    """Everything the recommenders and the evaluator need, built once."""

    train_matrix: csr_matrix                      # users x items, confidence-weighted
    train_binary: csr_matrix                      # users x items, 1 where interacted
    n_users: int
    n_items: int
    items: pd.DataFrame
    users: pd.DataFrame
    train_interactions: pd.DataFrame
    test_interactions: pd.DataFrame
    cutoff: pd.Timestamp
    seen_by_user: Dict[int, Set[int]] = field(default_factory=dict)
    relevant_by_user: Dict[int, Set[int]] = field(default_factory=dict)
    eval_users: List[int] = field(default_factory=list)
    cold_start_users: List[int] = field(default_factory=list)
    cold_start_items: Set[int] = field(default_factory=set)

    def item_popularity(self) -> np.ndarray:
        """Training interaction count per item -- used by the popularity baseline."""
        return np.asarray(self.train_binary.sum(axis=0)).ravel()

    def summary(self) -> dict:
        train_counts = np.asarray(self.train_binary.sum(axis=1)).ravel()
        return {
            "cutoff": str(self.cutoff),
            "n_users": self.n_users,
            "n_items": self.n_items,
            "train_interactions": int(len(self.train_interactions)),
            "test_interactions": int(len(self.test_interactions)),
            "train_pairs": int(self.train_binary.nnz),
            "sparsity": round(1.0 - self.train_binary.nnz / (self.n_users * self.n_items), 6),
            "eval_users": len(self.eval_users),
            "cold_start_users": len(self.cold_start_users),
            "cold_start_items_in_test": len(self.cold_start_items),
            "median_train_items_per_user": float(np.median(train_counts)),
            "mean_relevant_per_eval_user": round(
                float(np.mean([len(self.relevant_by_user[u]) for u in self.eval_users])), 2
            )
            if self.eval_users
            else 0.0,
        }


def build_dataset(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    users: pd.DataFrame,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    alpha: float = CONFIDENCE_ALPHA,
) -> Dataset:
    """Chronological split plus confidence-weighted training matrix."""
    interactions = interactions.copy()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    interactions = interactions.sort_values("timestamp")

    cutoff = interactions["timestamp"].quantile(1.0 - test_fraction)
    train = interactions[interactions["timestamp"] <= cutoff]
    test = interactions[interactions["timestamp"] > cutoff]

    n_users = int(users["user_id"].max()) + 1
    n_items = int(items["item_id"].max()) + 1

    # Aggregate repeated events into one confidence value per user-item pair.
    aggregated = (
        train.groupby(["user_id", "item_id"], sort=False)["event_weight"].sum().reset_index()
    )
    confidence = 1.0 + alpha * np.log1p(aggregated["event_weight"].to_numpy())

    train_matrix = csr_matrix(
        (confidence, (aggregated["user_id"], aggregated["item_id"])),
        shape=(n_users, n_items),
    )
    train_binary = csr_matrix(
        (np.ones(len(aggregated)), (aggregated["user_id"], aggregated["item_id"])),
        shape=(n_users, n_items),
    )

    seen_by_user = (
        aggregated.groupby("user_id")["item_id"].apply(set).to_dict()
    )

    # Relevance = interacted in test, not already seen in train.
    test_by_user = test.groupby("user_id")["item_id"].apply(set).to_dict()
    relevant_by_user = {
        user_id: item_set - seen_by_user.get(user_id, set())
        for user_id, item_set in test_by_user.items()
    }
    relevant_by_user = {u: s for u, s in relevant_by_user.items() if len(s) >= MIN_TEST_RELEVANT}

    # Warm users can be scored by every model; cold users have no training history and
    # are reported separately, because only popularity and metadata can serve them.
    eval_users = sorted(
        user_id
        for user_id in relevant_by_user
        if len(seen_by_user.get(user_id, set())) >= MIN_TRAIN_INTERACTIONS
    )
    cold_start_users = sorted(
        user_id for user_id in relevant_by_user if user_id not in set(eval_users)
    )

    # Items that appear in the test period with no training interactions at all.
    trained_items = set(aggregated["item_id"].unique())
    test_items = set(test["item_id"].unique())
    cold_start_items = test_items - trained_items

    return Dataset(
        train_matrix=train_matrix,
        train_binary=train_binary,
        n_users=n_users,
        n_items=n_items,
        items=items,
        users=users,
        train_interactions=train,
        test_interactions=test,
        cutoff=cutoff,
        seen_by_user=seen_by_user,
        relevant_by_user=relevant_by_user,
        eval_users=eval_users,
        cold_start_users=cold_start_users,
        cold_start_items=cold_start_items,
    )


def load_dataset(raw_dir: Path, test_fraction: float = DEFAULT_TEST_FRACTION) -> Dataset:
    interactions = pd.read_csv(raw_dir / "interactions.csv")
    items = pd.read_csv(raw_dir / "items.csv")
    users = pd.read_csv(raw_dir / "users.csv")
    return build_dataset(interactions, items, users, test_fraction)


def user_activity_segment(dataset: Dataset, user_id: int) -> str:
    """Bucket a user by how much training history they have.

    Recommender quality varies enormously with history length, so the evaluation reports
    per-segment numbers rather than one average that hides the cold users.
    """
    n_seen = len(dataset.seen_by_user.get(user_id, set()))
    if n_seen == 0:
        return "cold (0 items)"
    if n_seen <= 5:
        return "light (1-5)"
    if n_seen <= 20:
        return "medium (6-20)"
    return "heavy (21+)"


def main() -> None:
    here = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Prepare and split interaction data.")
    parser.add_argument("--raw", type=Path, default=here / "data" / "raw")
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    args = parser.parse_args()

    dataset = load_dataset(args.raw, args.test_fraction)
    info = dataset.summary()

    print(f"chronological split at {info['cutoff']}")
    print(f"  train : {info['train_interactions']:,} interactions, {info['train_pairs']:,} unique pairs")
    print(f"  test  : {info['test_interactions']:,} interactions")
    print(f"matrix  : {info['n_users']:,} x {info['n_items']:,}, {info['sparsity']:.4%} empty")
    print(f"eval users          : {info['eval_users']:,} (warm, with held-out items)")
    print(f"cold-start users    : {info['cold_start_users']:,} (no training history)")
    print(f"cold-start items    : {info['cold_start_items_in_test']} appear in test with no training data")
    print(f"median train items  : {info['median_train_items_per_user']:.0f} per user")
    print(f"mean relevant items : {info['mean_relevant_per_eval_user']:.2f} per eval user")

    segments = pd.Series(
        [user_activity_segment(dataset, u) for u in dataset.eval_users]
    ).value_counts()
    print(f"\neval users by history: {segments.to_dict()}")


if __name__ == "__main__":
    main()
