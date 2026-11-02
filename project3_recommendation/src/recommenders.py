"""Steps 2-4 of the approach: a popularity baseline, collaborative filtering, matrix
factorisation, a content-based model for new products, and a hybrid of all of them.

The models share one interface so the evaluator can treat them interchangeably:

    fit(dataset)                         -> self
    score_users(user_ids)                -> (n_users, n_items) score array
    recommend_batch(user_ids, k)         -> {user_id: [item_id, ...]}

Scoring is **batched** on purpose. Item-item CF needs a sparse-by-dense product against a
1500x1500 similarity matrix; doing that one user at a time is orders of magnitude slower
than doing it for a thousand users at once, and the evaluation scores several thousand
users against five models.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.data_prep import Dataset

NEG_INF = -1e18


class Recommender:
    """Common machinery: batching, masking already-seen items, top-k extraction."""

    name = "base"

    def fit(self, dataset: Dataset) -> "Recommender":  # pragma: no cover - interface
        raise NotImplementedError

    def score_users(self, user_ids: Sequence[int]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _mask_seen(self, scores: np.ndarray, user_ids: Sequence[int]) -> np.ndarray:
        """Remove items the user already interacted with during training."""
        for row, user_id in enumerate(user_ids):
            seen = self.dataset.seen_by_user.get(user_id)
            if seen:
                scores[row, list(seen)] = NEG_INF
        return scores

    def recommend_batch(
        self,
        user_ids: Sequence[int],
        k: int = 10,
        exclude_seen: bool = True,
        chunk_size: int = 1000,
    ) -> Dict[int, List[int]]:
        """Top-k item ids per user, in rank order."""
        recommendations: Dict[int, List[int]] = {}
        user_ids = list(user_ids)

        for start in range(0, len(user_ids), chunk_size):
            chunk = user_ids[start : start + chunk_size]
            scores = np.asarray(self.score_users(chunk), dtype=float).copy()
            if exclude_seen:
                scores = self._mask_seen(scores, chunk)

            # argpartition finds the top k without sorting all 1500 items, then only
            # those k are sorted.
            top_k = min(k, scores.shape[1])
            partitioned = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
            for row, user_id in enumerate(chunk):
                candidates = partitioned[row]
                ordered = candidates[np.argsort(-scores[row, candidates])]
                recommendations[user_id] = [
                    int(item) for item in ordered if scores[row, item] > NEG_INF
                ]
        return recommendations

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> List[int]:
        return self.recommend_batch([user_id], k=k, exclude_seen=exclude_seen)[user_id]


class PopularityRecommender(Recommender):
    """Step 2: the baseline every recommender must beat.

    With a power-law catalogue this is a strong baseline, not a straw man. An optional
    exponential time decay makes it stronger still -- what was popular last week beats
    what was popular six months ago.
    """

    def __init__(self, half_life_days: float | None = 21.0):
        self.half_life_days = half_life_days
        self.name = (
            "popularity" if half_life_days is None else f"popularity_decay_{half_life_days:g}d"
        )

    def fit(self, dataset: Dataset) -> "PopularityRecommender":
        self.dataset = dataset
        interactions = dataset.train_interactions

        if self.half_life_days is None:
            weights = interactions["event_weight"].to_numpy()
        else:
            age_days = (
                (dataset.cutoff - interactions["timestamp"]).dt.total_seconds().to_numpy() / 86400.0
            )
            decay = 0.5 ** (age_days / self.half_life_days)
            weights = interactions["event_weight"].to_numpy() * decay

        self.scores_ = np.zeros(dataset.n_items)
        np.add.at(self.scores_, interactions["item_id"].to_numpy(), weights)
        return self

    def score_users(self, user_ids: Sequence[int]) -> np.ndarray:
        return np.tile(self.scores_, (len(user_ids), 1))


class ItemItemCF(Recommender):
    """Step 3a: item-item collaborative filtering on cosine similarity.

    Similarity is computed between L2-normalised item columns of the confidence matrix,
    then truncated to each item's `top_n` neighbours. Truncation is not an optimisation
    detail -- keeping all 1500 neighbours lets a long tail of weak, noisy similarities
    dominate the sum and measurably degrades precision.
    """

    name = "item_item_cf"

    def __init__(self, top_n: int = 120, shrinkage: float = 10.0):
        self.top_n = top_n
        self.shrinkage = shrinkage

    def fit(self, dataset: Dataset) -> "ItemItemCF":
        self.dataset = dataset
        matrix = dataset.train_matrix

        # Cosine similarity between items, with shrinkage towards zero for item pairs
        # supported by few co-occurrences.
        item_norms = normalize(matrix.tocsc(), norm="l2", axis=0)
        similarity = (item_norms.T @ item_norms).toarray()

        co_counts = (dataset.train_binary.T @ dataset.train_binary).toarray()
        similarity *= co_counts / (co_counts + self.shrinkage)
        np.fill_diagonal(similarity, 0.0)

        # Keep only the strongest top_n neighbours per item.
        if self.top_n < similarity.shape[1]:
            cutoff_index = similarity.shape[1] - self.top_n
            thresholds = np.partition(similarity, cutoff_index, axis=1)[:, cutoff_index]
            similarity[similarity < thresholds[:, None]] = 0.0

        self.similarity_ = similarity
        return self

    def score_users(self, user_ids: Sequence[int]) -> np.ndarray:
        user_rows = self.dataset.train_matrix[list(user_ids)]
        return np.asarray(user_rows @ self.similarity_)


class ALSMatrixFactorization(Recommender):
    """Step 3b: implicit-feedback matrix factorisation, trained with alternating least
    squares (Hu, Koren & Volinsky 2008).

    Implemented directly rather than with a library so the mechanics are explicit. The
    objective is

        min  sum_{u,i} c_ui (p_ui - x_u . y_i)^2 + reg (||X||^2 + ||Y||^2)

    where `p_ui` is 1 if the user interacted with the item and `c_ui` is the confidence
    from `data_prep`. The key identity that makes this tractable is

        Y^T C_u Y = Y^T Y + Y_u^T (C_u - I) Y_u

    so the full `n_items x n_items` weighted Gram matrix is never formed: `Y^T Y` is
    computed once per iteration and only the user's own items contribute the correction.
    """

    name = "als_mf"

    def __init__(
        self,
        n_factors: int = 40,
        regularisation: float = 0.08,
        n_iterations: int = 15,
        seed: int = 13,
    ):
        self.n_factors = n_factors
        self.regularisation = regularisation
        self.n_iterations = n_iterations
        self.seed = seed
        self.loss_history_: List[float] = []

    def _solve_side(
        self, matrix: csr_matrix, fixed_factors: np.ndarray, n_entities: int
    ) -> np.ndarray:
        """One ALS half-step: solve for every row of the free side."""
        n_factors = self.n_factors
        gram = fixed_factors.T @ fixed_factors
        regularised_identity = self.regularisation * np.eye(n_factors)
        solved = np.zeros((n_entities, n_factors))

        indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
        for entity in range(n_entities):
            start, end = indptr[entity], indptr[entity + 1]
            if start == end:
                continue                               # no interactions: leave at zero

            item_indices = indices[start:end]
            confidence = data[start:end]
            factors_subset = fixed_factors[item_indices]

            # A = Y^T Y + Y_u^T (C_u - I) Y_u + reg I
            weighted = factors_subset * (confidence - 1.0)[:, None]
            A = gram + factors_subset.T @ weighted + regularised_identity
            # b = Y_u^T C_u p_u, and p_u = 1 on the observed entries
            b = factors_subset.T @ confidence

            solved[entity] = np.linalg.solve(A, b)
        return solved

    def _loss(self, matrix: csr_matrix) -> float:
        """Weighted squared error on the observed entries plus the regulariser.

        Only observed pairs are summed: the unobserved term is constant with respect to
        the stopping decision and summing 9M zeros every iteration is wasteful.
        """
        coo = matrix.tocoo()
        predicted = np.einsum(
            "ij,ij->i", self.user_factors_[coo.row], self.item_factors_[coo.col]
        )
        error = float((coo.data * (1.0 - predicted) ** 2).sum())
        penalty = self.regularisation * (
            float((self.user_factors_**2).sum()) + float((self.item_factors_**2).sum())
        )
        return error + penalty

    def fit(self, dataset: Dataset) -> "ALSMatrixFactorization":
        self.dataset = dataset
        matrix = dataset.train_matrix.tocsr()
        matrix_t = matrix.T.tocsr()

        rng = np.random.default_rng(self.seed)
        self.user_factors_ = rng.normal(0, 0.01, (dataset.n_users, self.n_factors))
        self.item_factors_ = rng.normal(0, 0.01, (dataset.n_items, self.n_factors))

        for _ in range(self.n_iterations):
            self.user_factors_ = self._solve_side(matrix, self.item_factors_, dataset.n_users)
            self.item_factors_ = self._solve_side(matrix_t, self.user_factors_, dataset.n_items)
            self.loss_history_.append(self._loss(matrix))

        return self

    def score_users(self, user_ids: Sequence[int]) -> np.ndarray:
        return self.user_factors_[list(user_ids)] @ self.item_factors_.T

    def similar_items(self, item_id: int, k: int = 10) -> List[tuple[int, float]]:
        """Nearest neighbours in latent space -- powers the API's /similar endpoint."""
        factors = normalize(self.item_factors_)
        similarity = factors @ factors[item_id]
        similarity[item_id] = -np.inf
        top = np.argsort(-similarity)[:k]
        return [(int(i), float(similarity[i])) for i in top]


class ContentBasedRecommender(Recommender):
    """Step 4: content-based model over item metadata, the cold-start fallback.

    Items are represented as TF-IDF vectors over their category, subcategory, brand, tags
    and discretised price/rating bands. A user profile is the confidence-weighted mean of
    the items they interacted with.

    This is the only model here that can recommend a product released after the training
    cut-off: it needs the item's description, not its interaction history.
    """

    name = "content_based"

    def __init__(self, max_features: int = 4000):
        self.max_features = max_features

    @staticmethod
    def _item_text(items: pd.DataFrame) -> pd.Series:
        """Bag of metadata tokens per item.

        Price and rating are bucketed into bands rather than used as numbers, so that
        "mid-priced" becomes a shared token two different items can match on.
        """
        price_band = pd.cut(
            items["price"],
            bins=[0, 25, 75, 200, 500, np.inf],
            labels=["budget", "lowmid", "mid", "premium", "luxury"],
        ).astype(str)
        rating_band = pd.cut(
            items["avg_rating"],
            bins=[0, 3.5, 4.0, 4.5, 5.0],
            labels=["rated_low", "rated_ok", "rated_good", "rated_top"],
        ).astype(str)

        return (
            items["category"].str.replace(" & ", "_", regex=False).str.replace(" ", "_", regex=False)
            + " "
            + items["subcategory"].str.replace(" ", "_", regex=False)
            + " "
            + items["brand"]
            + " "
            + items["tags"].fillna("")
            + " "
            + price_band
            + " "
            + rating_band
        )

    def fit(self, dataset: Dataset) -> "ContentBasedRecommender":
        self.dataset = dataset
        items = dataset.items.sort_values("item_id").reset_index(drop=True)

        self.vectoriser_ = TfidfVectorizer(
            max_features=self.max_features, token_pattern=r"[^\s]+", sublinear_tf=True
        )
        item_vectors = self.vectoriser_.fit_transform(self._item_text(items))
        self.item_vectors_ = normalize(item_vectors)

        # User profile = confidence-weighted average of the items they engaged with.
        profiles = dataset.train_matrix @ self.item_vectors_
        self.user_profiles_ = normalize(profiles)
        return self

    def score_users(self, user_ids: Sequence[int]) -> np.ndarray:
        profiles = self.user_profiles_[list(user_ids)]
        return np.asarray((profiles @ self.item_vectors_.T).todense())

    def similar_items(self, item_id: int, k: int = 10) -> List[tuple[int, float]]:
        similarity = np.asarray(
            (self.item_vectors_[item_id] @ self.item_vectors_.T).todense()
        ).ravel()
        similarity[item_id] = -np.inf
        top = np.argsort(-similarity)[:k]
        return [(int(i), float(similarity[i])) for i in top]


class HybridRecommender(Recommender):
    """Step 4 (continued): rank-level blend of the component models.

    Component scores live on incomparable scales -- ALS dot products, cosine
    similarities, decayed event counts. Blending the raw values would let whichever model
    happens to have the widest numeric range dominate. So each model's scores are
    converted to **percentile ranks per user** and the ranks are blended. It is scale-free
    and robust to outliers.

    Cold items are handled explicitly: an item with fewer than `cold_item_threshold`
    training interactions is scored by the content model alone, because the CF and MF
    scores for it are not low-confidence, they are meaningless.
    """

    name = "hybrid"

    def __init__(
        self,
        components: Dict[str, Recommender],
        weights: Dict[str, float],
        cold_item_threshold: int = 3,
    ):
        self.components = components
        self.weights = weights
        self.cold_item_threshold = cold_item_threshold

    def fit(self, dataset: Dataset) -> "HybridRecommender":
        """Components are expected to be fitted already; this only wires up the blend."""
        self.dataset = dataset
        popularity = dataset.item_popularity()
        self.cold_items_ = popularity < self.cold_item_threshold
        if "content_based" not in self.components:
            raise ValueError("hybrid requires a content_based component for cold items")
        return self

    @staticmethod
    def _to_percentile_ranks(scores: np.ndarray) -> np.ndarray:
        """Row-wise percentile ranks in [0, 1], with **tied scores sharing a rank**.

        Tie handling is not a detail here, it is the whole correctness of the blend.
        Item-item CF leaves most items at exactly 0.0 because the similarity matrix is
        truncated to each item's top neighbours. Ranking those ties by array position
        (as `argsort(argsort(x))` does) hands an arbitrary zero-score item a percentile
        as high as `n_zeros / n_items` -- around 0.87 on this catalogue -- so pure noise
        outranks genuine recommendations and the blend performs like its weakest member.
        Averaging tied ranks collapses that whole block to one middling value, which is
        what "no information about this item" should look like.
        """
        ranks = rankdata(scores, method="average", axis=1)
        return (ranks - 1.0) / max(scores.shape[1] - 1, 1)

    def score_users(self, user_ids: Sequence[int]) -> np.ndarray:
        blended = np.zeros((len(user_ids), self.dataset.n_items))
        total_weight = sum(self.weights.get(name, 0.0) for name in self.components)

        for name, model in self.components.items():
            weight = self.weights.get(name, 0.0)
            if weight == 0.0:
                continue
            blended += weight * self._to_percentile_ranks(
                np.asarray(model.score_users(user_ids), dtype=float)
            )
        if total_weight > 0:
            blended /= total_weight

        # For cold items, substitute the content-only rank: the collaborative signals
        # carry no information about an item nobody has interacted with yet.
        content_ranks = self._to_percentile_ranks(
            np.asarray(self.components["content_based"].score_users(user_ids), dtype=float)
        )
        blended[:, self.cold_items_] = content_ranks[:, self.cold_items_]
        return blended


def build_all(dataset: Dataset, weights: Dict[str, float] | None = None) -> Dict[str, Recommender]:
    """Fit every model and return them keyed by name, hybrid last."""
    models: Dict[str, Recommender] = {}

    for model in (
        PopularityRecommender(half_life_days=None),
        PopularityRecommender(half_life_days=21.0),
        ItemItemCF(),
        ALSMatrixFactorization(),
        ContentBasedRecommender(),
    ):
        models[model.name] = model.fit(dataset)

    components = {
        "item_item_cf": models["item_item_cf"],
        "als_mf": models["als_mf"],
        "content_based": models["content_based"],
        "popularity_decay_21d": models["popularity_decay_21d"],
    }
    default_weights = weights or {
        "als_mf": 0.40,
        "item_item_cf": 0.35,
        "content_based": 0.15,
        "popularity_decay_21d": 0.10,
    }
    models["hybrid"] = HybridRecommender(components, default_weights).fit(dataset)
    return models
