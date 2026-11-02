"""Metrics shared by the classification projects (1 and 2) and the ranking project (3).

Classification helpers report threshold-free metrics (ROC-AUC, PR-AUC, Brier) alongside
metrics at an operating threshold, because a churn or failure model is only useful once
somebody has chosen the cut-off at which they will act.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------


@dataclass
class ClassificationReport:
    """Metrics for one model at one operating threshold."""

    model: str
    threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    brier: float
    base_rate: float
    confusion: Dict[str, int] = field(default_factory=dict)
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        payload = {
            "model": self.model,
            "threshold": round(self.threshold, 4),
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "brier": round(self.brier, 4),
            "base_rate": round(self.base_rate, 4),
            "confusion": self.confusion,
        }
        payload.update({key: round(float(val), 4) for key, val in self.extra.items()})
        return payload


def classification_report(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    model: str,
    threshold: float = 0.5,
    extra: Dict[str, float] | None = None,
) -> ClassificationReport:
    """Score probabilistic predictions, both threshold-free and at `threshold`."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return ClassificationReport(
        model=model,
        threshold=float(threshold),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        brier=float(brier_score_loss(y_true, y_score)),
        base_rate=float(y_true.mean()),
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        extra=extra or {},
    )


def best_f1_threshold(
    y_true: Sequence[int], y_score: Sequence[float], grid: int = 200
) -> float:
    """Threshold that maximises F1 on the given scores.

    Used to pick a default operating point when no business cost is specified.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    candidates = np.linspace(0.01, 0.99, grid)
    scores = [f1_score(y_true, (y_score >= t).astype(int), zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def threshold_for_recall(
    y_true: Sequence[int], y_score: Sequence[float], target_recall: float
) -> float:
    """Highest threshold that still achieves at least `target_recall`.

    Preferred for predictive maintenance, where missing a failure costs far more than
    inspecting a healthy machine, so recall is fixed first and precision is reported.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    for threshold in np.linspace(0.99, 0.01, 199):
        if recall_score(y_true, (y_score >= threshold).astype(int), zero_division=0) >= target_recall:
            return float(threshold)
    return 0.01


def lift_table(
    y_true: Sequence[int], y_score: Sequence[float], n_bins: int = 10
) -> List[Dict[str, float]]:
    """Decile lift table: how much of the positive class each score decile captures.

    This is the view a retention or maintenance team actually works from — "if we only
    call the top 10%, what fraction of churners do we reach?"
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    y_true, y_score = y_true[order], y_score[order]

    total_positives = max(int(y_true.sum()), 1)
    base_rate = y_true.mean()
    bins = np.array_split(np.arange(len(y_true)), n_bins)

    rows: List[Dict[str, float]] = []
    captured = 0
    for i, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue
        bin_positives = int(y_true[idx].sum())
        captured += bin_positives
        bin_rate = bin_positives / len(idx)
        rows.append(
            {
                "decile": i,
                "customers": len(idx),
                "positives": bin_positives,
                "rate": round(float(bin_rate), 4),
                "lift": round(float(bin_rate / base_rate) if base_rate else 0.0, 3),
                "cumulative_capture": round(captured / total_positives, 4),
                "min_score": round(float(y_score[idx].min()), 4),
                "max_score": round(float(y_score[idx].max()), 4),
            }
        )
    return rows


def expected_calibration_error(
    y_true: Sequence[int], y_score: Sequence[float], n_bins: int = 10
) -> float:
    """Mean absolute gap between predicted probability and observed frequency."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error, total = 0.0, len(y_true)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (y_score >= low) & (y_score < high if high < 1.0 else y_score <= high)
        if not mask.any():
            continue
        error += mask.sum() * abs(y_score[mask].mean() - y_true[mask].mean())
    return float(error / total) if total else 0.0


# --------------------------------------------------------------------------------------
# Ranking (project 3)
# --------------------------------------------------------------------------------------


def precision_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    relevant = set(relevant)
    if k == 0:
        return 0.0
    top_k = list(recommended)[:k]
    return sum(1 for item in top_k if item in relevant) / k


def recall_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top_k = list(recommended)[:k]
    return sum(1 for item in top_k if item in relevant) / len(relevant)


def ndcg_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Binary-relevance NDCG@K with the standard log2 discount."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    gains = [1.0 if item in relevant else 0.0 for item in list(recommended)[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(dcg / ideal) if ideal else 0.0


def average_precision_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, item in enumerate(list(recommended)[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return float(score / min(len(relevant), k))


def hit_rate_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    relevant = set(relevant)
    return float(any(item in relevant for item in list(recommended)[:k]))
