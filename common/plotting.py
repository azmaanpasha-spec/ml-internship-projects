"""Figure helpers. Every function saves a PNG and returns its path.

Matplotlib runs on the non-interactive Agg backend so the pipelines work headless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PALETTE = ["#2b6cb0", "#dd6b20", "#2f855a", "#805ad5", "#c53030", "#2c7a7b"]


def _finish(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(scores: Dict[str, tuple], path: Path, title: str = "ROC curves") -> Path:
    """`scores` maps a model name to a (y_true, y_score) pair."""
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for i, (name, (y_true, y_score)) in enumerate(scores.items()):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    return _finish(fig, path)


def plot_pr_curves(
    scores: Dict[str, tuple], path: Path, title: str = "Precision-recall curves"
) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 5))
    base_rate = None
    for i, (name, (y_true, y_score)) in enumerate(scores.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color=PALETTE[i % len(PALETTE)], lw=2, label=f"{name} (AP={ap:.3f})")
        base_rate = float(np.asarray(y_true).mean())
    if base_rate is not None:
        ax.axhline(base_rate, ls="--", color="black", lw=1, alpha=0.5, label=f"base rate={base_rate:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    return _finish(fig, path)


def plot_calibration(
    scores: Dict[str, tuple], path: Path, n_bins: int = 10, title: str = "Calibration"
) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for i, (name, (y_true, y_score)) in enumerate(scores.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
        ax.plot(prob_pred, prob_true, "o-", color=PALETTE[i % len(PALETTE)], lw=2, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    return _finish(fig, path)


def plot_confusion(
    y_true: Sequence[int], y_pred: Sequence[int], path: Path, title: str = "Confusion matrix"
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(title)
    return _finish(fig, path)


def plot_feature_importance(
    names: Sequence[str], values: Sequence[float], path: Path, top_n: int = 20, title: str = "Feature importance"
) -> Path:
    order = np.argsort(values)[::-1][:top_n][::-1]
    fig, ax = plt.subplots(figsize=(7, max(4.0, 0.32 * len(order))))
    ax.barh([names[i] for i in order], [values[i] for i in order], color=PALETTE[0])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    return _finish(fig, path)


def plot_lift(rows: Sequence[Dict[str, float]], path: Path, title: str = "Decile lift") -> Path:
    deciles = [row["decile"] for row in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(deciles, [row["lift"] for row in rows], color=PALETTE[0])
    ax1.axhline(1.0, ls="--", color="black", lw=1, alpha=0.6)
    ax1.set_xlabel("Score decile (1 = highest risk)")
    ax1.set_ylabel("Lift vs. base rate")
    ax1.set_title("Lift by decile")
    ax1.grid(axis="y", alpha=0.25)

    ax2.plot(deciles, [row["cumulative_capture"] for row in rows], "o-", color=PALETTE[2], lw=2)
    ax2.plot(deciles, [d / len(deciles) for d in deciles], "k--", lw=1, alpha=0.5, label="random")
    ax2.set_xlabel("Deciles targeted")
    ax2.set_ylabel("Share of positives captured")
    ax2.set_title("Cumulative gains")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)
    fig.suptitle(title)
    return _finish(fig, path)


def plot_metric_bars(
    results: Dict[str, Dict[str, float]], metrics: Sequence[str], path: Path, title: str
) -> Path:
    """Grouped bar chart comparing several models across several metrics."""
    models = list(results)
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(1.9 * len(metrics) + 3, 4.6))
    for i, model in enumerate(models):
        values = [results[model].get(metric, 0.0) for metric in metrics]
        bars = ax.bar(x + i * width, values, width, label=model, color=PALETTE[i % len(PALETTE)])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    return _finish(fig, path)
