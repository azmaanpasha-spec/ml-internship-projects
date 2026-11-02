"""Unit tests for the shared metric implementations.

These are hand-worked examples rather than property tests: a ranking metric that is
subtly wrong still produces plausible-looking numbers, so the only way to trust NDCG is
to check it against a value computed by hand.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

from common.evaluation import (
    average_precision_at_k,
    best_f1_threshold,
    classification_report,
    expected_calibration_error,
    hit_rate_at_k,
    lift_table,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    threshold_for_recall,
)


# --------------------------------------------------------------------------------------
# Ranking metrics
# --------------------------------------------------------------------------------------


def test_precision_at_k_counts_only_the_top_k():
    recommended = [1, 2, 3, 4, 5]
    relevant = {1, 3, 9}
    assert precision_at_k(recommended, relevant, 5) == pytest.approx(2 / 5)
    # Item 3 sits at position 3, so it falls outside the top 2.
    assert precision_at_k(recommended, relevant, 2) == pytest.approx(1 / 2)


def test_recall_at_k_divides_by_the_relevant_set():
    assert recall_at_k([1, 2, 3], {1, 3, 9}, 3) == pytest.approx(2 / 3)
    assert recall_at_k([7, 8], {1}, 2) == 0.0


def test_recall_and_precision_handle_empty_relevant_set():
    assert recall_at_k([1, 2], set(), 2) == 0.0
    assert ndcg_at_k([1, 2], set(), 2) == 0.0
    assert average_precision_at_k([1, 2], set(), 2) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)


def test_ndcg_matches_a_hand_computed_value():
    # Relevant items at positions 1 and 3 (1-indexed) out of 3 recommendations.
    # DCG   = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0 + 0.5 = 1.5
    # IDCG  = 1/log2(2) + 1/log2(3)             = 1.0 + 0.63093 = 1.63093
    expected = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k([10, 20, 30], {10, 30}, 3) == pytest.approx(expected)


def test_ndcg_rewards_putting_the_hit_higher():
    high = ndcg_at_k([1, 9, 9], {1}, 3)
    low = ndcg_at_k([9, 9, 1], {1}, 3)
    assert high > low


def test_average_precision_matches_a_hand_computed_value():
    # Hits at positions 1 and 3: AP = (1/1 + 2/3) / min(2, 3) = 1.66667 / 2
    assert average_precision_at_k([1, 5, 2], {1, 2}, 3) == pytest.approx((1 + 2 / 3) / 2)


def test_hit_rate_is_binary():
    assert hit_rate_at_k([4, 5, 6], {6}, 3) == 1.0
    assert hit_rate_at_k([4, 5, 6], {7}, 3) == 0.0


# --------------------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------------------


def test_classification_report_confusion_matrix_is_correct():
    y_true = [1, 1, 0, 0, 1, 0]
    y_score = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    report = classification_report(y_true, y_score, model="test", threshold=0.5)

    # Predicted positive: 0.9, 0.8, 0.7 -> two true positives, one false positive.
    assert report.confusion == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert report.precision == pytest.approx(2 / 3)
    assert report.recall == pytest.approx(2 / 3)
    assert report.base_rate == pytest.approx(0.5)


def test_perfect_separation_gives_auc_one():
    report = classification_report(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], model="perfect", threshold=0.5
    )
    assert report.roc_auc == pytest.approx(1.0)
    assert report.f1 == pytest.approx(1.0)


def test_threshold_for_recall_achieves_the_target():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.2, 500)
    # A score that is informative but imperfect.
    y_score = np.clip(y_true * 0.5 + rng.normal(0.25, 0.2, 500), 0, 1)

    for target in (0.7, 0.8, 0.9):
        threshold = threshold_for_recall(y_true, y_score, target)
        achieved = ((y_score >= threshold) & (y_true == 1)).sum() / y_true.sum()
        assert achieved >= target - 1e-9


def test_best_f1_threshold_beats_the_default_on_an_imbalanced_problem():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(1)
    y_true = rng.binomial(1, 0.08, 2000)
    y_score = np.clip(y_true * 0.35 + rng.normal(0.15, 0.18, 2000), 0, 1)

    tuned = best_f1_threshold(y_true, y_score)
    tuned_f1 = f1_score(y_true, (y_score >= tuned).astype(int), zero_division=0)
    default_f1 = f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)
    assert tuned_f1 >= default_f1


def test_expected_calibration_error_is_zero_for_a_calibrated_model():
    # 1000 samples at p=0.3 where exactly 30% are positive, and likewise at p=0.8.
    y_score = np.concatenate([np.full(1000, 0.3), np.full(1000, 0.8)])
    y_true = np.concatenate(
        [np.r_[np.ones(300), np.zeros(700)], np.r_[np.ones(800), np.zeros(200)]]
    )
    assert expected_calibration_error(y_true, y_score) == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_detects_overconfidence():
    y_score = np.full(1000, 0.95)
    y_true = np.r_[np.ones(500), np.zeros(500)]      # claims 95%, delivers 50%
    assert expected_calibration_error(y_true, y_score) == pytest.approx(0.45, abs=1e-9)


# --------------------------------------------------------------------------------------
# Lift
# --------------------------------------------------------------------------------------


def test_lift_table_captures_all_positives_and_is_monotone_in_capture():
    rng = np.random.default_rng(2)
    y_true = rng.binomial(1, 0.25, 1000)
    y_score = np.clip(y_true * 0.4 + rng.normal(0.3, 0.2, 1000), 0, 1)

    rows = lift_table(y_true, y_score, n_bins=10)
    assert len(rows) == 10
    # Cumulative capture must reach 1.0 by the last decile and never decrease.
    captures = [row["cumulative_capture"] for row in rows]
    assert captures == sorted(captures)
    assert captures[-1] == pytest.approx(1.0)
    # An informative model puts more positives in the first decile than the last.
    assert rows[0]["rate"] > rows[-1]["rate"]
    assert rows[0]["lift"] > 1.0


def test_lift_table_of_a_random_score_has_lift_near_one():
    rng = np.random.default_rng(3)
    y_true = rng.binomial(1, 0.3, 4000)
    y_score = rng.random(4000)                        # no information at all

    rows = lift_table(y_true, y_score, n_bins=5)
    for row in rows:
        assert row["lift"] == pytest.approx(1.0, abs=0.25)
