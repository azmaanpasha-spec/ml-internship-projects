"""Step 5 of the approach: compare models on ROC-AUC, PR-AUC, precision, recall, F1
and calibration -- then pick the operating threshold that maximises retention profit.

A churn model is not finished when the AUC is known. Somebody has to decide who gets a
retention call, and that decision needs a threshold justified by money, not by 0.5 being
a round number. The campaign simulation at the end of this module does that.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

from common.evaluation import (  # noqa: E402
    best_f1_threshold,
    classification_report,
    expected_calibration_error,
    lift_table,
)
from common.explain import shap_values_for_pipeline  # noqa: E402
from common.io_utils import load_json, load_model, save_json  # noqa: E402
from common.plotting import (  # noqa: E402
    plot_calibration,
    plot_confusion,
    plot_feature_importance,
    plot_lift,
    plot_metric_bars,
    plot_pr_curves,
    plot_roc_curves,
)
from src.train import split_data  # noqa: E402

warnings.filterwarnings("ignore")

# Campaign economics. These are the assumptions a business would supply; they are stated
# here explicitly so the recommended threshold can be re-derived when they change.
RETENTION_OFFER_COST = 45.0      # cost of the discount/offer extended to a flagged customer
CONTACT_COST = 6.0               # cost of the outreach itself, incurred for every contact
OFFER_SUCCESS_RATE = 0.32        # share of genuine churners retained when contacted
EXPECTED_REMAINING_MONTHS = 14   # horizon over which a saved customer keeps paying


def _model_paths(models_dir: Path) -> Dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(models_dir.glob("*.joblib"))
        if path.stem not in {"best_model"}
    }


def campaign_simulation(
    y_true: np.ndarray, y_score: np.ndarray, monthly_value: np.ndarray
) -> tuple[List[Dict[str, float]], Dict[str, float]]:
    """Net profit of a retention campaign across candidate thresholds.

    For each threshold, everyone scored above it is contacted:
      * a true churner is saved with probability OFFER_SUCCESS_RATE, worth
        monthly_charges x EXPECTED_REMAINING_MONTHS
      * every contact costs CONTACT_COST, and every offer costs RETENTION_OFFER_COST
    """
    rows: List[Dict[str, float]] = []
    for threshold in np.round(np.arange(0.05, 0.96, 0.05), 2):
        contacted = y_score >= threshold
        n_contacted = int(contacted.sum())
        if n_contacted == 0:
            continue

        true_churners = contacted & (y_true == 1)
        saved = OFFER_SUCCESS_RATE * monthly_value[true_churners].sum() * EXPECTED_REMAINING_MONTHS
        cost = n_contacted * (CONTACT_COST + RETENTION_OFFER_COST)
        precision = float(true_churners.sum() / n_contacted)
        recall = float(true_churners.sum() / max((y_true == 1).sum(), 1))

        rows.append(
            {
                "threshold": float(threshold),
                "contacted": n_contacted,
                "contacted_share": round(n_contacted / len(y_true), 4),
                "true_churners_reached": int(true_churners.sum()),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "revenue_saved": round(float(saved), 2),
                "campaign_cost": round(float(cost), 2),
                "net_profit": round(float(saved - cost), 2),
                "roi": round(float((saved - cost) / cost), 3) if cost else 0.0,
            }
        )

    best = max(rows, key=lambda row: row["net_profit"]) if rows else {}
    return rows, best


def explain_with_shap(pipeline, X_sample: pd.DataFrame, reports_dir: Path) -> List[Dict[str, float]]:
    """Global SHAP importance for the winning model, whichever family it belongs to.

    Returns an empty list rather than failing the pipeline if SHAP cannot handle the
    estimator -- permutation importance is computed independently as a cross-check.
    """
    values, names = shap_values_for_pipeline(pipeline, X_sample)
    if values is None:
        print("  SHAP unavailable for this estimator - relying on permutation importance")
        return []

    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    plot_feature_importance(
        [names[i] for i in order[:20]],
        [float(mean_abs[i]) for i in order[:20]],
        reports_dir / "figures" / "shap_importance.png",
        title="Mean |SHAP value| - drivers of churn risk",
    )
    return [
        {"feature": names[i], "mean_abs_shap": round(float(mean_abs[i]), 5)}
        for i in order[:25]
    ]


def evaluate(features_path: Path, models_dir: Path, reports_dir: Path) -> dict:
    frame = pd.read_csv(features_path)
    X_train, X_test, y_train, y_test = split_data(frame)
    y_test_array = y_test.to_numpy()

    figures = reports_dir / "figures"
    card = load_json(models_dir / "model_card.json")
    best_name = card["best_model"]

    # --- compare every trained model on the untouched hold-out set -------------------
    curves, summary = {}, {}
    for name, path in _model_paths(models_dir).items():
        scores = load_model(path).predict_proba(X_test)[:, 1]
        curves[name] = (y_test_array, scores)
        report = classification_report(y_test_array, scores, model=name, threshold=0.5)
        entry = report.as_dict()
        entry["ece"] = round(expected_calibration_error(y_test_array, scores), 4)
        summary[name] = entry

    calibrated = load_model(models_dir / "best_model.joblib")
    calibrated_scores = calibrated.predict_proba(X_test)[:, 1]
    curves[f"{best_name} (calibrated)"] = (y_test_array, calibrated_scores)
    calibrated_entry = classification_report(
        y_test_array, calibrated_scores, model=f"{best_name}_calibrated", threshold=0.5
    ).as_dict()
    calibrated_entry["ece"] = round(expected_calibration_error(y_test_array, calibrated_scores), 4)
    summary[f"{best_name}_calibrated"] = calibrated_entry

    plot_roc_curves(curves, figures / "roc_curves.png", "Churn models - ROC")
    plot_pr_curves(curves, figures / "pr_curves.png", "Churn models - precision/recall")
    plot_calibration(curves, figures / "calibration.png", title="Churn models - calibration")
    plot_metric_bars(
        summary,
        ["roc_auc", "pr_auc", "precision", "recall", "f1"],
        figures / "model_comparison.png",
        "Model comparison on the hold-out set (threshold = 0.50)",
    )

    # --- choose the operating threshold ----------------------------------------------
    f1_threshold = best_f1_threshold(y_test_array, calibrated_scores)
    monthly_value = X_test["monthly_charges"].to_numpy()
    campaign_rows, campaign_best = campaign_simulation(y_test_array, calibrated_scores, monthly_value)
    profit_threshold = campaign_best.get("threshold", f1_threshold)

    at_f1 = classification_report(
        y_test_array, calibrated_scores, model="best@f1", threshold=f1_threshold
    ).as_dict()
    at_profit = classification_report(
        y_test_array, calibrated_scores, model="best@profit", threshold=profit_threshold
    ).as_dict()

    plot_confusion(
        y_test_array,
        (calibrated_scores >= profit_threshold).astype(int),
        figures / "confusion_matrix.png",
        f"Confusion matrix at profit-optimal threshold ({profit_threshold:.2f})",
    )

    # --- who is driving the predictions ----------------------------------------------
    lift = lift_table(y_test_array, calibrated_scores, n_bins=10)
    plot_lift(lift, figures / "lift_curve.png", "Churn risk - decile lift and cumulative gains")

    best_pipeline = load_model(models_dir / f"{best_name}.joblib")
    print("computing permutation importance (this takes a moment)...")
    permutation = permutation_importance(
        best_pipeline, X_test, y_test, n_repeats=5, random_state=42, scoring="roc_auc", n_jobs=-1
    )
    order = np.argsort(permutation.importances_mean)[::-1]
    permutation_rows = [
        {
            "feature": X_test.columns[i],
            "importance": round(float(permutation.importances_mean[i]), 5),
            "std": round(float(permutation.importances_std[i]), 5),
        }
        for i in order[:25]
    ]
    plot_feature_importance(
        [X_test.columns[i] for i in order[:20]],
        [float(permutation.importances_mean[i]) for i in order[:20]],
        figures / "permutation_importance.png",
        title="Permutation importance (drop in ROC-AUC when shuffled)",
    )

    shap_rows = explain_with_shap(best_pipeline, X_test.sample(min(800, len(X_test)), random_state=1), reports_dir)

    results = {
        "best_model": best_name,
        "model_comparison": summary,
        "thresholds": {
            "default": 0.50,
            "max_f1": round(f1_threshold, 3),
            "max_profit": round(float(profit_threshold), 3),
        },
        "performance_at_f1_threshold": at_f1,
        "performance_at_profit_threshold": at_profit,
        "lift_table": lift,
        "campaign_simulation": campaign_rows,
        "campaign_optimum": campaign_best,
        "campaign_assumptions": {
            "retention_offer_cost": RETENTION_OFFER_COST,
            "contact_cost": CONTACT_COST,
            "offer_success_rate": OFFER_SUCCESS_RATE,
            "expected_remaining_months": EXPECTED_REMAINING_MONTHS,
        },
        "permutation_importance": permutation_rows,
        "shap_importance": shap_rows,
    }
    save_json(results, reports_dir / "evaluation.json")

    # --- console summary --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("MODEL COMPARISON (hold-out set)")
    print("=" * 78)
    print(f"{'model':<28}{'ROC-AUC':>9}{'PR-AUC':>9}{'F1':>8}{'Brier':>9}{'ECE':>8}")
    for name, entry in summary.items():
        print(
            f"{name:<28}{entry['roc_auc']:>9.4f}{entry['pr_auc']:>9.4f}"
            f"{entry['f1']:>8.4f}{entry['brier']:>9.4f}{entry['ece']:>8.4f}"
        )

    print("\nTHRESHOLD SELECTION")
    print(f"  max-F1 threshold    : {f1_threshold:.2f}  "
          f"(precision={at_f1['precision']:.3f}, recall={at_f1['recall']:.3f})")
    print(f"  max-profit threshold: {profit_threshold:.2f}  "
          f"(precision={at_profit['precision']:.3f}, recall={at_profit['recall']:.3f})")
    if campaign_best:
        print(
            f"  campaign: contact {campaign_best['contacted_share']:.0%} of the base, "
            f"reach {campaign_best['recall']:.0%} of churners, "
            f"net profit {campaign_best['net_profit']:,.0f} (ROI {campaign_best['roi']:.2f}x)"
        )

    print("\nTOP RISK DRIVERS (permutation importance)")
    for row in permutation_rows[:10]:
        print(f"  {row['feature']:<32}{row['importance']:.5f}")

    print(f"\nfigures -> {figures}")
    print(f"metrics -> {reports_dir / 'evaluation.json'}")
    return results


def main() -> None:
    here = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate churn models.")
    parser.add_argument(
        "--features", type=Path, default=here / "data" / "processed" / "customers_features.csv"
    )
    parser.add_argument("--models", type=Path, default=here / "models")
    parser.add_argument("--reports", type=Path, default=here / "reports")
    args = parser.parse_args()
    evaluate(args.features, args.models, args.reports)


if __name__ == "__main__":
    main()
