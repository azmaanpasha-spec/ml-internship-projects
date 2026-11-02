"""Step 5 of the approach: recall, precision, PR-AUC and -- crucially -- false alarms.

Row-level metrics answer "of the machine-hours inside a failure window, how many did we
flag?". That is not the question a maintenance planner asks. They ask three things:

1. **Of the actual failures, how many did we catch at all?** One alert at any point in
   the window is a catch. This is *event-level recall*, computed in
   `event_level_detection()`, and it is the headline number of the project.
2. **How much warning did we get?** A catch three hours out is far less valuable than one
   three days out. `event_level_detection()` records the lead time of the first alert.
3. **How many false alarms will my team chase per week?** Precision alone is abstract;
   alerts-per-machine-per-week is what destroys trust in a system. Consecutive alerts are
   collapsed into *episodes*, because ten alerts in ten consecutive hours is one
   maintenance decision, not ten.

The module then prices those trade-offs and recommends the threshold that minimises total
expected cost, rather than defaulting to 0.5.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

from common.evaluation import (  # noqa: E402
    classification_report,
    expected_calibration_error,
    threshold_for_recall,
)
from common.explain import shap_values_for_pipeline  # noqa: E402
from common.io_utils import load_json, load_model, save_json  # noqa: E402
from common.plotting import (  # noqa: E402
    PALETTE,
    plot_calibration,
    plot_confusion,
    plot_feature_importance,
    plot_metric_bars,
    plot_pr_curves,
    plot_roc_curves,
)
from src.features import feature_columns  # noqa: E402
from src.train import time_based_split  # noqa: E402

warnings.filterwarnings("ignore")

# Maintenance economics. An unplanned failure costs production downtime and emergency
# labour; a precautionary inspection costs a slot and an engineer.
UNPLANNED_FAILURE_COST = 18_000.0
PLANNED_INSPECTION_COST = 450.0
INSPECTION_PREVENTION_RATE = 0.75   # share of caught failures actually averted
ALERT_COOLDOWN_HOURS = 24           # alerts within this window are one episode
RECALL_TARGETS = (0.70, 0.80, 0.90)


def alert_episodes(
    frame: pd.DataFrame, scores: np.ndarray, threshold: float, cooldown_hours: int = ALERT_COOLDOWN_HOURS
) -> pd.DataFrame:
    """Collapse consecutive alerts into distinct episodes, per machine.

    A technician responds to an alert once and then watches the machine; the next ten
    hourly alerts are the same decision. Counting raw alerting rows overstates the
    operational burden by an order of magnitude.
    """
    alerting = frame.loc[scores >= threshold, ["machine_id", "timestamp"]].copy()
    if alerting.empty:
        return pd.DataFrame(columns=["machine_id", "timestamp", "episode_id"])

    alerting = alerting.sort_values(["machine_id", "timestamp"])
    gap_hours = (
        alerting.groupby("machine_id")["timestamp"].diff().dt.total_seconds() / 3600.0
    )
    new_episode = gap_hours.isna() | (gap_hours > cooldown_hours)
    alerting["episode_id"] = new_episode.cumsum()
    return alerting.groupby("episode_id").first().reset_index()


def event_level_detection(
    frame: pd.DataFrame, scores: np.ndarray, threshold: float
) -> Dict[str, object]:
    """Per-failure detection rate and warning lead time.

    For every failure whose prediction window appears in the test set, the failure counts
    as *detected* if the model scored any hour in that window above the threshold. Lead
    time is measured from the first such hour to the failure itself.
    """
    working = frame[["machine_id", "timestamp", "label", "hours_to_failure",
                     "next_failure_time", "next_failure_mode", "next_failure_is_sudden"]].copy()
    working["score"] = scores
    windows = working[working["label"] == 1]

    records: List[dict] = []
    for (machine_id, failure_time), rows in windows.groupby(["machine_id", "next_failure_time"]):
        triggered = rows[rows["score"] >= threshold]
        detected = len(triggered) > 0
        records.append(
            {
                "machine_id": machine_id,
                "failure_time": failure_time,
                "failure_mode": rows["next_failure_mode"].iloc[0],
                "is_sudden": bool(rows["next_failure_is_sudden"].iloc[0]),
                "window_hours": len(rows),
                "detected": detected,
                # Lead time of the earliest alert in the window.
                "lead_time_hours": float(triggered["hours_to_failure"].max()) if detected else 0.0,
                "max_score": float(rows["score"].max()),
            }
        )

    detections = pd.DataFrame(records)
    if detections.empty:
        return {"failures_in_test": 0}

    caught = detections[detections["detected"]]

    def _breakdown(column: str) -> List[dict]:
        """Detection rate *and* lead time per group.

        Lead time is the more informative of the two. A sudden failure is still usually
        detected -- the alert window runs right up to the failure itself -- but it is
        detected two hours out instead of a day out, which is the difference between
        scheduling a repair and watching a machine break.
        """
        grouped = detections.groupby(column).agg(
            failures=("detected", "size"), detected=("detected", "sum")
        )
        lead = caught.groupby(column)["lead_time_hours"]
        grouped["detection_rate"] = (grouped["detected"] / grouped["failures"]).round(4)
        grouped["mean_lead_time_hours"] = lead.mean().round(2)
        grouped["median_lead_time_hours"] = lead.median().round(2)
        return grouped.reset_index().fillna(0.0).to_dict(orient="records")

    by_mode = _breakdown("failure_mode")
    by_suddenness = _breakdown("is_sudden")

    return {
        "threshold": round(float(threshold), 4),
        "failures_in_test": int(len(detections)),
        "failures_detected": int(detections["detected"].sum()),
        "event_level_recall": round(float(detections["detected"].mean()), 4),
        "mean_lead_time_hours": round(float(caught["lead_time_hours"].mean()), 2) if len(caught) else 0.0,
        "median_lead_time_hours": round(float(caught["lead_time_hours"].median()), 2) if len(caught) else 0.0,
        "by_failure_mode": by_mode,
        "by_suddenness": by_suddenness,
        "_detections": detections,
    }


def cost_analysis(
    frame: pd.DataFrame, scores: np.ndarray, n_machines: int, test_days: float
) -> tuple[List[dict], dict]:
    """Expected cost across thresholds, and the cost-optimal operating point."""
    rows: List[dict] = []

    for threshold in np.round(np.arange(0.05, 0.96, 0.05), 2):
        detection = event_level_detection(frame, scores, threshold)
        if not detection.get("failures_in_test"):
            continue

        episodes = alert_episodes(frame, scores, threshold)
        n_episodes = len(episodes)

        detected = detection["failures_detected"]
        missed = detection["failures_in_test"] - detected
        prevented = detected * INSPECTION_PREVENTION_RATE

        inspection_cost = n_episodes * PLANNED_INSPECTION_COST
        failure_cost = (missed + detected * (1 - INSPECTION_PREVENTION_RATE)) * UNPLANNED_FAILURE_COST
        do_nothing_cost = detection["failures_in_test"] * UNPLANNED_FAILURE_COST

        rows.append(
            {
                "threshold": float(threshold),
                "alert_episodes": n_episodes,
                "alerts_per_machine_per_week": round(
                    n_episodes / max(n_machines, 1) / max(test_days / 7.0, 1e-6), 3
                ),
                "event_level_recall": detection["event_level_recall"],
                "failures_prevented": round(float(prevented), 1),
                "mean_lead_time_hours": detection["mean_lead_time_hours"],
                "inspection_cost": round(inspection_cost, 2),
                "failure_cost": round(failure_cost, 2),
                "total_cost": round(inspection_cost + failure_cost, 2),
                "saving_vs_reactive": round(do_nothing_cost - inspection_cost - failure_cost, 2),
            }
        )

    best = min(rows, key=lambda row: row["total_cost"]) if rows else {}
    return rows, best


def plot_lead_time(detections: pd.DataFrame, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caught = detections[detections["detected"]]
    gradual = caught[~caught["is_sudden"]]["lead_time_hours"]
    sudden = caught[caught["is_sudden"]]["lead_time_hours"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Left: the lead-time distribution, separated by onset type. This is the plot that
    # shows why detection rate alone is a misleading summary.
    bins = np.arange(0, 26, 1)
    if len(gradual):
        ax1.hist(gradual, bins=bins, color=PALETTE[2], alpha=0.85,
                 label=f"gradual onset (n={len(gradual)}, median {gradual.median():.0f}h)")
    if len(sudden):
        ax1.hist(sudden, bins=bins, color=PALETTE[4], alpha=0.85,
                 label=f"sudden onset (n={len(sudden)}, median {sudden.median():.0f}h)")
    ax1.set_xlabel("Warning lead time (hours before failure)")
    ax1.set_ylabel("Failures caught")
    ax1.set_title("How much warning the model actually bought")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    # Right: detection rate and mean lead time side by side, same groups.
    grouped = detections.groupby("is_sudden").agg(
        rate=("detected", "mean"), failures=("detected", "size")
    )
    lead_means = caught.groupby("is_sudden")["lead_time_hours"].mean()
    labels = ["gradual" if not index else "sudden" for index in grouped.index]
    x = np.arange(len(labels))

    bars = ax2.bar(x - 0.2, grouped["rate"].to_numpy(), 0.4, color=PALETTE[2],
                   label="detection rate")
    ax2.bar_label(bars, fmt="%.2f", fontsize=8)
    ax2.set_ylabel("Event-level detection rate")
    ax2.set_ylim(0, 1.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{label}\n(n={int(n)})" for label, n in zip(labels, grouped["failures"])])

    twin = ax2.twinx()
    lead_bars = twin.bar(x + 0.2, lead_means.reindex(grouped.index).to_numpy(), 0.4,
                         color=PALETTE[1], label="mean lead time")
    twin.bar_label(lead_bars, fmt="%.1fh", fontsize=8)
    twin.set_ylabel("Mean lead time (hours)")
    twin.set_ylim(0, 30)

    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="upper center")
    ax2.set_title("Caught at a similar rate — with very different warning")
    ax2.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cost_curve(rows: List[dict], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [row["threshold"] for row in rows]
    fig, ax1 = plt.subplots(figsize=(7.4, 4.8))

    ax1.plot(thresholds, [row["total_cost"] for row in rows], "o-", color=PALETTE[0], lw=2,
             label="total expected cost")
    best = min(rows, key=lambda row: row["total_cost"])
    ax1.axvline(best["threshold"], color=PALETTE[1], ls="--", lw=2,
                label=f"optimum @ {best['threshold']:.2f}")
    ax1.set_xlabel("Alert threshold")
    ax1.set_ylabel("Expected cost")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, [row["alerts_per_machine_per_week"] for row in rows], "s--",
             color=PALETTE[2], lw=1.6, label="alerts / machine / week")
    ax2.set_ylabel("Alert episodes per machine per week")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="upper center")
    ax1.set_title("Choosing the alert threshold on cost, not convention")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_recall_vs_alerts(rows: List[dict], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(
        [row["alerts_per_machine_per_week"] for row in rows],
        [row["event_level_recall"] for row in rows],
        "o-", color=PALETTE[3], lw=2,
    )
    for row in rows[::3]:
        ax.annotate(
            f"{row['threshold']:.2f}",
            (row["alerts_per_machine_per_week"], row["event_level_recall"]),
            textcoords="offset points", xytext=(6, -10), fontsize=7,
        )
    ax.set_xlabel("Alert episodes per machine per week (operational burden)")
    ax.set_ylabel("Event-level recall (failures caught)")
    ax.set_title("The real trade-off: failures caught vs. alarms chased")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def evaluate(features_path: Path, models_dir: Path, reports_dir: Path) -> dict:
    frame = pd.read_parquet(features_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    numeric, categorical = feature_columns(frame)

    _, test, cutoff = time_based_split(frame)
    X_test = test[numeric + categorical]
    y_test = test["label"].astype(int).to_numpy()

    figures = reports_dir / "figures"
    card = load_json(models_dir / "model_card.json")
    best_name = card["best_model"]

    # --- model comparison on the held-out period -------------------------------------
    curves, summary = {}, {}
    for path in sorted(models_dir.glob("*.joblib")):
        if path.stem == "best_model":
            continue
        scores = load_model(path).predict_proba(X_test)[:, 1]
        curves[path.stem] = (y_test, scores)
        entry = classification_report(y_test, scores, model=path.stem, threshold=0.5).as_dict()
        entry["ece"] = round(expected_calibration_error(y_test, scores), 4)
        summary[path.stem] = entry

    calibrated = load_model(models_dir / "best_model.joblib")
    scores = calibrated.predict_proba(X_test)[:, 1]
    curves[f"{best_name} (calibrated)"] = (y_test, scores)
    calibrated_entry = classification_report(
        y_test, scores, model=f"{best_name}_calibrated", threshold=0.5
    ).as_dict()
    calibrated_entry["ece"] = round(expected_calibration_error(y_test, scores), 4)
    summary[f"{best_name}_calibrated"] = calibrated_entry

    plot_roc_curves(curves, figures / "roc_curves.png", "Failure prediction - ROC")
    plot_pr_curves(curves, figures / "pr_curves.png", "Failure prediction - precision/recall")
    plot_calibration(curves, figures / "calibration.png", title="Failure risk - calibration")
    plot_metric_bars(
        summary, ["pr_auc", "roc_auc", "precision", "recall", "f1"],
        figures / "model_comparison.png",
        "Model comparison on the held-out period (threshold = 0.50)",
    )

    # --- thresholds at fixed recall targets -------------------------------------------
    n_machines = int(test["machine_id"].nunique())
    test_days = float((test["timestamp"].max() - test["timestamp"].min()).total_seconds() / 86400)

    recall_table = []
    for target in RECALL_TARGETS:
        threshold = threshold_for_recall(y_test, scores, target)
        report = classification_report(y_test, scores, model=f"recall>={target}", threshold=threshold)
        detection = event_level_detection(test, scores, threshold)
        episodes = alert_episodes(test, scores, threshold)
        recall_table.append(
            {
                "target_row_recall": target,
                "threshold": round(float(threshold), 4),
                "row_precision": report.precision,
                "row_recall": report.recall,
                "event_level_recall": detection["event_level_recall"],
                "mean_lead_time_hours": detection["mean_lead_time_hours"],
                "alert_episodes": len(episodes),
                "alerts_per_machine_per_week": round(
                    len(episodes) / max(n_machines, 1) / max(test_days / 7.0, 1e-6), 3
                ),
            }
        )

    # --- cost-optimal threshold --------------------------------------------------------
    cost_rows, cost_best = cost_analysis(test, scores, n_machines, test_days)
    chosen_threshold = cost_best.get("threshold", 0.5)
    detection = event_level_detection(test, scores, chosen_threshold)
    detections = detection.pop("_detections")

    plot_confusion(
        y_test, (scores >= chosen_threshold).astype(int),
        figures / "confusion_matrix.png",
        f"Confusion matrix at the cost-optimal threshold ({chosen_threshold:.2f})",
    )
    plot_lead_time(detections, figures / "lead_time.png")
    plot_cost_curve(cost_rows, figures / "cost_curve.png")
    plot_recall_vs_alerts(cost_rows, figures / "recall_vs_alerts.png")
    detections.drop(columns=["failure_time"]).to_csv(
        reports_dir / "failure_detections.csv", index=False
    )

    # --- what the model is using --------------------------------------------------------
    best_pipeline = load_model(models_dir / f"{best_name}.joblib")
    shap_rows: List[dict] = []
    values, names = shap_values_for_pipeline(best_pipeline, X_test.sample(min(2000, len(X_test)), random_state=1))
    if values is not None:
        mean_abs = np.abs(values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        plot_feature_importance(
            [names[i] for i in order[:22]], [float(mean_abs[i]) for i in order[:22]],
            figures / "shap_importance.png",
            title="Mean |SHAP value| - what drives the failure risk score",
        )
        shap_rows = [
            {"feature": names[i], "mean_abs_shap": round(float(mean_abs[i]), 6)}
            for i in order[:30]
        ]

    results = {
        "split_cutoff": str(cutoff),
        "test_machines": n_machines,
        "test_days": round(test_days, 1),
        "model_comparison": summary,
        "recall_targets": recall_table,
        "cost_analysis": cost_rows,
        "cost_optimum": cost_best,
        "cost_assumptions": {
            "unplanned_failure_cost": UNPLANNED_FAILURE_COST,
            "planned_inspection_cost": PLANNED_INSPECTION_COST,
            "inspection_prevention_rate": INSPECTION_PREVENTION_RATE,
            "alert_cooldown_hours": ALERT_COOLDOWN_HOURS,
        },
        "event_level_detection": detection,
        "shap_importance": shap_rows,
    }
    save_json(results, reports_dir / "evaluation.json")

    # --- console report -----------------------------------------------------------------
    print("\n" + "=" * 86)
    print("MODEL COMPARISON (held-out period, threshold = 0.50)")
    print("=" * 86)
    print(f"{'model':<30}{'PR-AUC':>9}{'ROC-AUC':>9}{'Prec':>8}{'Recall':>8}{'Brier':>9}{'ECE':>8}")
    for name, entry in summary.items():
        print(
            f"{name:<30}{entry['pr_auc']:>9.4f}{entry['roc_auc']:>9.4f}{entry['precision']:>8.4f}"
            f"{entry['recall']:>8.4f}{entry['brier']:>9.4f}{entry['ece']:>8.4f}"
        )

    print("\nOPERATING POINTS AT FIXED ROW-LEVEL RECALL")
    print(f"{'target':>8}{'thresh':>9}{'row prec':>10}{'event recall':>14}{'lead (h)':>10}{'alerts/mch/wk':>15}")
    for row in recall_table:
        print(
            f"{row['target_row_recall']:>8.2f}{row['threshold']:>9.3f}{row['row_precision']:>10.3f}"
            f"{row['event_level_recall']:>14.3f}{row['mean_lead_time_hours']:>10.1f}"
            f"{row['alerts_per_machine_per_week']:>15.2f}"
        )

    print(f"\nCOST-OPTIMAL THRESHOLD: {chosen_threshold:.2f}")
    if cost_best:
        print(f"  alert episodes      : {cost_best['alert_episodes']} "
              f"({cost_best['alerts_per_machine_per_week']:.2f} per machine per week)")
        print(f"  event-level recall  : {cost_best['event_level_recall']:.1%} of failures caught")
        print(f"  mean lead time      : {cost_best['mean_lead_time_hours']:.1f} hours")
        print(f"  total expected cost : {cost_best['total_cost']:,.0f}")
        print(f"  saving vs. reactive : {cost_best['saving_vs_reactive']:,.0f}")

    print(f"\nEVENT-LEVEL DETECTION at threshold {chosen_threshold:.2f}")
    print(f"  failures in test    : {detection['failures_in_test']}")
    print(f"  detected            : {detection['failures_detected']} "
          f"({detection['event_level_recall']:.1%})")
    print(f"  median lead time    : {detection['median_lead_time_hours']:.1f} h")
    print(f"  {'group':<24}{'caught':>10}{'rate':>8}{'mean lead':>11}{'median lead':>13}")
    for row in detection["by_suddenness"]:
        label = "sudden onset" if row["is_sudden"] else "gradual onset"
        print(f"    {label:<22}{int(row['detected'])}/{int(row['failures']):<8}"
              f"{row['detection_rate']:>7.1%}{row['mean_lead_time_hours']:>11.1f}h"
              f"{row['median_lead_time_hours']:>12.1f}h")
    for row in detection["by_failure_mode"]:
        print(f"    {row['failure_mode']:<22}{int(row['detected'])}/{int(row['failures']):<8}"
              f"{row['detection_rate']:>7.1%}{row['mean_lead_time_hours']:>11.1f}h"
              f"{row['median_lead_time_hours']:>12.1f}h")

    if shap_rows:
        print("\nTOP RISK DRIVERS (mean |SHAP|)")
        for row in shap_rows[:12]:
            print(f"  {row['feature']:<42}{row['mean_abs_shap']:.6f}")

    print(f"\nfigures -> {figures}")
    print(f"metrics -> {reports_dir / 'evaluation.json'}")
    return results


def main() -> None:
    here = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate predictive maintenance models.")
    parser.add_argument(
        "--features", type=Path, default=here / "data" / "processed" / "features.parquet"
    )
    parser.add_argument("--models", type=Path, default=here / "models")
    parser.add_argument("--reports", type=Path, default=here / "reports")
    args = parser.parse_args()
    evaluate(args.features, args.models, args.reports)


if __name__ == "__main__":
    main()
