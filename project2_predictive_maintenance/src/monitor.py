"""Step 6 of the approach: the fleet monitoring view and alert trigger.

Produces what a maintenance planner would actually open each morning:

* every machine's current failure-risk score, ranked,
* an alert flag for machines above the cost-optimal threshold chosen in `evaluate.py`,
* the sensors driving each alert, from the per-machine SHAP contributions, so the
  work order says "vibration trend and excess temperature" rather than "the model
  says so",
* a 72-hour risk trajectory, because a machine at 0.4 and falling is a different
  situation from a machine at 0.4 and climbing.

Outputs `reports/fleet_status.csv`, `reports/alerts.json` and two figures.
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

from common.explain import aggregate_to_source_columns, shap_values_for_pipeline  # noqa: E402
from common.io_utils import load_json, load_model, save_json  # noqa: E402
from common.plotting import PALETTE  # noqa: E402
from src.features import feature_columns  # noqa: E402

warnings.filterwarnings("ignore")

TRAJECTORY_HOURS = 72
DEFAULT_THRESHOLD = 0.5

# Maps a feature name to the plain-language symptom it represents. Matching is by
# substring, longest first, so `vibration_rms_std_24h` resolves before `vibration_rms`.
SYMPTOM_PATTERNS: Dict[str, str] = {
    "vibration_rms_std": "unstable vibration (rising variability)",
    "vibration_rms_delta": "vibration rising sharply",
    "vibration_per_rpm": "vibration high for the current speed",
    "vibration_rms": "elevated vibration",
    "excess_temp_over_ambient": "running hot relative to ambient",
    "temperature_c_delta": "temperature climbing",
    "temperature_c_std": "unstable temperature",
    "temperature_c": "elevated temperature",
    "oil_quality_idx": "oil quality degrading",
    "oil_degradation": "oil quality degrading",
    "pressure_bar_delta": "pressure dropping",
    "pressure_bar": "abnormal pressure",
    "power_per_rpm": "drawing excess power for the current speed",
    "power_kw": "abnormal power draw",
    "acoustic_db": "abnormal acoustic signature",
    "rpm": "speed deviating from nominal",
    "sensors_abnormal": "several sensors abnormal at once",
    "max_abs_z_score": "a sensor far outside this machine's normal range",
    "mean_abs_z_score": "broad deviation from this machine's baseline",
    "hours_since_maintenance": "overdue for scheduled service",
    "maintenance_overdue_ratio": "overdue for scheduled service",
    "load": "unusual duty cycle",
    "efficiency_proxy": "efficiency falling",
    "thermal_load_ratio": "thermal load out of balance",
}


def risk_band(probability: float, threshold: float) -> str:
    """Bands are anchored to the chosen alert threshold, not to arbitrary round numbers."""
    if probability >= max(threshold * 2.0, 0.75):
        return "Critical"
    if probability >= threshold:
        return "Alert"
    if probability >= threshold * 0.5:
        return "Watch"
    return "Normal"


def _symptom_for(feature: str) -> str | None:
    for pattern in sorted(SYMPTOM_PATTERNS, key=len, reverse=True):
        if pattern in feature:
            return SYMPTOM_PATTERNS[pattern]
    return None


def _symptoms_from_contributions(
    contributions: np.ndarray, columns: List[str], max_symptoms: int = 3
) -> str:
    """Translate the strongest risk-increasing features into distinct symptoms."""
    symptoms: List[str] = []
    for index in np.argsort(contributions)[::-1]:
        if contributions[index] <= 0:
            break
        symptom = _symptom_for(columns[index])
        if symptom and symptom not in symptoms:
            symptoms.append(symptom)
        if len(symptoms) >= max_symptoms:
            break
    return "; ".join(symptoms) if symptoms else "no single dominant driver"


def build_fleet_view(
    features_path: Path, models_dir: Path, reports_dir: Path
) -> tuple[pd.DataFrame, List[dict]]:
    frame = pd.read_parquet(features_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    numeric, categorical = feature_columns(frame)
    model_columns = numeric + categorical

    card = load_json(models_dir / "model_card.json")
    calibrated = load_model(models_dir / "best_model.joblib")

    # The alert threshold comes from the cost analysis, so monitoring and evaluation
    # cannot drift apart.
    threshold = DEFAULT_THRESHOLD
    evaluation_path = reports_dir / "evaluation.json"
    if evaluation_path.exists():
        threshold = float(
            load_json(evaluation_path).get("cost_optimum", {}).get("threshold", DEFAULT_THRESHOLD)
        )

    # --- current state: the latest reading for every machine ---------------------------
    latest_index = frame.groupby("machine_id")["timestamp"].idxmax()
    latest = frame.loc[latest_index].sort_values("machine_id").reset_index(drop=True)
    latest["failure_risk"] = np.clip(
        calibrated.predict_proba(latest[model_columns])[:, 1], 0.0005, 0.9995
    )
    latest["risk_band"] = latest["failure_risk"].map(lambda p: risk_band(p, threshold))
    latest["alert"] = latest["failure_risk"] >= threshold

    # --- 72-hour trajectory, to separate rising risk from steady risk -----------------
    cutoff_time = frame["timestamp"].max() - pd.Timedelta(hours=TRAJECTORY_HOURS)
    recent = frame[frame["timestamp"] >= cutoff_time].copy()
    recent["failure_risk"] = calibrated.predict_proba(recent[model_columns])[:, 1]

    trajectory = recent.pivot_table(
        index="timestamp", columns="machine_id", values="failure_risk", aggfunc="mean"
    )
    risk_24h_ago = (
        recent[recent["timestamp"] <= frame["timestamp"].max() - pd.Timedelta(hours=24)]
        .groupby("machine_id")["failure_risk"]
        .mean()
    )
    latest["risk_change_24h"] = (
        latest["failure_risk"] - latest["machine_id"].map(risk_24h_ago)
    ).round(4)
    latest["trend"] = np.where(
        latest["risk_change_24h"] > 0.05, "rising",
        np.where(latest["risk_change_24h"] < -0.05, "falling", "stable"),
    )

    # --- why: per-machine SHAP contributions -------------------------------------------
    best_pipeline = load_model(models_dir / f"{card['best_model']}.joblib")
    values, names = shap_values_for_pipeline(best_pipeline, latest[model_columns])
    if values is not None:
        aggregated = aggregate_to_source_columns(values, names, model_columns).to_numpy()
        latest["symptoms"] = [
            _symptoms_from_contributions(aggregated[row], model_columns)
            for row in range(len(latest))
        ]
    else:
        latest["symptoms"] = "explanation unavailable"

    fleet = latest[
        [
            "machine_id", "machine_type", "criticality", "age_years", "timestamp",
            "failure_risk", "risk_band", "alert", "risk_change_24h", "trend",
            "hours_since_maintenance", "temperature_c", "vibration_rms", "pressure_bar",
            "oil_quality_idx", "power_kw", "max_abs_z_score", "sensors_abnormal_2sigma",
            "symptoms",
        ]
    ].sort_values("failure_risk", ascending=False).reset_index(drop=True)
    fleet["failure_risk"] = fleet["failure_risk"].round(4)

    alerts = [
        {
            "machine_id": row["machine_id"],
            "machine_type": row["machine_type"],
            "criticality": row["criticality"],
            "failure_risk": float(row["failure_risk"]),
            "risk_band": row["risk_band"],
            "trend": row["trend"],
            "risk_change_24h": float(row["risk_change_24h"]) if pd.notna(row["risk_change_24h"]) else 0.0,
            "symptoms": row["symptoms"],
            "recommended_action": (
                "Stop and inspect before next shift" if row["risk_band"] == "Critical"
                else "Schedule inspection within 24 hours"
            ),
            "as_of": str(row["timestamp"]),
        }
        for _, row in fleet[fleet["alert"]].iterrows()
    ]

    reports_dir.mkdir(parents=True, exist_ok=True)
    fleet.to_csv(reports_dir / "fleet_status.csv", index=False)
    save_json(
        {
            "as_of": str(frame["timestamp"].max()),
            "alert_threshold": round(threshold, 4),
            "machines_monitored": int(len(fleet)),
            "alerts": alerts,
            "band_counts": fleet["risk_band"].value_counts().to_dict(),
        },
        reports_dir / "alerts.json",
    )

    _plot_fleet(fleet, trajectory, threshold, reports_dir / "figures")

    # --- console view --------------------------------------------------------------------
    print(f"fleet status as of {frame['timestamp'].max()}")
    print(f"alert threshold: {threshold:.2f} (cost-optimal, from evaluate.py)")
    print(f"\nband counts: {fleet['risk_band'].value_counts().to_dict()}")
    print(f"\n{len(alerts)} machine(s) above the alert threshold:\n")
    for alert in alerts[:15]:
        print(f"  {alert['machine_id']}  risk={alert['failure_risk']:.3f}  "
              f"{alert['risk_band']:<9}{alert['trend']:<9}{alert['machine_type']} "
              f"({alert['criticality']} criticality)")
        print(f"      symptoms: {alert['symptoms']}")
        print(f"      action  : {alert['recommended_action']}")

    print("\nTOP 10 BY RISK")
    print(f"  {'machine':<11}{'risk':>7}{'24h':>8}{'trend':>9}  {'band':<9}{'type'}")
    for _, row in fleet.head(10).iterrows():
        change = row["risk_change_24h"] if pd.notna(row["risk_change_24h"]) else 0.0
        print(f"  {row['machine_id']:<11}{row['failure_risk']:>7.3f}{change:>+8.3f}"
              f"{row['trend']:>9}  {row['risk_band']:<9}{row['machine_type']}")

    print(f"\nwrote -> {reports_dir / 'fleet_status.csv'}")
    print(f"wrote -> {reports_dir / 'alerts.json'}")
    return fleet, alerts


def _plot_fleet(
    fleet: pd.DataFrame, trajectory: pd.DataFrame, threshold: float, figures: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)

    # 1. Ranked risk bar chart for the whole fleet.
    fig, ax = plt.subplots(figsize=(13, 4.8))
    colours = {
        "Critical": PALETTE[4], "Alert": PALETTE[1], "Watch": PALETTE[5], "Normal": PALETTE[2]
    }
    ax.bar(
        range(len(fleet)), fleet["failure_risk"],
        color=[colours.get(band, PALETTE[0]) for band in fleet["risk_band"]],
    )
    ax.axhline(threshold, ls="--", color="black", lw=1.4, label=f"alert threshold {threshold:.2f}")
    ax.set_xlabel("Machines, ranked by current failure risk")
    ax.set_ylabel("P(failure within 24h)")
    ax.set_title("Fleet failure risk")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for colour in colours.values()]
    ax.legend(
        handles + [plt.Line2D([0], [0], ls="--", color="black")],
        list(colours) + [f"threshold {threshold:.2f}"],
        fontsize=8, ncol=5, loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(figures / "fleet_risk.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 2. Risk trajectories for the highest-risk machines.
    top_machines = fleet.head(6)["machine_id"].tolist()
    available = [machine for machine in top_machines if machine in trajectory.columns]
    if available:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        for index, machine in enumerate(available):
            ax.plot(
                trajectory.index, trajectory[machine].rolling(3, min_periods=1).mean(),
                lw=2, color=PALETTE[index % len(PALETTE)], label=machine,
            )
        ax.axhline(threshold, ls="--", color="black", lw=1.2)
        ax.set_xlabel("Time")
        ax.set_ylabel("P(failure within 24h)")
        ax.set_title(f"Risk trajectory, last {TRAJECTORY_HOURS} hours - highest-risk machines")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(figures / "risk_trajectory.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    here = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Build the fleet monitoring view.")
    parser.add_argument(
        "--features", type=Path, default=here / "data" / "processed" / "features.parquet"
    )
    parser.add_argument("--models", type=Path, default=here / "models")
    parser.add_argument("--reports", type=Path, default=here / "reports")
    args = parser.parse_args()
    build_fleet_view(args.features, args.models, args.reports)


if __name__ == "__main__":
    main()
