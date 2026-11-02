"""Streamlit monitoring dashboard for the predictive maintenance model.

    streamlit run app.py

This is the deliverable from step 6 of the brief: a monitoring view showing machine risk
scores with alerts for high-risk equipment. The threshold explorer is the important part
-- it makes the recall/false-alarm trade-off something a planner can feel rather than
read about.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).absolute().parent
REPORTS = HERE / "reports"
FIGURES = REPORTS / "figures"
PROCESSED = HERE / "data" / "processed"

SENSORS = [
    "temperature_c", "vibration_rms", "pressure_bar", "rpm",
    "oil_quality_idx", "power_kw", "acoustic_db",
]

st.set_page_config(page_title="Fleet Maintenance Monitor", page_icon="🔧", layout="wide")


@st.cache_data
def load_fleet() -> tuple[pd.DataFrame, dict, dict]:
    fleet = pd.read_csv(REPORTS / "fleet_status.csv")
    alerts = json.loads((REPORTS / "alerts.json").read_text())
    evaluation = json.loads((REPORTS / "evaluation.json").read_text())
    return fleet, alerts, evaluation


@st.cache_data
def load_telemetry() -> pd.DataFrame:
    """Raw-ish telemetry for the per-machine sensor charts."""
    frame = pd.read_parquet(
        PROCESSED / "features.parquet",
        columns=["machine_id", "timestamp", "label", "hours_to_failure", *SENSORS],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


if not (REPORTS / "fleet_status.csv").exists():
    st.error("Monitoring outputs not found.")
    st.markdown(
        "Run the pipeline first:\n\n"
        "```bash\n"
        "python -m src.generate_data\n"
        "python -m src.labeling\n"
        "python -m src.features\n"
        "python -m src.train\n"
        "python -m src.evaluate\n"
        "python -m src.monitor\n"
        "```"
    )
    st.stop()

fleet, alerts, evaluation = load_fleet()
threshold = alerts["alert_threshold"]

st.title("Industrial Fleet — Predictive Maintenance Monitor")
st.caption(
    f"As of **{alerts['as_of']}** · {alerts['machines_monitored']} machines · "
    f"alert threshold **{threshold:.2f}** (cost-optimal)"
)

banner = st.columns(5)
bands = alerts["band_counts"]
banner[0].metric("Critical", bands.get("Critical", 0))
banner[1].metric("Alert", bands.get("Alert", 0))
banner[2].metric("Watch", bands.get("Watch", 0))
banner[3].metric("Normal", bands.get("Normal", 0))
banner[4].metric("Rising risk (24h)", int((fleet["trend"] == "rising").sum()))

overview, machine_view, performance, threshold_tab = st.tabs(
    ["Fleet overview", "Machine detail", "Model performance", "Threshold & cost"]
)

# --------------------------------------------------------------------------------------
with overview:
    if alerts["alerts"]:
        st.subheader(f"🚨 {len(alerts['alerts'])} machine(s) require attention")
        for alert in alerts["alerts"]:
            icon = "🔴" if alert["risk_band"] == "Critical" else "🟠"
            with st.expander(
                f"{icon} **{alert['machine_id']}** — risk {alert['failure_risk']:.1%} "
                f"({alert['trend']}) — {alert['machine_type']}",
                expanded=alert["risk_band"] == "Critical",
            ):
                columns = st.columns(4)
                columns[0].metric("Failure risk (24h)", f"{alert['failure_risk']:.1%}")
                columns[1].metric("24h change", f"{alert['risk_change_24h']:+.3f}")
                columns[2].metric("Criticality", alert["criticality"])
                columns[3].metric("Band", alert["risk_band"])
                st.markdown(f"**Symptoms:** {alert['symptoms']}")
                st.warning(f"**Action:** {alert['recommended_action']}")
    else:
        st.success("No machines above the alert threshold.")

    st.divider()
    if (FIGURES / "fleet_risk.png").exists():
        st.image(str(FIGURES / "fleet_risk.png"), use_container_width=True)
    if (FIGURES / "risk_trajectory.png").exists():
        st.image(str(FIGURES / "risk_trajectory.png"), use_container_width=True)

    st.subheader("All machines")
    filters = st.columns(3)
    types = filters[0].multiselect(
        "Machine type", sorted(fleet["machine_type"].unique()),
        default=sorted(fleet["machine_type"].unique()),
    )
    criticalities = filters[1].multiselect(
        "Criticality", sorted(fleet["criticality"].unique()),
        default=sorted(fleet["criticality"].unique()),
    )
    only_rising = filters[2].checkbox("Only rising risk", value=False)

    view = fleet[fleet["machine_type"].isin(types) & fleet["criticality"].isin(criticalities)]
    if only_rising:
        view = view[view["trend"] == "rising"]

    st.dataframe(
        view[["machine_id", "machine_type", "criticality", "failure_risk", "risk_band",
              "risk_change_24h", "trend", "hours_since_maintenance",
              "sensors_abnormal_2sigma", "symptoms"]],
        use_container_width=True, hide_index=True,
    )

# --------------------------------------------------------------------------------------
with machine_view:
    st.subheader("Per-machine sensor history")
    chosen = st.selectbox("Machine", fleet["machine_id"].tolist())
    record = fleet[fleet["machine_id"] == chosen].iloc[0]

    header = st.columns(5)
    header[0].metric("Failure risk", f"{record['failure_risk']:.1%}")
    header[1].metric("Band", record["risk_band"])
    header[2].metric("Type", record["machine_type"])
    header[3].metric("Age", f"{record['age_years']:.1f} yr")
    header[4].metric("Since service", f"{record['hours_since_maintenance']:.0f} h")
    st.info(f"**Symptoms:** {record['symptoms']}")

    telemetry = load_telemetry()
    machine_rows = telemetry[telemetry["machine_id"] == chosen].sort_values("timestamp")
    window = st.slider("Hours of history", 48, min(1200, len(machine_rows)), 336, step=24)
    recent = machine_rows.tail(window).set_index("timestamp")

    sensor_choice = st.multiselect(
        "Sensors", SENSORS, default=["vibration_rms", "temperature_c", "oil_quality_idx"]
    )
    if sensor_choice:
        # Each sensor has its own scale, so they are charted separately rather than
        # squashed onto one axis.
        for sensor in sensor_choice:
            st.markdown(f"**{sensor}**")
            st.line_chart(recent[[sensor]], height=180)

    failure_windows = recent[recent["label"] == 1]
    if len(failure_windows):
        st.caption(
            f"{len(failure_windows)} hour(s) in this window were inside a true "
            f"24-hour pre-failure period."
        )

# --------------------------------------------------------------------------------------
with performance:
    detection = evaluation["event_level_detection"]
    headline = st.columns(4)
    headline[0].metric("Failures in test period", detection["failures_in_test"])
    headline[1].metric("Failures caught", detection["failures_detected"],
                       f"{detection['event_level_recall']:.1%}")
    headline[2].metric("Median warning", f"{detection['median_lead_time_hours']:.0f} h")
    headline[3].metric("Mean warning", f"{detection['mean_lead_time_hours']:.0f} h")

    st.caption(
        "Event-level recall is the headline number: of the failures that occurred, how "
        "many produced at least one alert in time to act. Row-level recall counts "
        "machine-hours and reads higher, but nobody schedules maintenance by the hour."
    )

    st.subheader("Detection by failure type")
    left, right = st.columns(2)
    with left:
        suddenness = pd.DataFrame(detection["by_suddenness"])
        suddenness["type"] = suddenness["is_sudden"].map({True: "sudden", False: "gradual"})
        st.dataframe(
            suddenness[["type", "failures", "detected", "detection_rate"]],
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Sudden failures ramp over 2-8 hours by construction. Their low detection "
            "rate is a property of the problem, not a defect in the model."
        )
    with right:
        st.dataframe(
            pd.DataFrame(detection["by_failure_mode"]),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Model comparison")
    table = pd.DataFrame(evaluation["model_comparison"]).T
    st.dataframe(
        table[["pr_auc", "roc_auc", "precision", "recall", "f1", "brier", "ece"]]
        .style.format("{:.4f}")
        .background_gradient(subset=["pr_auc"], cmap="Greens"),
        use_container_width=True,
    )

    for name, caption in [
        ("pr_curves.png", "Precision-recall — the metric that matters with 5% positives"),
        ("lead_time.png", "Warning lead time, and gradual vs. sudden detection"),
        ("shap_importance.png", "What the model is actually using"),
        ("calibration.png", "Calibration of the risk scores"),
        ("confusion_matrix.png", "Confusion matrix at the chosen threshold"),
    ]:
        if (FIGURES / name).exists():
            st.image(str(FIGURES / name), caption=caption, use_container_width=True)

# --------------------------------------------------------------------------------------
with threshold_tab:
    st.subheader("Choosing the alert threshold")
    assumptions = evaluation["cost_assumptions"]
    st.caption(
        f"Current assumptions: unplanned failure "
        f"{assumptions['unplanned_failure_cost']:,.0f}, inspection "
        f"{assumptions['planned_inspection_cost']:,.0f}, "
        f"{assumptions['inspection_prevention_rate']:.0%} of caught failures averted."
    )

    cost = pd.DataFrame(evaluation["cost_analysis"])
    optimum = evaluation["cost_optimum"]

    headline = st.columns(4)
    headline[0].metric("Cost-optimal threshold", f"{optimum['threshold']:.2f}")
    headline[1].metric("Failures caught", f"{optimum['event_level_recall']:.1%}")
    headline[2].metric("Alerts / machine / week", f"{optimum['alerts_per_machine_per_week']:.2f}")
    headline[3].metric("Saving vs. reactive", f"{optimum['saving_vs_reactive']:,.0f}")

    picked = st.select_slider(
        "Explore a different threshold", options=cost["threshold"].tolist(),
        value=optimum["threshold"],
    )
    row = cost[cost["threshold"] == picked].iloc[0]
    explore = st.columns(5)
    explore[0].metric("Event recall", f"{row['event_level_recall']:.1%}")
    explore[1].metric("Alert episodes", int(row["alert_episodes"]))
    explore[2].metric("Alerts / machine / week", f"{row['alerts_per_machine_per_week']:.2f}")
    explore[3].metric("Mean lead time", f"{row['mean_lead_time_hours']:.0f} h")
    explore[4].metric("Total cost", f"{row['total_cost']:,.0f}",
                      f"{row['total_cost'] - optimum['total_cost']:+,.0f} vs optimum")

    st.line_chart(cost.set_index("threshold")[["total_cost", "inspection_cost", "failure_cost"]])

    for name in ("cost_curve.png", "recall_vs_alerts.png"):
        if (FIGURES / name).exists():
            st.image(str(FIGURES / name), use_container_width=True)

    st.subheader("Operating points at fixed recall")
    st.dataframe(pd.DataFrame(evaluation["recall_targets"]), use_container_width=True, hide_index=True)
