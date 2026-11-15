"""Fleet monitoring console for the predictive maintenance model.

    streamlit run app.py

This is the deliverable from step 6 of the brief — a monitoring view showing machine
risk scores with alerts for high-risk equipment — built as the thing a maintenance
planner opens each morning rather than a page of metrics.

  Fleet        which machines need attention right now, and why
  Machine      the sensor evidence behind one machine's score
  Detection    did we catch the failures, and how much warning did we buy
  Model        how the four candidates compare, scored live on the held-out period
  Economics    what threshold to run at, priced against your own assumptions

The economics tab is the important one: it turns "the model has PR-AUC 0.69" into
"you will chase 0.5 alarms per machine per week and save 1.3 million".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import ui  # noqa: E402

REPORTS = HERE / "reports"
MODELS = HERE / "models"
PROCESSED = HERE / "data" / "processed"

SENSORS = ["temperature_c", "vibration_rms", "pressure_bar", "rpm",
           "oil_quality_idx", "power_kw", "acoustic_db"]
SENSOR_LABELS = {
    "temperature_c": "Temperature (°C)", "vibration_rms": "Vibration (RMS)",
    "pressure_bar": "Pressure (bar)", "rpm": "Speed (RPM)",
    "oil_quality_idx": "Oil quality index", "power_kw": "Power (kW)",
    "acoustic_db": "Acoustic (dB)",
}

ui.setup_page(
    title="Fleet Maintenance Monitor",
    icon="🔧",
    heading="Industrial Fleet — Predictive Maintenance Monitor",
    subtitle="Will this machine fail in the next 24 hours? — 80 machines, 7 sensors, hourly",
    tagline="Alerts are grouped into episodes, not rows, and the threshold is chosen by "
            "expected cost. Change the cost assumptions in Economics and the whole "
            "operating point moves.",
)

if not (REPORTS / "fleet_status.csv").exists():
    ui.missing_artefacts([
        "python -m src.generate_data",
        "python -m src.labeling",
        "python -m src.features",
        "python -m src.train",
        "python -m src.evaluate",
        "python -m src.monitor",
    ])


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_reports():
    fleet = pd.read_csv(REPORTS / "fleet_status.csv")
    alerts = json.loads((REPORTS / "alerts.json").read_text())
    evaluation = json.loads((REPORTS / "evaluation.json").read_text())
    detections = pd.read_csv(REPORTS / "failure_detections.csv")
    return fleet, alerts, evaluation, detections


@st.cache_data(show_spinner="Loading telemetry…")
def load_telemetry() -> pd.DataFrame:
    frame = pd.read_parquet(
        PROCESSED / "features.parquet",
        columns=["machine_id", "timestamp", "label", "hours_to_failure",
                 "next_failure_mode", "hours_since_maintenance", *SENSORS],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


@st.cache_resource(show_spinner="Scoring the held-out period…")
def holdout_scores():
    """Score the test period with every saved model, for live curves."""
    from common.io_utils import load_model
    from src.features import feature_columns
    from src.train import time_based_split

    frame = pd.read_parquet(PROCESSED / "features.parquet")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    numeric, categorical = feature_columns(frame)
    _, test, cutoff = time_based_split(frame)

    X_test = test[numeric + categorical]
    y_test = test["label"].astype(int).to_numpy()

    scores = {}
    for path in sorted(MODELS.glob("*.joblib")):
        if path.stem == "best_model":
            continue
        scores[path.stem] = load_model(path).predict_proba(X_test)[:, 1]
    scores["calibrated"] = load_model(MODELS / "best_model.joblib").predict_proba(X_test)[:, 1]
    return y_test, scores, cutoff


@st.cache_data(show_spinner="Scoring machine…")
def machine_risk(machine_id: str) -> pd.DataFrame:
    """Risk trajectory for one machine, scored with the calibrated model."""
    from common.io_utils import load_model
    from src.features import feature_columns

    frame = pd.read_parquet(PROCESSED / "features.parquet")
    frame = frame[frame["machine_id"] == machine_id].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    numeric, categorical = feature_columns(frame)
    model = load_model(MODELS / "best_model.joblib")
    frame["risk"] = model.predict_proba(frame[numeric + categorical])[:, 1]
    return frame[["timestamp", "risk", "label", "hours_to_failure"]].sort_values("timestamp")


fleet, alerts, evaluation, detections = load_reports()
threshold = float(alerts["alert_threshold"])
detection = evaluation["event_level_detection"]

fleet_tab, machine_tab, detect_tab, model_tab, econ_tab = st.tabs(
    ["🚨 Fleet", "🔍 Machine", "🎯 Detection", "🔬 Model", "💰 Economics"]
)


# ======================================================================================
# FLEET
# ======================================================================================
with fleet_tab:
    bands = alerts["band_counts"]
    ui.kpi_row([
        {"label": "Critical", "value": bands.get("Critical", 0), "tone": "danger"},
        {"label": "Alert", "value": bands.get("Alert", 0), "tone": "warning"},
        {"label": "Watch", "value": bands.get("Watch", 0), "tone": "teal"},
        {"label": "Normal", "value": bands.get("Normal", 0), "tone": "success"},
        {"label": "Risk rising", "value": int((fleet["trend"] == "rising").sum()),
         "delta": "in the last 24h", "tone": "purple"},
        {"label": "Alert threshold", "value": f"{threshold:.2f}",
         "delta": "cost-optimal", "tone": "primary"},
    ])
    st.caption(f"As of {alerts['as_of']} · {alerts['machines_monitored']} machines monitored")

    if alerts["alerts"]:
        ui.section(f"{len(alerts['alerts'])} machines need attention",
                   "Ranked by risk. The symptoms come from each machine's own SHAP "
                   "contributions, translated into the language of the shop floor.")
        for alert in alerts["alerts"]:
            trend_mark = {"rising": "▲", "falling": "▼", "stable": "▬"}.get(alert["trend"], "")
            ui.entity_card(
                title=f"{alert['machine_id']} — {alert['failure_risk']:.1%} risk {trend_mark}",
                meta=(f"{alert['machine_type']} · {alert['criticality']} criticality · "
                      f"24h change {alert['risk_change_24h']:+.3f} · {alert['trend']}"),
                bullets=[s.strip() for s in str(alert["symptoms"]).split(";")],
                action=alert["recommended_action"],
                tone=alert["risk_band"],
            )
    else:
        st.success("No machines above the alert threshold.")

    ui.section("Whole fleet at a glance",
               "Every machine ranked by current failure risk. Bar colour is the risk "
               "band; hover for the machine's symptoms.")

    ranked = fleet.sort_values("failure_risk", ascending=False)
    figure = go.Figure(go.Bar(
        x=ranked["machine_id"], y=ranked["failure_risk"],
        marker=dict(color=[ui.RISK_TONES.get(b, ui.PRIMARY) for b in ranked["risk_band"]]),
        customdata=np.stack([ranked["machine_type"], ranked["risk_band"],
                             ranked["trend"], ranked["symptoms"].astype(str)], axis=-1),
        hovertemplate=("<b>%{x}</b><br>risk %{y:.1%}<br>%{customdata[0]}"
                       "<br>%{customdata[1]} · %{customdata[2]}"
                       "<br><i>%{customdata[3]}</i><extra></extra>"),
    ))
    figure.add_hline(y=threshold, line=dict(color=ui.DANGER, dash="dash", width=2),
                     annotation_text=f"alert threshold {threshold:.2f}")
    figure.update_layout(title="Current failure risk by machine",
                         yaxis=dict(title="P(failure within 24h)", tickformat=".0%"),
                         xaxis=dict(title="", tickangle=-90, tickfont=dict(size=8)))
    ui.chart(figure, height=420)

    ui.section("Filter the fleet")
    filters = st.columns([1.4, 1.2, 1, 1])
    types = filters[0].multiselect("Machine type", sorted(fleet["machine_type"].unique()),
                                   default=sorted(fleet["machine_type"].unique()))
    crits = filters[1].multiselect("Criticality", sorted(fleet["criticality"].unique()),
                                   default=sorted(fleet["criticality"].unique()))
    only_rising = filters[2].checkbox("Rising only")
    min_risk = filters[3].slider("Min risk", 0.0, 1.0, 0.0, 0.05)

    view = fleet[fleet["machine_type"].isin(types) & fleet["criticality"].isin(crits)
                 & (fleet["failure_risk"] >= min_risk)]
    if only_rising:
        view = view[view["trend"] == "rising"]

    st.dataframe(
        view[["machine_id", "machine_type", "criticality", "failure_risk", "risk_band",
              "risk_change_24h", "trend", "hours_since_maintenance",
              "sensors_abnormal_2sigma", "symptoms"]].sort_values("failure_risk",
                                                                  ascending=False),
        width="stretch", hide_index=True, height=340,
        column_config={
            "failure_risk": st.column_config.ProgressColumn(
                "Failure risk", min_value=0.0, max_value=1.0, format="%.1f%%"),
            "risk_change_24h": st.column_config.NumberColumn("24h Δ", format="%+.3f"),
            "hours_since_maintenance": st.column_config.NumberColumn(
                "Since service", format="%d h"),
            "sensors_abnormal_2sigma": st.column_config.NumberColumn("Abnormal sensors"),
        },
    )


# ======================================================================================
# MACHINE
# ======================================================================================
with machine_tab:
    ui.section("Machine inspection",
               "The sensor evidence behind a machine's score. Shaded bands are the true "
               "24-hour pre-failure windows — the periods the model is meant to flag.")

    picker = st.columns([2, 1.4, 1])
    ordered = fleet.sort_values("failure_risk", ascending=False)["machine_id"].tolist()
    chosen = picker[0].selectbox(
        "Machine", ordered,
        format_func=lambda m: (
            f"{m} — {fleet.loc[fleet['machine_id'] == m, 'failure_risk'].iloc[0]:.1%} "
            f"({fleet.loc[fleet['machine_id'] == m, 'risk_band'].iloc[0]})"),
    )
    sensor_choice = picker[1].multiselect(
        "Sensors", SENSORS, default=["vibration_rms", "temperature_c", "oil_quality_idx"],
        format_func=lambda s: SENSOR_LABELS[s],
    )
    window_days = picker[2].slider("Days of history", 3, 90, 21)

    record = fleet[fleet["machine_id"] == chosen].iloc[0]
    telemetry = load_telemetry()
    rows = telemetry[telemetry["machine_id"] == chosen].sort_values("timestamp")
    cutoff_time = rows["timestamp"].max() - pd.Timedelta(days=window_days)
    rows = rows[rows["timestamp"] >= cutoff_time]

    summary_left, summary_right = st.columns([1, 2.4])
    with summary_left:
        ui.chart(ui.gauge(float(record["failure_risk"]) * 100, "Failure risk (24h)",
                          threshold=threshold * 100, tone=record["risk_band"]), height=210)
        st.markdown(
            ui.badge(record["risk_band"], record["risk_band"]) + " " +
            ui.badge(record["trend"], "muted") + " " +
            ui.badge(f"{record['criticality']} criticality", "purple"),
            unsafe_allow_html=True)
    with summary_right:
        ui.entity_card(
            title=f"{record['machine_id']} — {record['machine_type']}",
            meta=(f"{record['age_years']:.1f} years old · "
                  f"{record['hours_since_maintenance']:.0f} h since service · "
                  f"{int(record['sensors_abnormal_2sigma'])} sensors beyond 2σ · "
                  f"24h change {record['risk_change_24h']:+.3f}"),
            bullets=[s.strip() for s in str(record["symptoms"]).split(";")],
            tone=record["risk_band"],
        )

    risk_trace = machine_risk(chosen)
    risk_trace = risk_trace[risk_trace["timestamp"] >= cutoff_time]

    if sensor_choice:
        figure = make_subplots(
            rows=len(sensor_choice) + 1, cols=1, shared_xaxes=True, vertical_spacing=0.035,
            subplot_titles=["Model risk score"] + [SENSOR_LABELS[s] for s in sensor_choice],
        )

        figure.add_trace(go.Scatter(
            x=risk_trace["timestamp"], y=risk_trace["risk"], name="Risk",
            line=dict(color=ui.DANGER, width=2), fill="tozeroy",
            fillcolor="rgba(224,85,106,0.15)",
            hovertemplate="%{x|%d %b %H:%M}<br>risk %{y:.1%}<extra></extra>"), row=1, col=1)
        figure.add_hline(y=threshold, line=dict(color=ui.WARNING, dash="dash", width=1.5),
                         row=1, col=1)

        for index, sensor in enumerate(sensor_choice, start=2):
            figure.add_trace(go.Scatter(
                x=rows["timestamp"], y=rows[sensor], name=SENSOR_LABELS[sensor],
                line=dict(color=ui.SERIES[index % len(ui.SERIES)], width=1.4),
                hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f}<extra></extra>",
            ), row=index, col=1)

        # Shade the true pre-failure windows across every subplot.
        windows = rows[rows["label"] == 1]
        if len(windows):
            breaks = windows["timestamp"].diff() > pd.Timedelta(hours=2)
            for _, group in windows.groupby(breaks.cumsum()):
                figure.add_vrect(
                    x0=group["timestamp"].min(), x1=group["timestamp"].max(),
                    fillcolor="rgba(224,85,106,0.16)", line_width=0, layer="below",
                )

        figure.update_layout(height=180 * (len(sensor_choice) + 1), showlegend=False,
                             title=f"{chosen} — last {window_days} days")
        figure.update_annotations(font=dict(size=12, color=ui.MUTED))
        ui.chart(figure)

        if len(windows):
            ui.insight(
                f"<b>{len(windows)}</b> hours in this window were inside a true "
                "pre-failure period (shaded red). Compare the risk trace at the top "
                "with the sensor traces below: the score should climb before the shading "
                "starts, not during it.", "warn")
        else:
            ui.insight("No true failure windows in this period — the machine was healthy "
                       "throughout, so the risk trace is the model's false-alarm "
                       "behaviour on a good machine.", "good")
    else:
        st.info("Pick at least one sensor.")


# ======================================================================================
# DETECTION
# ======================================================================================
with detect_tab:
    ui.kpi_row([
        {"label": "Failures in test", "value": detection["failures_in_test"],
         "tone": "primary"},
        {"label": "Caught", "value": detection["failures_detected"],
         "delta": f"{detection['event_level_recall']:.1%} event-level recall",
         "tone": "success"},
        {"label": "Median warning", "value": f"{detection['median_lead_time_hours']:.0f} h",
         "tone": "teal"},
        {"label": "Mean warning", "value": f"{detection['mean_lead_time_hours']:.1f} h",
         "tone": "warning"},
    ])

    ui.section("Detection rate hides what matters",
               "Sudden-onset failures are caught at almost the same rate as gradual ones "
               "— but the warning they give is an order of magnitude shorter. This is the "
               "central finding of the project.")

    suddenness = pd.DataFrame(detection["by_suddenness"])
    suddenness["group"] = suddenness["is_sudden"].map({True: "Sudden onset (2–8 h)",
                                                       False: "Gradual onset (48–220 h)"})

    compare = make_subplots(rows=1, cols=2,
                            subplot_titles=["Detection rate — nearly identical",
                                            "Mean lead time — an order of magnitude apart"])
    compare.add_trace(go.Bar(
        x=suddenness["group"], y=suddenness["detection_rate"],
        marker_color=[ui.SUCCESS, ui.DANGER][: len(suddenness)],
        text=[f"{r:.1%}" for r in suddenness["detection_rate"]], textposition="outside",
        hovertemplate="%{x}<br>%{y:.1%}<extra></extra>", showlegend=False), row=1, col=1)
    compare.add_trace(go.Bar(
        x=suddenness["group"], y=suddenness["mean_lead_time_hours"],
        marker_color=[ui.SUCCESS, ui.DANGER][: len(suddenness)],
        text=[f"{h:.1f} h" for h in suddenness["mean_lead_time_hours"]],
        textposition="outside",
        hovertemplate="%{x}<br>%{y:.1f} hours<extra></extra>", showlegend=False), row=1, col=2)
    compare.update_yaxes(tickformat=".0%", range=[0, 1.15], row=1, col=1)
    compare.update_yaxes(range=[0, 30], title_text="hours", row=1, col=2)
    ui.chart(compare, height=380)

    gradual = suddenness[~suddenness["is_sudden"]]
    sudden = suddenness[suddenness["is_sudden"]]
    if len(gradual) and len(sudden):
        ui.insight(
            f"Gradual failures: <b>{gradual['detection_rate'].iloc[0]:.1%}</b> caught with "
            f"<b>{gradual['mean_lead_time_hours'].iloc[0]:.1f} h</b> of warning. "
            f"Sudden: <b>{sudden['detection_rate'].iloc[0]:.1%}</b> caught with only "
            f"<b>{sudden['mean_lead_time_hours'].iloc[0]:.1f} h</b>. "
            "The alert window runs right up to the failure, so a model that spots a sudden "
            "fault two hours out still scores as having caught it. "
            "<b>Recall alone would have hidden this completely.</b>", "warn")

    ui.section("Lead time, failure by failure",
               "Each point is one real failure. Height is how much warning the model gave.")
    caught = detections[detections["detected"]].copy()
    caught["group"] = caught["is_sudden"].map({True: "sudden", False: "gradual"})
    scatter = go.Figure()
    for group, colour in [("gradual", ui.SUCCESS), ("sudden", ui.DANGER)]:
        subset = caught[caught["group"] == group]
        scatter.add_trace(go.Box(
            y=subset["lead_time_hours"], name=group, marker_color=colour,
            boxpoints="all", jitter=0.45, pointpos=0, marker=dict(size=6, opacity=0.65),
            customdata=subset[["machine_id", "failure_mode"]].to_numpy(),
            hovertemplate=("%{customdata[0]} · %{customdata[1]}"
                           "<br>%{y:.0f} h warning<extra></extra>")))
    missed = len(detections) - len(caught)
    scatter.update_layout(title=f"Warning lead time of every caught failure "
                                f"({len(caught)} caught, {missed} missed)",
                          yaxis_title="Hours of warning before failure")
    ui.chart(scatter, height=400)

    ui.section("Detection by failure mode")
    modes = pd.DataFrame(detection["by_failure_mode"])
    mode_fig = go.Figure()
    mode_fig.add_trace(go.Bar(
        x=modes["failure_mode"], y=modes["detection_rate"], name="Detection rate",
        marker_color=ui.PRIMARY, text=[f"{r:.0%}" for r in modes["detection_rate"]],
        textposition="outside",
        customdata=modes[["detected", "failures"]].to_numpy(),
        hovertemplate="%{x}<br>%{customdata[0]}/%{customdata[1]} caught<extra></extra>"))
    mode_fig.add_trace(go.Scatter(
        x=modes["failure_mode"], y=modes["mean_lead_time_hours"], name="Mean lead time",
        yaxis="y2", mode="lines+markers", line=dict(color=ui.WARNING, width=3),
        marker=dict(size=10), hovertemplate="%{x}<br>%{y:.1f} h<extra></extra>"))
    mode_fig.update_layout(
        yaxis=dict(title="Detection rate", tickformat=".0%", range=[0, 1.15]),
        yaxis2=dict(title="Mean lead (h)", overlaying="y", side="right", showgrid=False,
                    range=[0, 30]),
        title="Every failure mode, caught and how early")
    ui.chart(mode_fig, height=400)

    ui.insight(
        "Bearing wear and lubrication failure are caught completely — both drive "
        "vibration, the sensor with the largest relative deviation. Overheating is "
        "weakest, consistent with temperature being the one sensor that also tracks "
        "plant ambient, so a real thermal fault has to clear a noisier baseline.")

    with st.expander("Every failure in the test period"):
        st.dataframe(detections, width="stretch", hide_index=True,
                     column_config={"lead_time_hours": st.column_config.NumberColumn(
                         "Lead time", format="%.0f h")})


# ======================================================================================
# MODEL
# ======================================================================================
with model_tab:
    ui.section("Four candidates on the held-out period",
               "The split is chronological: every training row precedes every test row. "
               "Curves below are scored live from the saved models.")

    # Each metrics entry already carries its own "model" field, so reset_index(
    # names="model") collides with it. Take the name from the index instead.
    comparison = pd.DataFrame(evaluation["model_comparison"]).T
    comparison["model"] = comparison.index
    comparison = comparison.reset_index(drop=True)

    bar = go.Figure()
    for metric, colour in [("pr_auc", ui.PRIMARY), ("roc_auc", ui.MUTED)]:
        bar.add_trace(go.Bar(
            x=comparison["model"], y=comparison[metric], name=metric.upper().replace("_", "-"),
            marker_color=colour, text=[f"{v:.3f}" for v in comparison[metric]],
            textposition="outside", hovertemplate="%{x}<br>%{y:.4f}<extra></extra>"))
    bar.update_layout(barmode="group", title="PR-AUC separates the models; ROC-AUC does not",
                      yaxis=dict(range=[0, 1.1]))
    ui.chart(bar, height=380)

    spread_pr = comparison["pr_auc"].max() - comparison["pr_auc"].min()
    spread_roc = comparison["roc_auc"].max() - comparison["roc_auc"].min()
    ui.insight(
        f"Every model scores above <b>{comparison['roc_auc'].min():.2f}</b> ROC-AUC, a "
        f"spread of just {spread_roc:.3f} — which makes them look interchangeable. On "
        f"PR-AUC the spread is <b>{spread_pr:.3f}</b>. With 5.2% positives, ROC-AUC is "
        "dominated by the easy negatives and flatters everything.", "warn")

    if st.toggle("Compute live ROC / PR curves (scores 48,598 rows)", value=False):
        from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                     roc_auc_score, roc_curve)

        y_true, model_scores, cutoff = holdout_scores()
        st.caption(f"Chronological split at {cutoff}")

        roc_fig, pr_fig = go.Figure(), go.Figure()
        for index, (name, probs) in enumerate(model_scores.items()):
            colour = ui.SERIES[index % len(ui.SERIES)]
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_fig.add_trace(go.Scatter(
                x=fpr, y=tpr, name=f"{name} ({roc_auc_score(y_true, probs):.3f})",
                line=dict(color=colour, width=2.5),
                hovertemplate=f"{name}<br>FPR %{{x:.3f}}<br>TPR %{{y:.3f}}<extra></extra>"))
            precision, recall, _ = precision_recall_curve(y_true, probs)
            pr_fig.add_trace(go.Scatter(
                x=recall, y=precision,
                name=f"{name} ({average_precision_score(y_true, probs):.3f})",
                line=dict(color=colour, width=2.5),
                hovertemplate=f"{name}<br>R %{{x:.3f}}<br>P %{{y:.3f}}<extra></extra>"))

        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                     line=dict(color=ui.MUTED, dash="dash")))
        roc_fig.update_layout(title="ROC", xaxis_title="False positive rate",
                              yaxis_title="True positive rate")
        pr_fig.add_hline(y=float(np.mean(y_true)), line=dict(color=ui.MUTED, dash="dash"),
                         annotation_text=f"base rate {np.mean(y_true):.3f}")
        pr_fig.update_layout(title="Precision–Recall", xaxis_title="Recall",
                             yaxis_title="Precision")

        curve_cols = st.columns(2)
        with curve_cols[0]:
            ui.chart(roc_fig, height=400)
        with curve_cols[1]:
            ui.chart(pr_fig, height=400)

    ui.section("Calibration", "Weighted tree ensembles are confident and wrong. Isotonic "
                              "calibration barely moves the ranking but transforms the "
                              "probabilities — which is what the cost model consumes.")
    calib = comparison[["model", "brier", "ece"]].sort_values("ece")
    calib_fig = go.Figure()
    calib_fig.add_trace(go.Bar(x=calib["model"], y=calib["ece"], name="Calibration error",
                               marker_color=ui.WARNING,
                               text=[f"{v:.3f}" for v in calib["ece"]],
                               textposition="outside"))
    calib_fig.add_trace(go.Bar(x=calib["model"], y=calib["brier"], name="Brier score",
                               marker_color=ui.TEAL,
                               text=[f"{v:.3f}" for v in calib["brier"]],
                               textposition="outside"))
    calib_fig.update_layout(barmode="group", title="Lower is better")
    ui.chart(calib_fig, height=360)

    if evaluation.get("shap_importance"):
        ui.section("What the model is using",
                   "Mean |SHAP| over 2,000 sampled test rows.")
        shap_rows = pd.DataFrame(evaluation["shap_importance"]).head(15)
        ui.chart(ui.horizontal_bars(shap_rows["feature"].tolist(),
                                    shap_rows["mean_abs_shap"].tolist(),
                                    "Top risk drivers", ui.PURPLE))
        ui.insight(
            "The top two are <b>maintenance-schedule</b> features, not sensor features. "
            "That is partly an artefact of the simulator — the wear clock resets after "
            "every repair — and on real telemetry it would need auditing, because "
            "<code>hours_since_maintenance</code> can become a proxy for when a work "
            "order was closed rather than for machine condition.", "warn")


# ======================================================================================
# ECONOMICS
# ======================================================================================
with econ_tab:
    ui.section("What threshold should we actually run at?",
               "Nothing here defaults to 0.5. Set your own costs and the optimum moves.")

    base = evaluation["cost_assumptions"]
    inputs = st.columns(4)
    failure_cost = inputs[0].slider("Unplanned failure cost", 2_000, 60_000,
                                    int(base["unplanned_failure_cost"]), 1_000)
    inspection_cost = inputs[1].slider("Planned inspection cost", 50, 3_000,
                                       int(base["planned_inspection_cost"]), 50)
    prevention = inputs[2].slider("Failures averted when caught", 0.2, 1.0,
                                  float(base["inspection_prevention_rate"]), 0.05)
    cooldown = inputs[3].select_slider("Alert cooldown (fixed)",
                                       options=[int(base["alert_cooldown_hours"])],
                                       value=int(base["alert_cooldown_hours"]))

    cost = pd.DataFrame(evaluation["cost_analysis"]).copy()
    total_failures = detection["failures_in_test"]
    cost["detected"] = cost["event_level_recall"] * total_failures
    cost["missed"] = total_failures - cost["detected"]
    cost["inspection_total"] = cost["alert_episodes"] * inspection_cost
    cost["failure_total"] = (cost["missed"]
                             + cost["detected"] * (1 - prevention)) * failure_cost
    cost["total"] = cost["inspection_total"] + cost["failure_total"]
    cost["saving"] = total_failures * failure_cost - cost["total"]

    optimum = cost.loc[cost["total"].idxmin()]
    moved = abs(float(optimum["threshold"]) - threshold) > 1e-9

    ui.kpi_row([
        {"label": "Optimal threshold", "value": f"{optimum['threshold']:.2f}",
         "delta": "moved from 0.15" if moved else "as reported", "tone": "primary"},
        {"label": "Failures caught", "value": f"{optimum['event_level_recall']:.1%}",
         "tone": "success"},
        {"label": "Alerts / machine / week",
         "value": f"{optimum['alerts_per_machine_per_week']:.2f}",
         "delta": f"{int(optimum['alert_episodes'])} episodes", "tone": "warning"},
        {"label": "Saving vs. reactive", "value": f"{optimum['saving']:,.0f}",
         "tone": "success" if optimum["saving"] > 0 else "danger"},
    ])

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=cost["threshold"], y=cost["failure_total"],
                                name="Cost of missed failures", stackgroup="one",
                                line=dict(color=ui.DANGER, width=0.5),
                                fillcolor="rgba(224,85,106,0.45)",
                                hovertemplate="%{x:.2f}<br>%{y:,.0f}<extra></extra>"))
    figure.add_trace(go.Scatter(x=cost["threshold"], y=cost["inspection_total"],
                                name="Cost of inspections", stackgroup="one",
                                line=dict(color=ui.TEAL, width=0.5),
                                fillcolor="rgba(56,178,196,0.45)",
                                hovertemplate="%{x:.2f}<br>%{y:,.0f}<extra></extra>"))
    figure.add_trace(go.Scatter(x=cost["threshold"], y=cost["total"], name="Total",
                                line=dict(color=ui.PRIMARY, width=3),
                                hovertemplate="%{x:.2f}<br>total %{y:,.0f}<extra></extra>"))
    figure.add_vline(x=float(optimum["threshold"]),
                     line=dict(color=ui.WARNING, dash="dash", width=2),
                     annotation_text=f"optimum {optimum['threshold']:.2f}")
    figure.update_layout(title="Total expected cost decomposed",
                         xaxis_title="Alert threshold", yaxis_title="Expected cost")
    ui.chart(figure, height=420)

    ui.insight(
        f"At these assumptions a missed failure costs "
        f"<b>{failure_cost / inspection_cost:.0f}×</b> an inspection, so the model should "
        f"deliberately over-refer — which is why the optimum sits at "
        f"<b>{optimum['threshold']:.2f}</b> rather than 0.5. Raise the inspection cost "
        "and watch it climb.")

    ui.section("The trade-off a planner actually feels",
               "Every extra failure caught costs more alarms chased. Hover any point for "
               "its threshold.")
    trade = go.Figure(go.Scatter(
        x=cost["alerts_per_machine_per_week"], y=cost["event_level_recall"],
        mode="lines+markers", line=dict(color=ui.PURPLE, width=3),
        marker=dict(size=9, color=cost["total"], colorscale="RdYlGn_r",
                    showscale=True, colorbar=dict(title="Cost", thickness=12)),
        customdata=cost[["threshold", "total", "mean_lead_time_hours"]].to_numpy(),
        hovertemplate=("threshold %{customdata[0]:.2f}<br>%{x:.2f} alerts/machine/week"
                       "<br>%{y:.1%} of failures caught"
                       "<br>%{customdata[2]:.1f} h mean warning"
                       "<br>cost %{customdata[1]:,.0f}<extra></extra>")))
    trade.update_layout(title="Failures caught vs. alarms chased",
                        xaxis_title="Alert episodes per machine per week",
                        yaxis_title="Event-level recall", yaxis=dict(tickformat=".0%"))
    ui.chart(trade, height=400)

    ui.section("Operating points at fixed recall")
    st.dataframe(pd.DataFrame(evaluation["recall_targets"]), width="stretch",
                 hide_index=True,
                 column_config={
                     "event_level_recall": st.column_config.ProgressColumn(
                         "Failures caught", min_value=0.0, max_value=1.0, format="%.1f%%"),
                     "mean_lead_time_hours": st.column_config.NumberColumn(
                         "Mean lead", format="%.1f h"),
                     "alerts_per_machine_per_week": st.column_config.NumberColumn(
                         "Alerts/machine/week", format="%.2f"),
                 })
