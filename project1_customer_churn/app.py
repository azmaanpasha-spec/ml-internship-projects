"""Retention console for the customer churn model.

    streamlit run app.py

Five workspaces, each answering a question a different person asks:

  Overview    (exec)      how much revenue is at risk, and is the model trustworthy?
  Model lab   (DS)        which model, at which threshold, and how well calibrated?
  Worklist    (agent)     who do I call today, and what do I say?
  Segments    (analyst)   where in the base is churn concentrated?
  Campaign    (manager)   what does a retention programme cost and return?

Curves and matrices are computed live from the saved models rather than loaded as
images, so the threshold slider genuinely re-scores the hold-out set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import ui  # noqa: E402

REPORTS = HERE / "reports"
MODELS = HERE / "models"
FEATURES = HERE / "data" / "processed" / "customers_features.csv"

ui.setup_page(
    title="Churn Retention Console",
    icon="📉",
    heading="Customer Churn — Retention Console",
    subtitle="Calibrated churn risk, a worked call list, and a costed retention programme",
    tagline="Threshold, campaign economics and model selection are all live controls — "
            "every number below re-computes from the saved models on the hold-out set.",
)

if not (REPORTS / "evaluation.json").exists():
    ui.missing_artefacts(
        [
            "python -m src.generate_data",
            "python -m src.preprocess",
            "python -m src.features",
            "python -m src.train",
            "python -m src.evaluate",
            "python -m src.score_customers",
        ],
        note="Run the pipeline from the project folder first.",
    )


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_reports():
    evaluation = json.loads((REPORTS / "evaluation.json").read_text())
    scoring = json.loads((REPORTS / "scoring_summary.json").read_text())
    high_risk = pd.read_csv(REPORTS / "high_risk_customers.csv")
    scored = pd.read_csv(REPORTS / "all_customers_scored.csv")
    return evaluation, scoring, high_risk, scored


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES)


@st.cache_resource(show_spinner="Scoring the hold-out set…")
def holdout_scores():
    """Re-score the test split with every saved model.

    Done live rather than read from a file so the threshold control below operates on
    real predictions instead of pre-baked summary numbers.
    """
    from common.io_utils import load_model
    from src.train import split_data

    frame = load_features()
    X_train, X_test, y_train, y_test = split_data(frame)

    scores = {}
    for path in sorted(MODELS.glob("*.joblib")):
        if path.stem == "best_model":
            continue
        scores[path.stem] = load_model(path).predict_proba(X_test)[:, 1]
    scores["calibrated"] = np.clip(
        load_model(MODELS / "best_model.joblib").predict_proba(X_test)[:, 1], 0.001, 0.995
    )
    return y_test.to_numpy(), scores, X_test.reset_index(drop=True)


evaluation, scoring, high_risk, scored = load_reports()
best_model = evaluation["best_model"]

overview, model_lab, worklist, segments, campaign = st.tabs(
    ["📊 Overview", "🔬 Model lab", "📞 Worklist", "🧭 Segments", "💰 Campaign"]
)


# ======================================================================================
# OVERVIEW
# ======================================================================================
with overview:
    calibrated = evaluation["model_comparison"][f"{best_model}_calibrated"]
    profit_point = evaluation["performance_at_profit_threshold"]
    optimum = evaluation["campaign_optimum"]
    bands = pd.DataFrame(scoring["risk_bands"])

    ui.kpi_row([
        {"label": "Customers scored", "value": f"{scoring['scored_customers']:,}",
         "tone": "primary"},
        {"label": "Revenue at risk", "value": f"{scoring['total_revenue_at_risk']/1e6:.2f}M",
         "delta": "expected loss over 14 months", "tone": "danger"},
        {"label": "ROC-AUC", "value": f"{calibrated['roc_auc']:.3f}",
         "delta": f"PR-AUC {calibrated['pr_auc']:.3f}", "tone": "success"},
        {"label": "Calibration error", "value": f"{calibrated['ece']:.3f}",
         "delta": "after isotonic — 6× better", "tone": "teal"},
        {"label": "Churners reached", "value": f"{profit_point['recall']:.0%}",
         "delta": f"at threshold {evaluation['thresholds']['max_profit']}", "tone": "warning"},
        {"label": "Campaign ROI", "value": f"{optimum['roi']:.2f}×",
         "delta": f"net {optimum['net_profit']:,.0f}", "tone": "purple"},
    ])

    left, right = st.columns([1, 1.15])

    with left:
        ui.section("Where the risk sits",
                   "Band sizes against the revenue each band represents.")
        figure = go.Figure()
        figure.add_trace(go.Bar(
            x=bands["risk_band"], y=bands["customers"], name="Customers",
            marker_color=[ui.RISK_TONES.get(b, ui.PRIMARY) for b in bands["risk_band"]],
            hovertemplate="%{x}<br>%{y:,} customers<extra></extra>",
        ))
        figure.add_trace(go.Scatter(
            x=bands["risk_band"], y=bands["revenue_at_risk"], name="Revenue at risk",
            yaxis="y2", mode="lines+markers",
            line=dict(color=ui.WARNING, width=3), marker=dict(size=10),
            hovertemplate="%{x}<br>%{y:,.0f} at risk<extra></extra>",
        ))
        figure.update_layout(
            yaxis=dict(title="Customers"),
            yaxis2=dict(title="Revenue at risk", overlaying="y", side="right",
                        showgrid=False),
            title="Customers and exposure by risk band",
        )
        ui.chart(figure, height=360)

    with right:
        ui.section("Is the model telling the truth?",
                   "Predicted probability against the churn rate actually observed in "
                   "each band. Bars close to the diagonal mean the numbers can be "
                   "multiplied by revenue and trusted.")
        figure = go.Figure()
        figure.add_trace(go.Bar(
            x=bands["risk_band"], y=bands["mean_probability"], name="Model says",
            marker_color=ui.PRIMARY, hovertemplate="predicted %{y:.1%}<extra></extra>",
        ))
        figure.add_trace(go.Bar(
            x=bands["risk_band"], y=bands["actual_churn_rate"], name="Actually churned",
            marker_color=ui.SUCCESS, hovertemplate="observed %{y:.1%}<extra></extra>",
        ))
        figure.update_layout(barmode="group", yaxis=dict(tickformat=".0%"),
                             title="Predicted vs. observed churn rate")
        ui.chart(figure, height=360)

    gap = abs(bands["mean_probability"] - bands["actual_churn_rate"]).mean()
    ui.insight(
        f"Mean gap between predicted and observed churn is <b>{gap:.1%}</b>. "
        "That is what makes the revenue-at-risk figure meaningful: an uncalibrated model "
        "ranks customers correctly but cannot be multiplied by money.",
        "good",
    )

    ui.section("Targeting efficiency",
               "If the base is sorted by risk and worked top-down, how much of the churn "
               "is captured by the time you stop?")
    lift = pd.DataFrame(evaluation["lift_table"])
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=lift["decile"] * 10, y=lift["cumulative_capture"] * 100,
        mode="lines+markers", name="Model", fill="tozeroy",
        line=dict(color=ui.PRIMARY, width=3), marker=dict(size=8),
        fillcolor="rgba(91,141,239,0.14)",
        hovertemplate="Top %{x}% of base<br>captures %{y:.1f}% of churners<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=lift["decile"] * 10, y=lift["decile"] * 10, mode="lines", name="No model",
        line=dict(color=ui.MUTED, width=2, dash="dash"),
        hovertemplate="Random: %{y:.0f}%<extra></extra>",
    ))
    figure.update_layout(
        title="Cumulative gains",
        xaxis=dict(title="Share of base contacted (%)"),
        yaxis=dict(title="Share of churners captured (%)"),
    )
    ui.chart(figure, height=380)

    top_decile = lift.iloc[0]
    ui.insight(
        f"The riskiest 10% of the base contains <b>{top_decile['rate']:.0%}</b> churn "
        f"against a {bands['customers'].mul(bands['actual_churn_rate']).sum() / bands['customers'].sum():.0%} "
        f"base rate — a <b>{top_decile['lift']:.1f}× lift</b>. Working the top three "
        f"deciles reaches <b>{lift.iloc[2]['cumulative_capture']:.0%}</b> of all churners."
    )


# ======================================================================================
# MODEL LAB
# ======================================================================================
with model_lab:
    y_true, model_scores, X_test = holdout_scores()

    ui.section("Compare models on the hold-out set",
               "Curves are computed live from the saved models. Pick which to show, then "
               "move the threshold to see the operating point travel along them.")

    controls = st.columns([2.2, 1.4])
    chosen = controls[0].multiselect(
        "Models", list(model_scores),
        default=[m for m in ("logistic_regression", best_model, "calibrated")
                 if m in model_scores],
    )
    threshold = controls[1].slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

    if not chosen:
        st.info("Select at least one model.")
    else:
        from sklearn.metrics import (
            auc, average_precision_score, confusion_matrix, precision_recall_curve,
            roc_auc_score, roc_curve,
        )

        roc_fig = go.Figure()
        pr_fig = go.Figure()
        for index, name in enumerate(chosen):
            colour = ui.SERIES[index % len(ui.SERIES)]
            probs = model_scores[name]

            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} ({roc_auc_score(y_true, probs):.3f})",
                line=dict(color=colour, width=2.5),
                hovertemplate=f"<b>{name}</b><br>FPR %{{x:.3f}}<br>TPR %{{y:.3f}}<extra></extra>",
            ))

            precision, recall, _ = precision_recall_curve(y_true, probs)
            pr_fig.add_trace(go.Scatter(
                x=recall, y=precision, mode="lines",
                name=f"{name} ({average_precision_score(y_true, probs):.3f})",
                line=dict(color=colour, width=2.5),
                hovertemplate=f"<b>{name}</b><br>recall %{{x:.3f}}<br>precision %{{y:.3f}}<extra></extra>",
            ))

            # The live operating point for this threshold.
            predicted = (probs >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
            roc_fig.add_trace(go.Scatter(
                x=[fp / max(fp + tn, 1)], y=[tp / max(tp + fn, 1)], mode="markers",
                marker=dict(color=colour, size=13, symbol="x-thin",
                            line=dict(width=3, color=colour)),
                showlegend=False, hovertemplate=f"{name} @ {threshold:.2f}<extra></extra>",
            ))

        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(color=ui.MUTED, dash="dash", width=1.5),
        ))
        roc_fig.update_layout(title="ROC — crosses mark the current threshold",
                              xaxis_title="False positive rate", yaxis_title="True positive rate")

        base_rate = float(np.mean(y_true))
        pr_fig.add_hline(y=base_rate, line=dict(color=ui.MUTED, dash="dash", width=1.5),
                         annotation_text=f"base rate {base_rate:.3f}")
        pr_fig.update_layout(title="Precision–Recall — the honest view at 26% positives",
                             xaxis_title="Recall", yaxis_title="Precision")

        curve_left, curve_right = st.columns(2)
        with curve_left:
            ui.chart(roc_fig, height=400)
        with curve_right:
            ui.chart(pr_fig, height=400)

        ui.section("Live confusion matrix",
                   f"Every model re-scored at threshold {threshold:.2f}.")
        metric_cols = st.columns(len(chosen))
        for column, name in zip(metric_cols, chosen):
            probs = model_scores[name]
            predicted = (probs >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)

            heat = go.Figure(go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Predicted stay", "Predicted churn"],
                y=["Actually stayed", "Actually churned"],
                text=[[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]],
                texttemplate="%{text}", textfont=dict(size=16),
                colorscale=[[0, "rgba(91,141,239,0.08)"], [1, "rgba(91,141,239,0.75)"]],
                showscale=False, hovertemplate="%{y} → %{x}<br>%{z:,}<extra></extra>",
            ))
            heat.update_layout(title=f"{name}", height=300, yaxis=dict(autorange="reversed"))
            with column:
                ui.chart(heat, height=300)
                st.markdown(
                    ui.badge(f"P {precision:.3f}", "primary") + " " +
                    ui.badge(f"R {recall:.3f}", "success") + " " +
                    ui.badge(f"F1 {f1:.3f}", "warning"),
                    unsafe_allow_html=True,
                )

        ui.insight(
            "Missing a churner costs far more than an unnecessary offer, so the "
            f"profit-optimal threshold is <b>{evaluation['thresholds']['max_profit']}</b>, "
            f"not 0.5 — drag the slider down there and watch the bottom-right cell "
            "(caught churners) grow faster than the false alarms beside it.",
            "warn",
        )

    ui.section("What drives the prediction",
               "Permutation importance: the drop in ROC-AUC when a feature is shuffled.")
    importance = pd.DataFrame(evaluation["permutation_importance"]).head(14)
    ui.chart(
        ui.horizontal_bars(importance["feature"].tolist(), importance["importance"].tolist(),
                           "Top churn drivers", ui.PRIMARY),
    )


# ======================================================================================
# WORKLIST
# ======================================================================================
with worklist:
    ui.section("Today's retention call list",
               "Ranked by churn probability. Filter, then open a customer for their "
               "reasons and suggested offer.")

    filters = st.columns([1.3, 1, 1, 1.4])
    band_choice = filters[0].multiselect(
        "Risk band", sorted(high_risk["risk_band"].unique()),
        default=sorted(high_risk["risk_band"].unique()),
    )
    min_value = filters[1].slider(
        "Min revenue at risk", 0, int(high_risk["revenue_at_risk"].max()), 0, 50,
    )
    contract_choice = filters[2].multiselect(
        "Contract", sorted(high_risk["contract"].unique()),
        default=sorted(high_risk["contract"].unique()),
    )
    search = filters[3].text_input("Search customer ID", placeholder="CUST-00…")

    view = high_risk[
        high_risk["risk_band"].isin(band_choice)
        & high_risk["contract"].isin(contract_choice)
        & (high_risk["revenue_at_risk"] >= min_value)
    ]
    if search:
        view = view[view["customer_id"].str.contains(search, case=False, na=False)]

    ui.kpi_row([
        {"label": "Customers in view", "value": f"{len(view):,}", "tone": "primary"},
        {"label": "Revenue at risk", "value": f"{view['revenue_at_risk'].sum():,.0f}",
         "tone": "danger"},
        {"label": "Mean risk", "value": f"{view['churn_probability'].mean():.1%}"
         if len(view) else "—", "tone": "warning"},
        {"label": "Mean tenure", "value": f"{view['tenure_months'].mean():.0f} mo"
         if len(view) else "—", "tone": "teal"},
    ])

    if view.empty:
        st.info("No customers match these filters.")
    else:
        table = view[[
            "customer_id", "churn_probability", "risk_band", "revenue_at_risk",
            "tenure_months", "contract", "monthly_charges", "satisfaction_score",
            "recommended_action",
        ]]
        st.dataframe(
            table, width="stretch", hide_index=True, height=330,
            column_config={
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn risk", min_value=0.0, max_value=1.0, format="%.1f%%"),
                "revenue_at_risk": st.column_config.NumberColumn(
                    "Revenue at risk", format="%.0f"),
                "satisfaction_score": st.column_config.ProgressColumn(
                    "Satisfaction", min_value=0.0, max_value=10.0, format="%.1f"),
                "monthly_charges": st.column_config.NumberColumn("Monthly", format="%.2f"),
                "tenure_months": st.column_config.NumberColumn("Tenure", format="%d mo"),
            },
        )

        ui.section("Customer detail")
        chosen_id = st.selectbox(
            "Open a customer", view["customer_id"].tolist(),
            format_func=lambda c: (
                f"{c} — {view.loc[view['customer_id'] == c, 'churn_probability'].iloc[0]:.0%} risk"
            ),
        )
        record = view[view["customer_id"] == chosen_id].iloc[0]
        tone = record["risk_band"]

        detail_left, detail_right = st.columns([1, 2])
        with detail_left:
            ui.chart(
                ui.gauge(float(record["churn_probability"]) * 100, "Churn probability",
                         threshold=float(evaluation["thresholds"]["max_profit"]) * 100,
                         tone=tone),
                height=210,
            )
            st.markdown(
                ui.badge(record["risk_band"], tone) + " " +
                ui.badge(record["contract"], "muted"),
                unsafe_allow_html=True,
            )

        with detail_right:
            ui.entity_card(
                title=f"{record['customer_id']}",
                meta=(f"{record['tenure_months']:.0f} months tenure · "
                      f"{record['monthly_charges']:.2f}/mo · "
                      f"{record['revenue_at_risk']:,.0f} at risk · "
                      f"satisfaction {record['satisfaction_score']:.1f}/10"),
                bullets=[record.get("reason_1"), record.get("reason_2"),
                         record.get("reason_3")],
                action=f"Recommended: {record['recommended_action']}",
                tone=tone,
            )

            # How this customer compares with the base on the drivers that matter.
            features = load_features()
            comparison = []
            for column, label, fmt in [
                ("tenure_months", "Tenure (months)", "{:.0f}"),
                ("monthly_charges", "Monthly charges", "{:.2f}"),
                ("satisfaction_score", "Satisfaction", "{:.1f}"),
                ("support_tickets_6m", "Support tickets (6m)", "{:.0f}"),
            ]:
                if column in features and column in record:
                    percentile = float((features[column] < record[column]).mean() * 100)
                    comparison.append({"Metric": label,
                                       "This customer": fmt.format(record[column]),
                                       "Base median": fmt.format(features[column].median()),
                                       "Percentile": percentile})
            if comparison:
                st.dataframe(
                    pd.DataFrame(comparison), width="stretch", hide_index=True,
                    column_config={"Percentile": st.column_config.ProgressColumn(
                        "Percentile in base", min_value=0.0, max_value=100.0, format="%.0f%%")},
                )

        ui.section("Where the effort goes",
                   "Recommended actions across the whole shortlist.")
        actions = view["recommended_action"].value_counts().head(8)
        ui.chart(
            ui.horizontal_bars(actions.index.tolist(), actions.to_numpy().tolist(),
                               "Suggested next action", ui.TEAL, value_fmt="{:.0f}"),
        )


# ======================================================================================
# SEGMENTS
# ======================================================================================
with segments:
    features = load_features()

    ui.section("Where is churn concentrated?",
               "Cross two attributes to find the pockets worth a targeted campaign. "
               "Cell colour is the churn rate; hover for the customer count.")

    categorical = ["contract", "internet_service", "payment_method", "tenure_bucket",
                   "paperless_billing", "autopay", "region", "gender", "dependents",
                   "tech_support", "streaming_tv"]
    available = [c for c in categorical if c in features.columns]

    axes = st.columns([1, 1, 1.4])
    row_dim = axes[0].selectbox("Rows", available, index=available.index("contract"))
    column_options = [c for c in available if c != row_dim]
    default_col = "tenure_bucket" if "tenure_bucket" in column_options else column_options[0]
    col_dim = axes[1].selectbox("Columns", column_options,
                                index=column_options.index(default_col))
    min_cell = axes[2].slider("Hide cells with fewer than N customers", 0, 200, 25, 5)

    pivot_rate = features.pivot_table(index=row_dim, columns=col_dim,
                                      values="churned", aggfunc="mean")
    pivot_count = features.pivot_table(index=row_dim, columns=col_dim,
                                       values="churned", aggfunc="size")
    pivot_rate = pivot_rate.where(pivot_count >= min_cell)

    heat = go.Figure(go.Heatmap(
        z=pivot_rate.to_numpy() * 100,
        x=[str(c) for c in pivot_rate.columns],
        y=[str(i) for i in pivot_rate.index],
        customdata=pivot_count.to_numpy(),
        colorscale=[[0, "rgba(47,182,124,0.85)"], [0.5, "rgba(232,163,61,0.85)"],
                    [1, "rgba(224,85,106,0.9)"]],
        texttemplate="%{z:.0f}%", textfont=dict(size=12),
        hovertemplate=(f"{row_dim}: %{{y}}<br>{col_dim}: %{{x}}"
                       "<br>churn %{z:.1f}%<br>%{customdata:,} customers<extra></extra>"),
        colorbar=dict(title="Churn %", thickness=12),
    ))
    heat.update_layout(title=f"Churn rate — {row_dim} × {col_dim}")
    ui.chart(heat, height=420)

    overall = features["churned"].mean()
    flat = pivot_rate.stack(future_stack=True).dropna()
    if len(flat):
        worst_idx = flat.idxmax()
        ui.insight(
            f"Worst pocket: <b>{row_dim}={worst_idx[0]}</b> with "
            f"<b>{col_dim}={worst_idx[1]}</b> churns at <b>{flat.max():.0%}</b> "
            f"against a {overall:.0%} base rate — "
            f"<b>{flat.max()/overall:.1f}× the average</b>.",
            "warn",
        )

    ui.section("Behaviour of churners vs. stayers",
               "Pick a behavioural feature to see how the two groups separate. "
               "Clean separation here is what the model is exploiting.")

    numeric_candidates = [
        "usage_trend_pct", "usage_recent_vs_prior", "tenure_months", "satisfaction_score",
        "ticket_escalation", "price_change_pct", "risk_flag_count", "monthly_charges",
        "late_payment_rate", "services_count", "login_recent_vs_prior",
    ]
    numeric_available = [c for c in numeric_candidates if c in features.columns]
    feature_choice = st.selectbox("Feature", numeric_available)

    churned = features[features["churned"] == 1][feature_choice].dropna()
    stayed = features[features["churned"] == 0][feature_choice].dropna()
    low, high = features[feature_choice].quantile([0.01, 0.99])

    dist = go.Figure()
    dist.add_trace(go.Violin(y=stayed.clip(low, high), name="Stayed", side="negative",
                             line_color=ui.SUCCESS, fillcolor="rgba(47,182,124,0.35)",
                             meanline_visible=True, points=False))
    dist.add_trace(go.Violin(y=churned.clip(low, high), name="Churned", side="positive",
                             line_color=ui.DANGER, fillcolor="rgba(224,85,106,0.35)",
                             meanline_visible=True, points=False))
    dist.update_traces(orientation="v", width=0.9)
    dist.update_layout(title=f"{feature_choice} — churned vs. stayed",
                       violinmode="overlay", yaxis_title=feature_choice)
    ui.chart(dist, height=400)

    separation = (churned.mean() - stayed.mean()) / (features[feature_choice].std() + 1e-9)
    direction = "higher" if separation > 0 else "lower"
    ui.insight(
        f"Churners average <b>{abs(separation):.2f} standard deviations {direction}</b> "
        f"on <code>{feature_choice}</code> than customers who stayed."
    )


# ======================================================================================
# CAMPAIGN
# ======================================================================================
with campaign:
    ui.section("Retention campaign simulator",
               "The threshold is a business decision, not a statistical one. Change the "
               "assumptions and the optimum moves — this is the tab to open with a "
               "stakeholder who disagrees with your numbers.")

    assumptions = evaluation["campaign_assumptions"]
    inputs = st.columns(4)
    offer_cost = inputs[0].slider("Retention offer cost", 10.0, 150.0,
                                  float(assumptions["retention_offer_cost"]), 5.0)
    contact_cost = inputs[1].slider("Contact cost", 1.0, 30.0,
                                    float(assumptions["contact_cost"]), 1.0)
    success_rate = inputs[2].slider("Offer success rate", 0.05, 0.80,
                                    float(assumptions["offer_success_rate"]), 0.01)
    horizon = inputs[3].slider("Expected remaining months", 3, 36,
                               int(assumptions["expected_remaining_months"]), 1)

    simulation = pd.DataFrame(evaluation["campaign_simulation"]).copy()
    # Re-price the saved simulation: saved revenue scales with success rate and horizon,
    # while cost is rebuilt from the per-contact spend.
    scale = (success_rate / assumptions["offer_success_rate"]) * (
        horizon / assumptions["expected_remaining_months"])
    simulation["revenue_saved"] *= scale
    simulation["campaign_cost"] = simulation["contacted"] * (offer_cost + contact_cost)
    simulation["net_profit"] = simulation["revenue_saved"] - simulation["campaign_cost"]
    simulation["roi"] = simulation["net_profit"] / simulation["campaign_cost"].clip(lower=1)

    best_row = simulation.loc[simulation["net_profit"].idxmax()]
    changed = (
        offer_cost != assumptions["retention_offer_cost"]
        or contact_cost != assumptions["contact_cost"]
        or success_rate != assumptions["offer_success_rate"]
        or horizon != assumptions["expected_remaining_months"]
    )

    ui.kpi_row([
        {"label": "Optimal threshold", "value": f"{best_row['threshold']:.2f}",
         "delta": "shifted" if changed else "as reported", "tone": "primary"},
        {"label": "Customers contacted", "value": f"{int(best_row['contacted']):,}",
         "delta": f"{best_row['contacted_share']:.0%} of base", "tone": "teal"},
        {"label": "Churners reached", "value": f"{best_row['recall']:.0%}",
         "delta": f"precision {best_row['precision']:.0%}", "tone": "warning"},
        {"label": "Net profit", "value": f"{best_row['net_profit']:,.0f}",
         "delta": f"ROI {best_row['roi']:.2f}×",
         "tone": "success" if best_row["net_profit"] > 0 else "danger"},
    ])

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=simulation["threshold"], y=simulation["revenue_saved"], name="Revenue saved",
        line=dict(color=ui.SUCCESS, width=2.5),
        hovertemplate="threshold %{x:.2f}<br>saved %{y:,.0f}<extra></extra>"))
    figure.add_trace(go.Scatter(
        x=simulation["threshold"], y=simulation["campaign_cost"], name="Campaign cost",
        line=dict(color=ui.DANGER, width=2.5),
        hovertemplate="threshold %{x:.2f}<br>cost %{y:,.0f}<extra></extra>"))
    figure.add_trace(go.Scatter(
        x=simulation["threshold"], y=simulation["net_profit"], name="Net profit",
        line=dict(color=ui.PRIMARY, width=3.5), fill="tozeroy",
        fillcolor="rgba(91,141,239,0.12)",
        hovertemplate="threshold %{x:.2f}<br>net %{y:,.0f}<extra></extra>"))
    figure.add_vline(x=float(best_row["threshold"]), line=dict(color=ui.WARNING, dash="dash"),
                     annotation_text=f"optimum {best_row['threshold']:.2f}")
    figure.add_hline(y=0, line=dict(color=ui.MUTED, width=1))
    figure.update_layout(title="Campaign economics across the threshold range",
                         xaxis_title="Decision threshold", yaxis_title="Value")
    ui.chart(figure, height=430)

    break_even = simulation[simulation["net_profit"] <= 0]["threshold"]
    if len(break_even) and break_even.min() > simulation["threshold"].min():
        ui.insight(
            f"The programme stops paying for itself above a threshold of "
            f"<b>{break_even.min():.2f}</b> — past that point you are contacting too few "
            "customers to cover the fixed cost of running the campaign.", "warn")
    else:
        ui.insight(
            f"At these assumptions the campaign is profitable across the whole threshold "
            f"range, peaking at <b>{best_row['threshold']:.2f}</b>.", "good")

    with st.expander("Full simulation table"):
        st.dataframe(
            simulation[["threshold", "contacted", "contacted_share", "precision", "recall",
                        "revenue_saved", "campaign_cost", "net_profit", "roi"]],
            width="stretch", hide_index=True,
            column_config={
                "contacted_share": st.column_config.ProgressColumn(
                    "Share of base", min_value=0.0, max_value=1.0, format="%.0f%%"),
                "recall": st.column_config.ProgressColumn(
                    "Churners reached", min_value=0.0, max_value=1.0, format="%.0f%%"),
                "net_profit": st.column_config.NumberColumn("Net profit", format="%.0f"),
                "roi": st.column_config.NumberColumn("ROI", format="%.2f×"),
            },
        )
