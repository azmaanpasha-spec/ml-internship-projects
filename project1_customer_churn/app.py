"""Streamlit retention console for the churn model.

    streamlit run app.py

Four views: fleet-level summary, model comparison, the worked high-risk list, and an
interactive campaign simulator that re-prices the retention programme as the assumptions
change -- which is the part a business stakeholder actually wants to touch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).absolute().parent
REPORTS = HERE / "reports"
FIGURES = REPORTS / "figures"

st.set_page_config(page_title="Churn Retention Console", page_icon="📉", layout="wide")


@st.cache_data
def load_reports() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    evaluation = json.loads((REPORTS / "evaluation.json").read_text())
    training = json.loads((REPORTS / "training_results.json").read_text())
    high_risk = pd.read_csv(REPORTS / "high_risk_customers.csv")
    scored = pd.read_csv(REPORTS / "all_customers_scored.csv")
    return evaluation, training, high_risk, scored


def missing_artefacts() -> None:
    st.error("Model outputs not found.")
    st.markdown(
        "Run the pipeline first:\n\n"
        "```bash\n"
        "python -m src.generate_data\n"
        "python -m src.preprocess\n"
        "python -m src.features\n"
        "python -m src.train\n"
        "python -m src.evaluate\n"
        "python -m src.score_customers\n"
        "```"
    )


if not (REPORTS / "evaluation.json").exists():
    missing_artefacts()
    st.stop()

evaluation, training, high_risk, scored = load_reports()

st.title("Customer Churn — Retention Console")
st.caption(
    f"Best model: **{evaluation['best_model']}** (isotonic-calibrated) · "
    f"{len(scored):,} customers scored"
)

overview, comparison, worklist, simulator = st.tabs(
    ["Overview", "Model comparison", "High-risk worklist", "Campaign simulator"]
)

# --------------------------------------------------------------------------------------
with overview:
    best = evaluation["model_comparison"][f"{evaluation['best_model']}_calibrated"]
    profit = evaluation["performance_at_profit_threshold"]
    optimum = evaluation.get("campaign_optimum", {})

    row = st.columns(5)
    row[0].metric("ROC-AUC", f"{best['roc_auc']:.3f}")
    row[1].metric("PR-AUC", f"{best['pr_auc']:.3f}")
    row[2].metric("Brier score", f"{best['brier']:.3f}", help="lower is better; measures calibration")
    row[3].metric("Recall at operating point", f"{profit['recall']:.1%}")
    row[4].metric("Precision at operating point", f"{profit['precision']:.1%}")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Revenue at risk by band")
        bands = pd.DataFrame(json.loads((REPORTS / "scoring_summary.json").read_text())["risk_bands"])
        st.dataframe(bands, use_container_width=True, hide_index=True)
        st.caption(
            "`actual_churn_rate` tracking `mean_probability` is the calibration check: "
            "the model's stated confidence matches observed reality."
        )
    with right:
        st.subheader("Decile lift")
        if (FIGURES / "lift_curve.png").exists():
            st.image(str(FIGURES / "lift_curve.png"), use_container_width=True)

    st.subheader("Top churn drivers")
    drivers = pd.DataFrame(evaluation["permutation_importance"]).head(12)
    st.bar_chart(drivers.set_index("feature")["importance"], horizontal=True)
    st.caption("Permutation importance: the drop in ROC-AUC when each feature is shuffled.")

# --------------------------------------------------------------------------------------
with comparison:
    st.subheader("Hold-out performance by model")
    table = pd.DataFrame(evaluation["model_comparison"]).T
    display_columns = ["roc_auc", "pr_auc", "precision", "recall", "f1", "brier", "ece"]
    st.dataframe(
        table[display_columns].style.format("{:.4f}").background_gradient(
            subset=["roc_auc", "pr_auc"], cmap="Greens"
        ),
        use_container_width=True,
    )
    st.caption(
        "`ece` is expected calibration error. Note how sharply it falls after isotonic "
        "calibration — the ranking barely changes, but the probabilities become usable "
        "for budgeting."
    )

    for name, caption in [
        ("model_comparison.png", "Metrics side by side"),
        ("roc_curves.png", "ROC curves"),
        ("pr_curves.png", "Precision-recall curves — the honest view with a 26% positive class"),
        ("calibration.png", "Calibration: predicted probability vs. observed frequency"),
        ("shap_importance.png", "SHAP feature attributions"),
    ]:
        if (FIGURES / name).exists():
            st.image(str(FIGURES / name), caption=caption, use_container_width=True)

# --------------------------------------------------------------------------------------
with worklist:
    st.subheader("Ranked high-risk customers")
    controls = st.columns([1, 1, 2])
    bands = controls[0].multiselect(
        "Risk band", sorted(high_risk["risk_band"].unique()),
        default=sorted(high_risk["risk_band"].unique()),
    )
    min_value = controls[1].number_input("Min revenue at risk", value=0.0, step=100.0)
    search = controls[2].text_input("Search customer ID")

    filtered = high_risk[high_risk["risk_band"].isin(bands)]
    filtered = filtered[filtered["revenue_at_risk"] >= min_value]
    if search:
        filtered = filtered[filtered["customer_id"].str.contains(search, case=False, na=False)]

    st.caption(f"{len(filtered):,} customers match. Total revenue at risk: "
               f"{filtered['revenue_at_risk'].sum():,.0f}")
    st.dataframe(
        filtered[
            ["customer_id", "churn_probability", "risk_band", "revenue_at_risk", "tenure_months",
             "contract", "monthly_charges", "recommended_action"]
        ],
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.subheader("Customer detail")
    if len(filtered):
        chosen = st.selectbox("Customer", filtered["customer_id"].tolist())
        record = filtered[filtered["customer_id"] == chosen].iloc[0]

        detail = st.columns(4)
        detail[0].metric("Churn probability", f"{record['churn_probability']:.1%}")
        detail[1].metric("Risk band", record["risk_band"])
        detail[2].metric("Revenue at risk", f"{record['revenue_at_risk']:,.0f}")
        detail[3].metric("Tenure", f"{record['tenure_months']:.0f} months")

        st.markdown("**Why this customer is flagged**")
        for key in ("reason_1", "reason_2", "reason_3"):
            if isinstance(record[key], str) and record[key]:
                st.markdown(f"- {record[key]}")
        st.success(f"**Recommended action:** {record['recommended_action']}")

# --------------------------------------------------------------------------------------
with simulator:
    st.subheader("Retention campaign simulator")
    st.caption(
            "The threshold is a business decision, not a statistical one. These are the "
            "assumptions behind the recommended operating point — change them and the "
            "optimum moves."
    )

    assumptions = evaluation["campaign_assumptions"]
    inputs = st.columns(4)
    offer_cost = inputs[0].number_input(
        "Retention offer cost", value=float(assumptions["retention_offer_cost"]), step=5.0
    )
    contact_cost = inputs[1].number_input(
        "Contact cost", value=float(assumptions["contact_cost"]), step=1.0
    )
    success_rate = inputs[2].slider(
        "Offer success rate", 0.05, 0.80, float(assumptions["offer_success_rate"]), 0.01
    )
    horizon = inputs[3].number_input(
        "Expected remaining months", value=int(assumptions["expected_remaining_months"]), step=1
    )

    simulation = pd.DataFrame(evaluation["campaign_simulation"])
    # Re-price the saved simulation under the user's assumptions. Revenue saved scales
    # with the success rate and horizon; cost scales with the per-contact spend.
    base = assumptions
    scale_revenue = (success_rate / base["offer_success_rate"]) * (
        horizon / base["expected_remaining_months"]
    )
    simulation["revenue_saved"] = simulation["revenue_saved"] * scale_revenue
    simulation["campaign_cost"] = simulation["contacted"] * (offer_cost + contact_cost)
    simulation["net_profit"] = simulation["revenue_saved"] - simulation["campaign_cost"]
    simulation["roi"] = simulation["net_profit"] / simulation["campaign_cost"].clip(lower=1)

    best_row = simulation.loc[simulation["net_profit"].idxmax()]
    headline = st.columns(4)
    headline[0].metric("Optimal threshold", f"{best_row['threshold']:.2f}")
    headline[1].metric("Customers contacted", f"{int(best_row['contacted']):,}")
    headline[2].metric("Churners reached", f"{best_row['recall']:.1%}")
    headline[3].metric("Net profit", f"{best_row['net_profit']:,.0f}", f"ROI {best_row['roi']:.2f}x")

    st.line_chart(simulation.set_index("threshold")[["net_profit", "campaign_cost", "revenue_saved"]])
    st.dataframe(
        simulation[["threshold", "contacted", "precision", "recall", "revenue_saved",
                    "campaign_cost", "net_profit", "roi"]].round(3),
        use_container_width=True, hide_index=True,
    )
