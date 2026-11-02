"""Step 6 of the approach: score the base and produce a ranked high-risk list with
reasons a human can act on.

A probability on its own does not tell a retention agent what to say. Each flagged
customer therefore gets:

* a calibrated churn probability and a risk band,
* the revenue at risk, so the list can be worked in value order,
* up to three plain-language reasons, taken from the per-customer SHAP contributions
  and translated against the customer's own values,
* a suggested next action derived from the dominant reason.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

from common.explain import aggregate_to_source_columns, shap_values_for_pipeline  # noqa: E402
from common.io_utils import load_json, load_model, save_json  # noqa: E402

warnings.filterwarnings("ignore")

EXPECTED_REMAINING_MONTHS = 14

# Feature -> (human-readable reason, suggested action). The callable receives the
# customer's row and returns False when the feature is present in the model's
# explanation but not actually adverse for this customer.
ReasonRule = Tuple[str, str, Callable[[pd.Series], bool]]

REASON_RULES: Dict[str, ReasonRule] = {
    "usage_recent_vs_prior": (
        "Usage down {drop:.0%} over the last 3 months vs. the prior 3",
        "Ask what changed; offer a plan better matched to current usage",
        lambda row: row.get("usage_recent_vs_prior", 1.0) < 0.9,
    ),
    "usage_slope_6m": (
        "Steadily declining data usage across the 6-month history",
        "Re-engagement outreach with a usage-based incentive",
        lambda row: row.get("usage_slope_6m", 0.0) < 0,
    ),
    "usage_dropped_sharply": (
        "Sharp usage drop flagged (>25% fall quarter on quarter)",
        "Priority call; investigate service quality in their area",
        lambda row: row.get("usage_dropped_sharply", 0) == 1,
    ),
    "usage_trend_pct": (
        "Negative usage trend relative to their own typical level",
        "Re-engagement outreach with a usage-based incentive",
        lambda row: row.get("usage_trend_pct", 0.0) < 0,
    ),
    "contract": (
        "On a month-to-month contract, so can leave with no penalty",
        "Offer a discounted 12-month term to lock in tenure",
        lambda row: row.get("contract") == "Month-to-month",
    ),
    "tenure_months": (
        "Short tenure ({tenure:.0f} months) - still inside the early-churn window",
        "Onboarding check-in call and first-year loyalty credit",
        lambda row: row.get("tenure_months", 99) <= 12,
    ),
    "is_new_customer": (
        "New customer (under 6 months) with elevated risk",
        "Onboarding check-in call and first-year loyalty credit",
        lambda row: row.get("is_new_customer", 0) == 1,
    ),
    "price_change_pct": (
        "Bill rose {increase:.0%} in the last 3 months",
        "Pre-emptive goodwill credit or price-lock offer",
        lambda row: row.get("price_change_pct", 0.0) > 0.03,
    ),
    "had_price_increase": (
        "Recent price increase on the account",
        "Pre-emptive goodwill credit or price-lock offer",
        lambda row: row.get("had_price_increase", 0) == 1,
    ),
    "ticket_escalation": (
        "Support contacts rising ({escalation:+.1f} tickets vs. the prior quarter)",
        "Route to a senior agent; confirm the open issue is resolved",
        lambda row: row.get("ticket_escalation", 0.0) > 0,
    ),
    "support_tickets_6m": (
        "{tickets:.0f} support tickets in 6 months",
        "Route to a senior agent; confirm the open issue is resolved",
        lambda row: row.get("support_tickets_6m", 0) >= 2,
    ),
    "escalations_6m": (
        "{escalations:.0f} escalated complaints in the last 6 months",
        "Service-recovery call from a retention specialist",
        lambda row: row.get("escalations_6m", 0) >= 1,
    ),
    "avg_resolution_hours": (
        "Slow issue resolution (avg {resolution:.0f} hours)",
        "Service-recovery call from a retention specialist",
        lambda row: row.get("avg_resolution_hours", 0.0) > 25,
    ),
    "satisfaction_score": (
        "Low satisfaction score ({satisfaction:.1f}/10)",
        "Service-recovery call from a retention specialist",
        lambda row: row.get("satisfaction_score", 10.0) < 6.5,
    ),
    "late_payments_12m": (
        "{late:.0f} late payments in the last 12 months",
        "Offer autopay enrolment with a small billing incentive",
        lambda row: row.get("late_payments_12m", 0) >= 2,
    ),
    "late_payment_rate": (
        "Frequent late payments relative to tenure",
        "Offer autopay enrolment with a small billing incentive",
        lambda row: row.get("late_payment_rate", 0.0) > 0.15,
    ),
    "manual_payment": (
        "Pays manually by cheque/electronic check, not autopay",
        "Offer autopay enrolment with a small billing incentive",
        lambda row: row.get("manual_payment", 0) == 1,
    ),
    "autopay": (
        "Not enrolled in autopay",
        "Offer autopay enrolment with a small billing incentive",
        lambda row: row.get("autopay") == "No",
    ),
    "has_protective_addons": (
        "No tech-support or security add-ons, so low switching friction",
        "Bundle tech support free for 3 months",
        lambda row: row.get("has_protective_addons", 1) == 0,
    ),
    "tech_support": (
        "No tech-support package on the account",
        "Bundle tech support free for 3 months",
        lambda row: row.get("tech_support") == "No",
    ),
    "services_count": (
        "Only {services:.0f} services subscribed - weak product attachment",
        "Cross-sell a complementary service at an introductory rate",
        lambda row: row.get("services_count", 9) <= 2,
    ),
    "internet_service": (
        "Fiber customer - highest-churn segment in the base",
        "Proactive line-quality check and speed guarantee",
        lambda row: row.get("internet_service") == "Fiber optic",
    ),
    "login_recent_vs_prior": (
        "App/portal logins falling quarter on quarter",
        "Re-engagement campaign highlighting unused features",
        lambda row: row.get("login_recent_vs_prior", 1.0) < 0.9,
    ),
    "charges_per_gb": (
        "Paying a high rate per GB compared with the base",
        "Plan review to find a better-value tariff",
        lambda row: row.get("charges_per_gb", 0.0) > 3.0,
    ),
    "risk_flag_count": (
        "{flags:.0f} independent risk flags triggered",
        "Priority retention call",
        lambda row: row.get("risk_flag_count", 0) >= 3,
    ),
}


def _format_reason(template: str, row: pd.Series) -> str:
    """Fill a reason template with the customer's own numbers."""
    values = {
        "drop": 1.0 - float(row.get("usage_recent_vs_prior", 1.0)),
        "increase": float(row.get("price_change_pct", 0.0)),
        "escalation": float(row.get("ticket_escalation", 0.0)),
        "tickets": float(row.get("support_tickets_6m", 0)),
        "escalations": float(row.get("escalations_6m", 0)),
        "resolution": float(row.get("avg_resolution_hours", 0.0)),
        "satisfaction": float(row.get("satisfaction_score", 0.0)),
        "late": float(row.get("late_payments_12m", 0)),
        "tenure": float(row.get("tenure_months", 0)),
        "services": float(row.get("services_count", 0)),
        "flags": float(row.get("risk_flag_count", 0)),
    }
    try:
        return template.format(**values)
    except (KeyError, ValueError):
        return template


def _shap_contributions(pipeline, X: pd.DataFrame) -> np.ndarray | None:
    """Per-customer SHAP values aggregated back to the original column names.

    One-hot columns are summed back into their source feature, so the explanation talks
    about `contract`, not `contract_Month-to-month`.
    """
    values, names = shap_values_for_pipeline(pipeline, X)
    if values is None:
        print("  per-customer SHAP unavailable - using rule-based reason ordering")
        return None
    return aggregate_to_source_columns(values, names, list(X.columns)).to_numpy()


def _reasons_for_row(
    row: pd.Series, column_order: List[str], contributions: np.ndarray | None, max_reasons: int = 3
) -> Tuple[List[str], str]:
    """Top reasons for one customer, plus the suggested action for the strongest one."""
    if contributions is not None:
        # Walk features by descending risk-increasing SHAP contribution.
        ranked = [column_order[i] for i in np.argsort(contributions)[::-1]]
    else:
        # No SHAP: fall back to the fixed rule order, which is roughly importance-ordered.
        ranked = list(REASON_RULES)

    reasons, actions = [], []
    for feature in ranked:
        rule = REASON_RULES.get(feature)
        if rule is None:
            continue
        template, action, applies = rule
        if not applies(row):
            continue
        text = _format_reason(template, row)
        if text in reasons:
            continue
        reasons.append(text)
        actions.append(action)
        if len(reasons) >= max_reasons:
            break

    if not reasons:
        reasons = ["Elevated model risk without a single dominant driver"]
        actions = ["Standard retention offer"]
    return reasons, actions[0]


def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "Critical"
    if probability >= 0.50:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"


def score_base(
    features_path: Path, models_dir: Path, reports_dir: Path, top_n: int = 500
) -> pd.DataFrame:
    frame = pd.read_csv(features_path)
    card = load_json(models_dir / "model_card.json")
    model_columns = card["numeric_features"] + card["categorical_features"]

    calibrated = load_model(models_dir / "best_model.joblib")
    X = frame[model_columns]
    # Isotonic calibration maps its top bin to exactly 1.0. Reporting "100% certain to
    # churn" to a retention agent is indefensible, so the scores are clipped to a
    # plausible range. This changes the ranking not at all and the credibility a lot.
    frame["churn_probability"] = np.clip(calibrated.predict_proba(X)[:, 1], 0.001, 0.995)

    frame["risk_band"] = frame["churn_probability"].map(risk_band)
    frame["risk_decile"] = (
        pd.qcut(frame["churn_probability"].rank(method="first", ascending=False), 10, labels=False) + 1
    )
    frame["revenue_at_risk"] = (
        frame["churn_probability"] * frame["monthly_charges"] * EXPECTED_REMAINING_MONTHS
    ).round(2)

    ranked = frame.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    shortlist = ranked.head(top_n).copy()

    print(f"explaining the top {len(shortlist)} customers...")
    best_pipeline = load_model(models_dir / f"{card['best_model']}.joblib")
    contributions = _shap_contributions(best_pipeline, shortlist[model_columns])

    reason_rows, action_rows = [], []
    for position, (_, row) in enumerate(shortlist.iterrows()):
        row_contributions = contributions[position] if contributions is not None else None
        reasons, action = _reasons_for_row(row, model_columns, row_contributions)
        reason_rows.append(reasons)
        action_rows.append(action)

    shortlist["reason_1"] = [r[0] if len(r) > 0 else "" for r in reason_rows]
    shortlist["reason_2"] = [r[1] if len(r) > 1 else "" for r in reason_rows]
    shortlist["reason_3"] = [r[2] if len(r) > 2 else "" for r in reason_rows]
    shortlist["recommended_action"] = action_rows

    output_columns = [
        "customer_id", "churn_probability", "risk_band", "risk_decile", "tenure_months",
        "contract", "monthly_charges", "revenue_at_risk", "satisfaction_score",
        "support_tickets_6m", "usage_recent_vs_prior", "price_change_pct",
        "reason_1", "reason_2", "reason_3", "recommended_action",
    ]
    shortlist = shortlist[output_columns].round(
        {"churn_probability": 4, "usage_recent_vs_prior": 3, "price_change_pct": 4}
    )

    out_dir = reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(out_dir / "high_risk_customers.csv", index=False)
    ranked[["customer_id", "churn_probability", "risk_band", "risk_decile", "revenue_at_risk"]].to_csv(
        out_dir / "all_customers_scored.csv", index=False
    )

    band_summary = (
        frame.groupby("risk_band")
        .agg(
            customers=("customer_id", "count"),
            mean_probability=("churn_probability", "mean"),
            revenue_at_risk=("revenue_at_risk", "sum"),
            actual_churn_rate=("churned", "mean"),
        )
        .round(4)
        .reindex(["Critical", "High", "Medium", "Low"])
        .dropna(how="all")
    )
    save_json(
        {
            "scored_customers": int(len(frame)),
            "shortlist_size": int(len(shortlist)),
            "total_revenue_at_risk": round(float(frame["revenue_at_risk"].sum()), 2),
            "risk_bands": band_summary.reset_index().to_dict(orient="records"),
            "top_actions": shortlist["recommended_action"].value_counts().head(10).to_dict(),
        },
        out_dir / "scoring_summary.json",
    )

    print("\nRISK BAND SUMMARY")
    print(band_summary.to_string())
    print(f"\ntotal revenue at risk: {frame['revenue_at_risk'].sum():,.0f}")
    print("\nTOP 10 HIGHEST-RISK CUSTOMERS")
    for _, row in shortlist.head(10).iterrows():
        print(f"  {row['customer_id']}  p={row['churn_probability']:.3f}  "
              f"{row['risk_band']:<9} at-risk={row['revenue_at_risk']:>8,.0f}")
        print(f"      - {row['reason_1']}")
        if row["reason_2"]:
            print(f"      - {row['reason_2']}")
        print(f"      -> {row['recommended_action']}")

    print(f"\nwrote -> {out_dir / 'high_risk_customers.csv'}")
    return shortlist


def main() -> None:
    here = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Score customers and rank churn risk.")
    parser.add_argument(
        "--features", type=Path, default=here / "data" / "processed" / "customers_features.csv"
    )
    parser.add_argument("--models", type=Path, default=here / "models")
    parser.add_argument("--reports", type=Path, default=here / "reports")
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()
    score_base(args.features, args.models, args.reports, args.top_n)


if __name__ == "__main__":
    main()
