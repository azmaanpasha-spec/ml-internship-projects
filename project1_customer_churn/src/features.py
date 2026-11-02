"""Step 3 of the approach: turn six months of history into behavioural features.

The raw table says what a customer looks like today. These features say where the
customer is *heading* -- usage trend, support-contact escalation, recent billing change
-- which is what separates a churn model from a static profile lookup.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

HISTORY_MONTHS = 6
USAGE_COLUMNS = [f"usage_gb_m{m}" for m in range(1, HISTORY_MONTHS + 1)]
CHARGE_COLUMNS = [f"charges_m{m}" for m in range(1, HISTORY_MONTHS + 1)]
TICKET_COLUMNS = [f"tickets_m{m}" for m in range(1, HISTORY_MONTHS + 1)]
LOGIN_COLUMNS = [f"logins_m{m}" for m in range(1, HISTORY_MONTHS + 1)]

SERVICE_COLUMNS = [
    "phone_service", "multiple_lines", "online_security", "tech_support",
    "device_protection", "streaming_tv", "streaming_movies",
]

# Columns that must never reach the model: identifiers and generator ground truth.
LEAKAGE_COLUMNS = ["customer_id", "churned", "_true_churn_probability"]


def _monthly_matrix(frame: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """History as an (n_customers, n_months) array ordered oldest -> newest.

    Raw columns run m1 = most recent, so they are reversed here; a positive slope then
    means "increasing over time", which is the intuitive reading.
    """
    return frame[list(reversed(columns))].to_numpy(dtype=float)


def _nanmean(matrix: np.ndarray, axis: int | None = None) -> np.ndarray:
    """`np.nanmean` that quietly returns NaN for all-NaN rows.

    Customers who joined this month have no history at all; that is expected, and the
    resulting NaN is imputed at model time rather than papered over here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(matrix, axis=axis)


def _nanstd(matrix: np.ndarray, axis: int | None = None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(matrix, axis=axis)


def _slope(matrix: np.ndarray) -> np.ndarray:
    """Least-squares slope per row, using only the months that are actually observed.

    Rows with fewer than two observations get slope 0 -- no trend can be claimed.
    """
    n_rows, n_months = matrix.shape
    time = np.arange(n_months, dtype=float)
    slopes = np.zeros(n_rows)

    observed = ~np.isnan(matrix)
    for i in range(n_rows):
        mask = observed[i]
        if mask.sum() < 2:
            continue
        x, y = time[mask], matrix[i, mask]
        x_centered = x - x.mean()
        denominator = (x_centered**2).sum()
        if denominator > 1e-9:
            slopes[i] = float((x_centered * (y - y.mean())).sum() / denominator)
    return slopes


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise ratio that returns 1.0 where the denominator is ~0 or unknown."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.ones_like(numerator)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _recent_vs_prior(matrix: np.ndarray, window: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the last `window` months and of the `window` before it (oldest-first input)."""
    recent = _nanmean(matrix[:, -window:], axis=1)
    prior = _nanmean(matrix[:, :-window], axis=1) if matrix.shape[1] > window else recent
    return recent, prior


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns to a cleaned frame."""
    frame = frame.copy()

    usage = _monthly_matrix(frame, USAGE_COLUMNS)
    charges = _monthly_matrix(frame, CHARGE_COLUMNS)
    tickets = _monthly_matrix(frame, TICKET_COLUMNS)
    logins = _monthly_matrix(frame, LOGIN_COLUMNS)

    # --- Usage dynamics: the single strongest behavioural signal ----------------------
    usage_mean = _nanmean(usage, axis=1)
    frame["usage_mean_6m"] = usage_mean
    frame["usage_slope_6m"] = _slope(usage)
    # Slope normalised by level, so a 2 GB/month drop means more for a light user.
    frame["usage_trend_pct"] = _safe_ratio(frame["usage_slope_6m"].to_numpy(), usage_mean)
    frame["usage_volatility"] = _safe_ratio(_nanstd(usage, axis=1), usage_mean)

    usage_recent, usage_prior = _recent_vs_prior(usage)
    frame["usage_recent_3m"] = usage_recent
    frame["usage_recent_vs_prior"] = _safe_ratio(usage_recent, usage_prior)
    frame["usage_dropped_sharply"] = (frame["usage_recent_vs_prior"] < 0.75).astype(int)

    # --- Billing dynamics: a recent price change is a classic churn trigger ----------
    charges_recent, charges_prior = _recent_vs_prior(charges)
    frame["charges_recent_3m"] = charges_recent
    frame["price_change_pct"] = _safe_ratio(charges_recent, charges_prior) - 1.0
    frame["had_price_increase"] = (frame["price_change_pct"] > 0.03).astype(int)
    frame["charges_slope_6m"] = _slope(charges)

    # --- Support dynamics: escalating contact frequency, not just the raw count ------
    tickets_recent, tickets_prior = _recent_vs_prior(tickets)
    frame["tickets_recent_3m"] = tickets_recent
    frame["tickets_prior_3m"] = tickets_prior
    frame["ticket_escalation"] = tickets_recent - tickets_prior
    frame["tickets_per_month"] = _safe_ratio(
        frame["support_tickets_6m"].to_numpy(),
        np.maximum(frame["observed_history_months"].to_numpy(), 1),
    )
    frame["escalation_rate"] = _safe_ratio(
        frame["escalations_6m"].to_numpy(),
        np.maximum(frame["support_tickets_6m"].to_numpy(), 1),
    )

    # --- Engagement ------------------------------------------------------------------
    logins_recent, logins_prior = _recent_vs_prior(logins)
    frame["logins_recent_3m"] = logins_recent
    frame["login_slope_6m"] = _slope(logins)
    frame["login_recent_vs_prior"] = _safe_ratio(logins_recent, logins_prior)

    # --- Value and tenure ------------------------------------------------------------
    frame["avg_revenue_per_month"] = _safe_ratio(
        frame["total_charges"].to_numpy(), np.maximum(frame["tenure_months"].to_numpy(), 1)
    )
    frame["charges_per_gb"] = _safe_ratio(
        frame["monthly_charges"].to_numpy(), np.maximum(usage_mean, 0.1)
    )
    frame["tenure_bucket"] = pd.cut(
        frame["tenure_months"],
        bins=[-1, 6, 12, 24, 48, 100],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"],
    ).astype(str)
    frame["is_new_customer"] = (frame["tenure_months"] <= 6).astype(int)

    # --- Product footprint: more services subscribed means more switching friction ----
    frame["services_count"] = sum(
        (frame[column] == "Yes").astype(int) for column in SERVICE_COLUMNS
    )
    frame["has_protective_addons"] = (
        (frame["online_security"] == "Yes") | (frame["tech_support"] == "Yes")
    ).astype(int)
    frame["streaming_count"] = (
        (frame["streaming_tv"] == "Yes").astype(int) + (frame["streaming_movies"] == "Yes").astype(int)
    )

    # --- Payment friction ------------------------------------------------------------
    frame["late_payment_rate"] = _safe_ratio(
        frame["late_payments_12m"].to_numpy(),
        np.minimum(np.maximum(frame["tenure_months"].to_numpy(), 1), 12),
    )
    frame["manual_payment"] = (
        (frame["autopay"] == "No") & (frame["payment_method"].isin(["Electronic check", "Mailed check"]))
    ).astype(int)

    # --- Composite risk flag: cheap, explainable, and a strong single predictor -------
    frame["risk_flag_count"] = (
        frame["usage_dropped_sharply"]
        + frame["had_price_increase"]
        + (frame["contract"] == "Month-to-month").astype(int)
        + (frame["ticket_escalation"] > 0).astype(int)
        + frame["manual_payment"]
        + (frame["satisfaction_score"] < 6).astype(int)
    )

    return frame.replace([np.inf, -np.inf], np.nan)


def feature_columns(frame: pd.DataFrame) -> tuple[List[str], List[str]]:
    """Split the modelling columns into numeric and categorical, excluding leakage."""
    candidates = [column for column in frame.columns if column not in LEAKAGE_COLUMNS]
    # Per-month raw history is dropped: the engineered trend features carry the same
    # information in a form that generalises, and keeping both invites overfitting.
    raw_history = set(USAGE_COLUMNS + CHARGE_COLUMNS + TICKET_COLUMNS + LOGIN_COLUMNS)
    candidates = [column for column in candidates if column not in raw_history]

    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in candidates if c not in numeric]
    return numeric, categorical


def main() -> None:
    here = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Engineer churn features.")
    parser.add_argument("--clean", type=Path, default=here / "data" / "processed" / "customers_clean.csv")
    parser.add_argument("--out", type=Path, default=here / "data" / "processed" / "customers_features.csv")
    args = parser.parse_args()

    frame = build_features(pd.read_csv(args.clean))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    numeric, categorical = feature_columns(frame)
    print(f"engineered frame: {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    print(f"model inputs: {len(numeric)} numeric + {len(categorical)} categorical")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
