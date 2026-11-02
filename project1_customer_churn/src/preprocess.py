"""Step 2 of the approach: clean the raw CRM export.

Handles the four problems the brief calls out -- missing values, categorical
inconsistency, outliers, and the structure needed before class imbalance can be
addressed -- and emits a data-quality report so the cleaning is auditable rather than
silent.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HISTORY_MONTHS = 6
USAGE_COLUMNS = [f"usage_gb_m{m}" for m in range(1, HISTORY_MONTHS + 1)]
CHARGE_COLUMNS = [f"charges_m{m}" for m in range(1, HISTORY_MONTHS + 1)]
TICKET_COLUMNS = [f"tickets_m{m}" for m in range(1, HISTORY_MONTHS + 1)]
LOGIN_COLUMNS = [f"logins_m{m}" for m in range(1, HISTORY_MONTHS + 1)]

GENDER_MAP = {
    "female": "Female", "f": "Female",
    "male": "Male", "m": "Male",
}


@dataclass
class CleaningReport:
    """What the cleaning step actually changed -- printed and saved alongside the data."""

    rows_in: int = 0
    rows_out: int = 0
    duplicates_dropped: int = 0
    categories_normalised: Dict[str, int] = field(default_factory=dict)
    missing_before: Dict[str, int] = field(default_factory=dict)
    missing_after: Dict[str, int] = field(default_factory=dict)
    outliers_capped: Dict[str, int] = field(default_factory=dict)
    class_balance: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "duplicates_dropped": self.duplicates_dropped,
            "categories_normalised": self.categories_normalised,
            "missing_before": self.missing_before,
            "missing_after": self.missing_after,
            "outliers_capped": self.outliers_capped,
            "class_balance": self.class_balance,
            "notes": self.notes,
        }


def _normalise_categories(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Collapse the spelling variants that come from multiple source systems."""
    raw_gender = frame["gender"].astype(str)
    cleaned_gender = raw_gender.str.strip().str.lower().map(GENDER_MAP)
    report.categories_normalised["gender"] = int((cleaned_gender != raw_gender).sum())
    frame["gender"] = cleaned_gender.fillna("Unknown")

    raw_region = frame["region"].astype(str)
    cleaned_region = raw_region.str.strip().str.title()
    report.categories_normalised["region"] = int((cleaned_region != raw_region).sum())
    frame["region"] = cleaned_region

    # Every remaining text column: strip whitespace, title-case the yes/no flags.
    for column in frame.select_dtypes(exclude=[np.number]).columns:
        if column == "customer_id":
            continue
        frame[column] = frame[column].astype(str).str.strip()
        if set(frame[column].str.lower().unique()) <= {"yes", "no"}:
            frame[column] = frame[column].str.capitalize()
    return frame


def _coerce_numeric(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """`total_charges` arrives as a blank string for never-billed customers."""
    blanks = frame["total_charges"].astype(str).str.strip().eq("").sum()
    frame["total_charges"] = pd.to_numeric(frame["total_charges"], errors="coerce")

    # A customer with zero tenure genuinely has zero lifetime charges -- not unknown.
    day_one = frame["tenure_months"] == 0
    frame.loc[day_one & frame["total_charges"].isna(), "total_charges"] = 0.0
    report.notes.append(
        f"total_charges: {int(blanks)} blank strings coerced; "
        f"{int(day_one.sum())} day-one customers set to 0.0 (structurally zero, not missing)"
    )
    return frame


def _cap_outliers(frame: pd.DataFrame, report: CleaningReport, quantile: float = 0.995) -> pd.DataFrame:
    """Winsorise meter glitches rather than dropping the customers they belong to."""
    for column in USAGE_COLUMNS + CHARGE_COLUMNS + ["avg_resolution_hours", "total_charges"]:
        if column not in frame:
            continue
        ceiling = frame[column].quantile(quantile)
        if not np.isfinite(ceiling):
            continue
        above = int((frame[column] > ceiling).sum())
        if above:
            frame[column] = frame[column].clip(upper=ceiling)
            report.outliers_capped[column] = above
    return frame


def _impute(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Separate structural absence from genuine missingness, and flag both."""
    # No ticket raised => no resolution time exists. Encode as 0 with an explicit flag.
    frame["had_support_contact"] = (frame["support_tickets_6m"] > 0).astype(int)
    frame["avg_resolution_hours_missing"] = frame["avg_resolution_hours"].isna().astype(int)
    frame["avg_resolution_hours"] = frame["avg_resolution_hours"].fillna(0.0)

    # History months before a customer joined are structurally absent. The engineered
    # features in features.py use only the months that exist, so the NaNs stay for now
    # and an explicit count of observed months becomes a feature in its own right.
    frame["observed_history_months"] = frame[USAGE_COLUMNS].notna().sum(axis=1)

    # Genuine gaps in recent usage: fill from that customer's own history, then global median.
    usage_missing_before = int(frame[USAGE_COLUMNS].isna().sum().sum())
    row_median = frame[USAGE_COLUMNS].median(axis=1)
    for column in USAGE_COLUMNS:
        gap = frame[column].isna() & (frame["observed_history_months"] > 0)
        frame.loc[gap, column] = row_median[gap]
    report.notes.append(
        f"usage history: {usage_missing_before} missing cells, "
        f"{int(frame[USAGE_COLUMNS].isna().sum().sum())} remain (customers younger than the window)"
    )

    for column in [*CHARGE_COLUMNS, *TICKET_COLUMNS, *LOGIN_COLUMNS]:
        gap = frame[column].isna() & (frame["observed_history_months"] > 0)
        frame.loc[gap, column] = frame[column].median()

    frame["satisfaction_score"] = frame["satisfaction_score"].fillna(
        frame["satisfaction_score"].median()
    )
    return frame


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning sequence and return the frame plus an audit report."""
    report = CleaningReport(rows_in=len(frame))
    frame = frame.copy()

    tracked = ["total_charges", "avg_resolution_hours", *USAGE_COLUMNS]
    report.missing_before = {
        column: int(frame[column].isna().sum() + frame[column].astype(str).str.strip().eq("").sum())
        if frame[column].dtype == object
        else int(frame[column].isna().sum())
        for column in tracked
    }

    before = len(frame)
    frame = frame.drop_duplicates(subset="customer_id", keep="first").reset_index(drop=True)
    report.duplicates_dropped = before - len(frame)

    frame = _normalise_categories(frame, report)
    frame = _coerce_numeric(frame, report)
    frame = _cap_outliers(frame, report)
    frame = _impute(frame, report)

    report.missing_after = {column: int(frame[column].isna().sum()) for column in tracked}
    report.rows_out = len(frame)

    if "churned" in frame:
        positives = int(frame["churned"].sum())
        report.class_balance = {
            "positives": positives,
            "negatives": int(len(frame) - positives),
            "positive_rate": round(float(frame["churned"].mean()), 4),
            "imbalance_ratio": round(float((len(frame) - positives) / max(positives, 1)), 2),
        }
        report.notes.append(
            "class imbalance is handled at model level (class_weight / scale_pos_weight) "
            "rather than by resampling, so predicted probabilities stay calibrated"
        )

    return frame, report


def main() -> None:
    here = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Clean the raw churn export.")
    parser.add_argument("--raw", type=Path, default=here / "data" / "raw" / "customers.csv")
    parser.add_argument("--out", type=Path, default=here / "data" / "processed" / "customers_clean.csv")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    cleaned, report = clean(raw)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.out, index=False)

    print(f"rows {report.rows_in:,} -> {report.rows_out:,} ({report.duplicates_dropped} duplicates dropped)")
    print(f"category fixes: {report.categories_normalised}")
    print(f"outliers capped: {report.outliers_capped}")
    for note in report.notes:
        print(f"  - {note}")
    print(f"class balance: {report.class_balance}")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
