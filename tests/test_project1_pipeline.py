"""Correctness tests for the churn cleaning and feature-engineering stages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import use_project


@pytest.fixture(scope="module")
def raw():
    use_project(1)
    from src.generate_data import generate

    return generate(n_customers=600, seed=3)


def test_generator_produces_the_requested_churn_rate(raw):
    """The intercept solver should land close to the target rate."""
    assert raw["churned"].mean() == pytest.approx(0.26, abs=0.05)


def test_generator_emits_the_messiness_the_cleaner_is_built_for(raw):
    """If the raw data were clean, the cleaning stage would be untested decoration."""
    assert raw["gender"].nunique() > 2, "expected inconsistent gender spellings"
    assert raw["customer_id"].duplicated().any(), "expected duplicate CRM rows"
    assert raw.isna().sum().sum() > 0, "expected missing values"
    assert raw["region"].nunique() > 5, "expected region spelling variants"


def test_day_one_customers_get_a_blank_total_charges():
    """The blank-string case the cleaner must coerce.

    Zero-tenure customers are rare (well under 1% of rows), so this needs a large enough
    sample to contain any -- a few hundred customers often contains none, which would
    make the assertion flaky rather than wrong.
    """
    use_project(1)
    from src.generate_data import generate

    frame = generate(n_customers=6000, seed=3)
    day_one = frame[frame["tenure_months"] == 0]
    assert len(day_one) > 0, "sample contained no day-one customers to test"
    assert day_one["total_charges"].astype(str).str.strip().eq("").all()


def test_cleaning_normalises_categories_and_removes_duplicates(raw):
    use_project(1)
    from src.preprocess import clean

    cleaned, report = clean(raw)

    assert set(cleaned["gender"].unique()) <= {"Female", "Male", "Unknown"}
    assert not cleaned["customer_id"].duplicated().any()
    assert report.duplicates_dropped > 0
    # Region variants like " North " and "NORTH" must collapse to one label.
    assert all(region == region.strip().title() for region in cleaned["region"].unique())


def test_total_charges_becomes_numeric_with_day_one_customers_at_zero(raw):
    """A never-billed customer has zero lifetime charges -- that is known, not missing."""
    use_project(1)
    from src.preprocess import clean

    cleaned, _ = clean(raw)
    assert pd.api.types.is_numeric_dtype(cleaned["total_charges"])

    day_one = cleaned[cleaned["tenure_months"] == 0]
    if len(day_one):
        assert (day_one["total_charges"] == 0.0).all()
        assert day_one["total_charges"].notna().all()


def test_outliers_are_capped_rather_than_dropped(raw):
    """Winsorising keeps the customer; dropping the row would bias the sample."""
    use_project(1)
    from src.preprocess import clean

    rows_before = raw["customer_id"].nunique()
    cleaned, report = clean(raw)

    assert len(cleaned) == rows_before, "cleaning dropped customers, not just outliers"
    assert report.outliers_capped, "expected at least one capped column"


def test_missing_indicator_flags_are_created(raw):
    use_project(1)
    from src.preprocess import clean

    cleaned, _ = clean(raw)
    assert "avg_resolution_hours_missing" in cleaned
    assert cleaned["avg_resolution_hours"].notna().all()
    # Customers with no tickets must be exactly the flagged ones.
    no_contact = cleaned["support_tickets_6m"] == 0
    assert (cleaned.loc[no_contact, "had_support_contact"] == 0).all()


def test_usage_slope_sign_matches_a_constructed_trend():
    """A feature that gets the direction of a trend wrong is worse than no feature."""
    use_project(1)
    from src.features import build_features
    from src.preprocess import clean
    from src.generate_data import generate

    frame, _ = clean(generate(n_customers=200, seed=4))

    # Force a strictly falling and a strictly rising customer. m1 is the most recent
    # month, so a falling customer has the *smallest* value at m1.
    falling, rising = frame.index[0], frame.index[1]
    for month in range(1, 7):
        frame.loc[falling, f"usage_gb_m{month}"] = float(10 * month)       # 10 .. 60
        frame.loc[rising, f"usage_gb_m{month}"] = float(70 - 10 * month)   # 60 .. 10

    featured = build_features(frame)
    assert featured.loc[falling, "usage_slope_6m"] < 0
    assert featured.loc[rising, "usage_slope_6m"] > 0
    assert featured.loc[falling, "usage_recent_vs_prior"] < 1.0
    assert featured.loc[rising, "usage_recent_vs_prior"] > 1.0


def test_feature_columns_exclude_the_label_and_ground_truth():
    use_project(1)
    from src.features import build_features, feature_columns
    from src.preprocess import clean
    from src.generate_data import generate

    frame, _ = clean(generate(n_customers=200, seed=5))
    featured = build_features(frame)
    numeric, categorical = feature_columns(featured)
    inputs = set(numeric) | set(categorical)

    for forbidden in ("churned", "_true_churn_probability", "customer_id"):
        assert forbidden not in inputs, f"{forbidden} leaked into the feature set"

    # The per-month raw history is intentionally dropped in favour of the trends.
    assert "usage_gb_m1" not in inputs
    assert "usage_slope_6m" in inputs


def test_engineered_features_contain_no_infinities():
    """Ratio features divide by quantities that can be zero."""
    use_project(1)
    from src.features import build_features, feature_columns
    from src.preprocess import clean
    from src.generate_data import generate

    frame, _ = clean(generate(n_customers=400, seed=6))
    featured = build_features(frame)
    numeric, _ = feature_columns(featured)

    assert not np.isinf(featured[numeric].to_numpy(dtype=float)).any()


def test_risk_flag_count_stays_within_its_range():
    use_project(1)
    from src.features import build_features
    from src.preprocess import clean
    from src.generate_data import generate

    frame, _ = clean(generate(n_customers=300, seed=7))
    featured = build_features(frame)
    assert featured["risk_flag_count"].between(0, 6).all()
