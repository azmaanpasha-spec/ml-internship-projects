"""Correctness tests for the predictive-maintenance labelling and feature pipeline.

These are the highest-value tests in the repository. Both of the rules under test here
silently inflate the reported score when broken, and neither failure is visible in the
metrics -- a leaking pipeline just looks like a very good model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import use_project

SENSORS = [
    "temperature_c", "vibration_rms", "pressure_bar", "rpm",
    "oil_quality_idx", "power_kw", "acoustic_db",
]


def _toy_fleet(n_hours: int = 60, failure_hour: int = 40, downtime: int = 5):
    """One machine, 60 hourly readings, a single failure at hour 40.

    Hand-built so the expected label for every row can be reasoned about exactly.
    """
    timestamps = pd.date_range("2026-01-01", periods=n_hours, freq="h")
    rng = np.random.default_rng(0)

    readings = pd.DataFrame(
        {
            "machine_id": "MCH-001",
            "timestamp": timestamps,
            **{sensor: rng.normal(50, 1, n_hours) for sensor in SENSORS},
            "ambient_temp_c": rng.normal(22, 0.5, n_hours),
            "load_pct": rng.normal(70, 5, n_hours),
            "machine_state": ["running"] * n_hours,
            "hours_since_maintenance": np.arange(n_hours),
            "_health_index": np.ones(n_hours),
            "operating_hours": np.arange(n_hours) + 1000,
        }
    )
    readings.loc[failure_hour : failure_hour + downtime - 1, "machine_state"] = "down"

    failures = pd.DataFrame(
        [
            {
                "machine_id": "MCH-001",
                "failure_time": timestamps[failure_hour],
                "failure_mode": "bearing_wear",
                "onset_hours": 80.0,
                "is_sudden": False,
                "severity": 1.0,
                "downtime_hours": downtime,
                "repair_cost": 5000.0,
            }
        ]
    )
    machines = pd.DataFrame(
        [
            {
                "machine_id": "MCH-001",
                "machine_type": "Centrifugal Pump",
                "age_years": 5.0,
                "rated_power_kw": 22.0,
                "site": "Plant-A",
                "criticality": "High",
                "commissioned": "2021-01-01",
            }
        ]
    )
    return readings, failures, machines


def test_labels_match_a_hand_computed_expectation():
    """Every row of the toy fleet is accounted for exactly.

    With 60 hours, a failure at hour 40, 5 hours of downtime and a 24-hour horizon:
      hours  0-15  negative  (failure is 25-40h away, and the window is observed)
      hours 16-39  positive  (failure is 1-24h away)
      hours 40-44  dropped   (machine is down)
      hours 45-59  dropped   (no future failure and the 24h window runs past the data)
    """
    use_project(2)
    from src.labeling import label_readings

    readings, failures, _ = _toy_fleet()
    labelled, summary = label_readings(readings, failures, horizon_hours=24, gap_hours=0)

    assert summary.rows_dropped_machine_down == 5
    assert summary.rows_dropped_censored == 15
    assert summary.rows_out == 40
    assert summary.positives == 24
    assert summary.failures_covered == 1

    hours = labelled["timestamp"].dt.hour + 24 * (labelled["timestamp"].dt.day - 1)
    positives = set(hours[labelled["label"] == 1])
    assert positives == set(range(16, 40))


def test_rows_where_the_machine_is_down_are_never_kept():
    """Sensors read near zero during repair, which is trivially separable.

    A model trained on those rows learns to detect that a machine has *already* failed.
    """
    use_project(2)
    from src.labeling import label_readings

    readings, failures, _ = _toy_fleet()
    labelled, _ = label_readings(readings, failures, horizon_hours=24)
    assert (labelled["machine_state"] == "running").all()


def test_censored_rows_are_dropped_not_assumed_negative():
    """A longer horizon must censor strictly more rows.

    If the final rows were kept and labelled 0, the negative class would be padded with
    assumptions rather than observations.
    """
    use_project(2)
    from src.labeling import label_readings

    readings, failures, _ = _toy_fleet()
    _, short = label_readings(readings, failures, horizon_hours=12)
    _, long = label_readings(readings, failures, horizon_hours=36)
    assert long.rows_dropped_censored > short.rows_dropped_censored


def test_gap_excludes_failures_that_are_too_close_to_act_on():
    """With a 6-hour gap, the last 6 hours before failure must not be positive."""
    use_project(2)
    from src.labeling import label_readings

    readings, failures, _ = _toy_fleet()
    labelled, summary = label_readings(readings, failures, horizon_hours=24, gap_hours=6)

    positive_rows = labelled[labelled["label"] == 1]
    assert positive_rows["hours_to_failure"].min() > 6
    assert summary.positives == 18          # hours 16-33 inclusive


def test_feature_pipeline_does_not_leak_future_information():
    """The empirical leakage check, on the real generator's output.

    Features are rebuilt from a series truncated at time t and compared with the same row
    built from the full series. Any feature that looked forward would differ.
    """
    use_project(2)
    from src.generate_data import generate
    from src.features import verify_causality
    from src.labeling import label_readings

    machines, readings, failures = generate(n_machines=3, n_days=20, seed=5)
    labelled, _ = label_readings(readings, failures, horizon_hours=24)

    result = verify_causality(labelled, machines, n_checks=2)
    assert result["features_compared"] > 0, "the check did not actually compare anything"
    assert result["passed"], f"future leakage detected: {result['mismatches'][:5]}"


def test_rolling_features_use_only_past_values():
    """A spike at the final hour must not change any earlier row's features."""
    use_project(2)
    from src.features import build_features, feature_columns
    from src.labeling import label_readings

    readings, failures, machines = _toy_fleet(n_hours=120, failure_hour=90)
    labelled, _ = label_readings(readings, failures, horizon_hours=24)

    baseline = build_features(labelled, machines, drop_warmup=False)

    spiked_input = labelled.copy()
    spiked_input.loc[spiked_input.index[-1], "vibration_rms"] *= 50.0
    spiked = build_features(spiked_input, machines, drop_warmup=False)

    numeric, _ = feature_columns(baseline)
    # Compare everything except the final row, which legitimately changed.
    pd.testing.assert_frame_equal(
        baseline[numeric].iloc[:-1], spiked[numeric].iloc[:-1], check_exact=False, rtol=1e-9
    )


def test_ground_truth_columns_are_excluded_from_model_inputs():
    """The simulator's health index and every future-derived column must not be features."""
    use_project(2)
    from src.features import build_features, feature_columns
    from src.labeling import label_readings

    readings, failures, machines = _toy_fleet(n_hours=120, failure_hour=90)
    labelled, _ = label_readings(readings, failures, horizon_hours=24)
    frame = build_features(labelled, machines)

    numeric, categorical = feature_columns(frame)
    model_inputs = set(numeric) | set(categorical)

    for forbidden in (
        "_health_index", "label", "hours_to_failure", "next_failure_time",
        "next_failure_mode", "next_failure_is_sudden", "machine_state", "machine_id",
    ):
        assert forbidden not in model_inputs, f"{forbidden} leaked into the feature set"


def test_time_based_split_puts_every_training_row_before_every_test_row():
    use_project(2)
    from src.features import build_features
    from src.labeling import label_readings
    from src.generate_data import generate
    from src.train import time_based_split

    machines, readings, failures = generate(n_machines=3, n_days=25, seed=6)
    labelled, _ = label_readings(readings, failures, horizon_hours=24)
    frame = build_features(labelled, machines)

    train, test, cutoff = time_based_split(frame, test_fraction=0.3)
    assert len(train) > 0 and len(test) > 0
    assert pd.to_datetime(train["timestamp"]).max() <= cutoff
    assert pd.to_datetime(test["timestamp"]).min() > cutoff


def test_machine_based_split_shares_no_machine_between_sides():
    use_project(2)
    from src.features import build_features
    from src.labeling import label_readings
    from src.generate_data import generate
    from src.train import machine_based_split

    machines, readings, failures = generate(n_machines=8, n_days=18, seed=7)
    labelled, _ = label_readings(readings, failures, horizon_hours=24)
    frame = build_features(labelled, machines)

    train, test = machine_based_split(frame, test_fraction=0.25)
    assert set(train["machine_id"]) & set(test["machine_id"]) == set()
    assert test["machine_id"].nunique() == 2
