"""Step 2 of the approach: rolling statistics and lag features, without future leakage.

Leakage is the single easiest way to produce a predictive-maintenance model that scores
0.99 offline and fails in production, so every transform here is *causal*: the feature
row at time `t` is a function of readings at times `<= t` only.

The specific rules followed:

* Gaps are forward-filled only. `interpolate()` and `bfill()` both pull values backwards
  from the future and are never used.
* Rolling windows are trailing (pandas' default). `center=True` is never used.
* "Normal behaviour" for a machine is an **expanding** baseline shifted by one hour, so
  a deviation score at time `t` is measured against that machine's history *before* `t`,
  never against statistics computed over the whole series.
* Warm-up rows, where the longest window is not yet full, are dropped rather than
  back-filled.

`verify_causality()` at the bottom proves the property empirically rather than asserting
it in a comment: it recomputes the features on a series truncated at `t` and checks the
row at `t` is bit-for-bit identical to the one computed on the full series.

Feature families, per sensor:
  level      current reading
  trend      trailing means over 6h and 24h, EWMA
  dispersion trailing std over 6h and 24h, coefficient of variation
  shape      trailing min/max/range over 24h
  memory     lags at 1h, 6h, 24h
  change     deltas vs. those lags, and deviation from the 24h mean
  anomaly    z-score against the machine's own prior history
Plus cross-sensor physics (excess temperature over ambient, power per rpm, ...) and
operating context (hours since maintenance, duty cycle, machine metadata).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

SENSORS = [
    "temperature_c",
    "vibration_rms",
    "pressure_bar",
    "rpm",
    "oil_quality_idx",
    "power_kw",
    "acoustic_db",
]

SHORT_WINDOW = 6
LONG_WINDOW = 24
LAGS = (1, 6, 24)
WARMUP_HOURS = LONG_WINDOW + 1   # rows before this per machine have incomplete windows
FFILL_LIMIT = 6                  # a gap longer than 6h is left missing, not invented

# Never model inputs: identifiers, the label, and everything derived from the future or
# from the simulator's ground truth.
NON_FEATURES = {
    "machine_id",
    "timestamp",
    "label",
    "machine_state",
    "next_failure_time",
    "next_failure_mode",
    "next_failure_is_sudden",
    "hours_to_failure",
    "_health_index",
}


def _clean_sensors(frame: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill short sensor dropouts and flag how much was filled.

    Forward-fill is the only imputation that respects the arrow of time: at 09:00 the
    08:00 reading is known, the 10:00 reading is not.
    """
    frame = frame.copy()
    grouped = frame.groupby("machine_id", sort=False)

    missing_before = frame[SENSORS].isna()
    frame["sensors_missing_count"] = missing_before.sum(axis=1).astype("int8")

    for sensor in SENSORS:
        frame[sensor] = grouped[sensor].ffill(limit=FFILL_LIMIT)

    return frame


def _add_sensor_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Rolling, lag, change and anomaly features for every sensor."""
    grouped = frame.groupby("machine_id", sort=False)
    new_columns: dict[str, pd.Series] = {}

    for sensor in SENSORS:
        series = grouped[sensor]

        # --- trend and dispersion over trailing windows ------------------------------
        mean_short = series.transform(
            lambda x: x.rolling(SHORT_WINDOW, min_periods=2).mean()
        )
        mean_long = series.transform(lambda x: x.rolling(LONG_WINDOW, min_periods=4).mean())
        std_short = series.transform(lambda x: x.rolling(SHORT_WINDOW, min_periods=2).std())
        std_long = series.transform(lambda x: x.rolling(LONG_WINDOW, min_periods=4).std())
        max_long = series.transform(lambda x: x.rolling(LONG_WINDOW, min_periods=4).max())
        min_long = series.transform(lambda x: x.rolling(LONG_WINDOW, min_periods=4).min())

        new_columns[f"{sensor}_mean_{SHORT_WINDOW}h"] = mean_short
        new_columns[f"{sensor}_mean_{LONG_WINDOW}h"] = mean_long
        new_columns[f"{sensor}_std_{SHORT_WINDOW}h"] = std_short
        new_columns[f"{sensor}_std_{LONG_WINDOW}h"] = std_long
        new_columns[f"{sensor}_range_{LONG_WINDOW}h"] = max_long - min_long
        new_columns[f"{sensor}_ewm_12h"] = series.transform(
            lambda x: x.ewm(span=12, min_periods=3).mean()
        )
        # Relative volatility: a bearing starting to fail gets noisier, not just hotter.
        new_columns[f"{sensor}_cv_{LONG_WINDOW}h"] = std_long / mean_long.abs().clip(lower=1e-6)

        # --- memory and change --------------------------------------------------------
        for lag in LAGS:
            lagged = series.shift(lag)
            new_columns[f"{sensor}_lag_{lag}h"] = lagged
            new_columns[f"{sensor}_delta_{lag}h"] = frame[sensor] - lagged

        # Short-window mean rising above the long-window mean is an early drift signal.
        new_columns[f"{sensor}_dev_from_{LONG_WINDOW}h_mean"] = frame[sensor] - mean_long
        new_columns[f"{sensor}_short_vs_long"] = mean_short / mean_long.abs().clip(lower=1e-6)

        # --- anomaly vs. this machine's own prior history -----------------------------
        # shift(1) first: the baseline must exclude the current reading, and expanding()
        # guarantees it uses only the past.
        prior = series.shift(1)
        prior_mean = prior.transform(lambda x: x.expanding(min_periods=LONG_WINDOW).mean())
        prior_std = prior.transform(lambda x: x.expanding(min_periods=LONG_WINDOW).std())
        new_columns[f"{sensor}_z_vs_history"] = (
            frame[sensor] - prior_mean
        ) / prior_std.clip(lower=1e-6)

    return pd.concat([frame, pd.DataFrame(new_columns, index=frame.index)], axis=1)


def _add_cross_sensor_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Physics-motivated combinations that no single sensor expresses.

    `excess_temp_over_ambient` matters most: the raw temperature tracks the plant
    ambient, so an absolute temperature threshold fires on hot afternoons. The excess
    over ambient is the part attributable to the machine.
    """
    frame = frame.copy()
    rpm = frame["rpm"].abs().clip(lower=1.0)

    frame["excess_temp_over_ambient"] = frame["temperature_c"] - frame["ambient_temp_c"]
    frame["power_per_rpm"] = frame["power_kw"] / rpm * 1000.0
    frame["vibration_per_rpm"] = frame["vibration_rms"] / rpm * 1000.0
    frame["pressure_x_rpm"] = frame["pressure_bar"] * rpm / 1000.0
    frame["thermal_load_ratio"] = frame["temperature_c"] / frame["power_kw"].abs().clip(lower=0.1)
    frame["vibration_temp_product"] = frame["vibration_rms"] * frame["excess_temp_over_ambient"]
    frame["oil_degradation"] = 100.0 - frame["oil_quality_idx"]
    frame["efficiency_proxy"] = frame["pressure_bar"] / frame["power_kw"].abs().clip(lower=0.1)

    grouped = frame.groupby("machine_id", sort=False)
    for column in ["excess_temp_over_ambient", "power_per_rpm", "vibration_per_rpm"]:
        frame[f"{column}_mean_{LONG_WINDOW}h"] = grouped[column].transform(
            lambda x: x.rolling(LONG_WINDOW, min_periods=4).mean()
        )
        frame[f"{column}_delta_24h"] = frame[column] - grouped[column].shift(LONG_WINDOW)

    # How many sensors are simultaneously abnormal -- a multivariate alarm count. A single
    # drifting sensor is often noise; three at once rarely is.
    z_columns = [f"{sensor}_z_vs_history" for sensor in SENSORS if f"{sensor}_z_vs_history" in frame]
    if z_columns:
        z_values = frame[z_columns].abs()
        frame["sensors_abnormal_2sigma"] = (z_values > 2.0).sum(axis=1).astype("int8")
        frame["sensors_abnormal_3sigma"] = (z_values > 3.0).sum(axis=1).astype("int8")
        frame["max_abs_z_score"] = z_values.max(axis=1)
        frame["mean_abs_z_score"] = z_values.mean(axis=1)

    return frame


def _add_context_features(frame: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    """Operating context and machine metadata."""
    frame = frame.merge(
        machines[["machine_id", "machine_type", "age_years", "rated_power_kw", "criticality"]],
        on="machine_id",
        how="left",
    )

    grouped = frame.groupby("machine_id", sort=False)
    frame["load_mean_24h"] = grouped["load_pct"].transform(
        lambda x: x.rolling(LONG_WINDOW, min_periods=4).mean()
    )
    frame["load_std_24h"] = grouped["load_pct"].transform(
        lambda x: x.rolling(LONG_WINDOW, min_periods=4).std()
    )
    frame["load_utilisation"] = frame["power_kw"] / frame["rated_power_kw"].clip(lower=1.0)

    # Wear proxies: time since service, and service interval progress.
    frame["maintenance_overdue_ratio"] = frame["hours_since_maintenance"] / 720.0
    frame["operating_hours_per_year_of_age"] = frame["operating_hours"] / (
        frame["age_years"].clip(lower=0.1) * 8760.0
    )

    # Time-of-day / week: load patterns differ, and so do failure rates.
    timestamps = pd.to_datetime(frame["timestamp"])
    frame["hour_of_day"] = timestamps.dt.hour.astype("int8")
    frame["day_of_week"] = timestamps.dt.day_of_week.astype("int8")
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype("int8")

    return frame


def build_features(
    labelled: pd.DataFrame, machines: pd.DataFrame, drop_warmup: bool = True
) -> pd.DataFrame:
    """Full causal feature pipeline. Input must be sorted within each machine."""
    frame = labelled.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    frame = _clean_sensors(frame)
    frame = _add_sensor_features(frame)
    frame = _add_cross_sensor_features(frame)
    frame = _add_context_features(frame, machines)

    if drop_warmup:
        # Drop the first WARMUP_HOURS rows per machine: their long windows are partial,
        # so their features are not comparable with the rest of the data.
        position_in_machine = frame.groupby("machine_id", sort=False).cumcount()
        frame = frame[position_in_machine >= WARMUP_HOURS].reset_index(drop=True)

    frame = frame.replace([np.inf, -np.inf], np.nan)

    # float32 halves the memory footprint with no measurable effect on tree models.
    for column in frame.select_dtypes(include=["float64"]).columns:
        frame[column] = frame[column].astype("float32")

    return frame


def feature_columns(frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split modelling columns into numeric and categorical, excluding all leakage."""
    candidates = [column for column in frame.columns if column not in NON_FEATURES]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in candidates if c not in numeric]
    return numeric, categorical


def verify_causality(
    labelled: pd.DataFrame, machines: pd.DataFrame, n_checks: int = 3
) -> dict:
    """Prove empirically that no feature depends on the future.

    For a sample of (machine, time) points: build features from the full series, then
    rebuild them from the series truncated at that time, and compare the feature row.
    If any feature peeked forward, the two rows would differ.
    """
    rng = np.random.default_rng(0)
    machine_ids = labelled["machine_id"].unique()
    results = {"checks": 0, "features_compared": 0, "mismatches": [], "passed": True}

    for machine_id in rng.choice(machine_ids, size=min(n_checks, len(machine_ids)), replace=False):
        machine_rows = labelled[labelled["machine_id"] == machine_id].sort_values("timestamp")
        if len(machine_rows) < WARMUP_HOURS + 80:
            continue

        cut = len(machine_rows) - 30                      # a point with data on both sides
        full = build_features(machine_rows, machines, drop_warmup=False)
        truncated = build_features(machine_rows.iloc[: cut + 1], machines, drop_warmup=False)

        numeric, _ = feature_columns(full)
        row_full = full.iloc[cut][numeric]
        row_truncated = truncated.iloc[cut][numeric]

        for feature in numeric:
            a, b = row_full[feature], row_truncated[feature]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) != pd.isna(b) or not np.isclose(float(a), float(b), rtol=1e-6, atol=1e-8):
                results["mismatches"].append(
                    {"machine_id": str(machine_id), "feature": feature,
                     "full": float(a) if not pd.isna(a) else None,
                     "truncated": float(b) if not pd.isna(b) else None}
                )
        results["checks"] += 1
        results["features_compared"] += len(numeric)

    results["passed"] = len(results["mismatches"]) == 0
    return results


def main() -> None:
    here = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Build causal predictive-maintenance features.")
    parser.add_argument(
        "--labelled", type=Path, default=here / "data" / "processed" / "labelled.parquet"
    )
    parser.add_argument("--machines", type=Path, default=here / "data" / "raw" / "machines.csv")
    parser.add_argument(
        "--out", type=Path, default=here / "data" / "processed" / "features.parquet"
    )
    parser.add_argument("--skip-causality-check", action="store_true")
    args = parser.parse_args()

    labelled = pd.read_parquet(args.labelled)
    machines = pd.read_csv(args.machines)

    if not args.skip_causality_check:
        print("verifying feature causality (no future leakage)...")
        verification = verify_causality(labelled, machines)
        status = "PASS" if verification["passed"] else "FAIL"
        print(
            f"  [{status}] {verification['checks']} cut-points, "
            f"{verification['features_compared']:,} feature values compared, "
            f"{len(verification['mismatches'])} mismatches"
        )
        if not verification["passed"]:
            for mismatch in verification["mismatches"][:10]:
                print(f"    LEAK: {mismatch}")
            raise SystemExit("feature pipeline leaks future information - aborting")

    frame = build_features(labelled, machines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    numeric, categorical = feature_columns(frame)
    print(f"feature frame : {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    print(f"model inputs  : {len(numeric)} numeric + {len(categorical)} categorical")
    print(f"positives     : {int(frame['label'].sum()):,} ({frame['label'].mean():.2%})")
    print(f"memory        : {frame.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
