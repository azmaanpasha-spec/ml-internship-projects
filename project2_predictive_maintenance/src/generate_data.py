"""Simulate an industrial fleet: hourly sensor telemetry, maintenance history, failures.

This is a physics-flavoured simulator rather than random noise with labels bolted on,
because the modelling problem is only meaningful if the signal has realistic structure:

* **Distinct failure modes.** A bearing failure shows up in vibration and temperature; a
  seal leak shows up in pressure and oil quality. A model therefore has to use the
  sensors jointly, and single-sensor thresholds are provably insufficient.
* **Gradual degradation ramps.** Each failure has an onset time, and the sensor
  signature grows as `progress ** 1.8` -- accelerating, so the early warning is faint.
  This is what makes the prediction horizon a real design choice.
* **Some failures are sudden.** A configurable share have an onset of only a few hours.
  They are close to unpredictable, which puts an honest ceiling on achievable recall --
  a model claiming 99% recall on this data would be a bug, not a result.
* **Dirty telemetry.** Sensors drop out, stick at their last value, and spike. The
  machine is also offline during repairs, and those rows must not be scored.

Outputs three CSVs, mirroring how this data really arrives (an asset register, a
telemetry historian, and a work-order log):
    machines.csv        one row per machine
    sensor_readings.csv one row per machine-hour
    failure_events.csv  one row per failure, with its true onset time

Run:  python -m src.generate_data
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SEED = 7
DEFAULT_MACHINES = 80
DEFAULT_DAYS = 90
START_TIME = pd.Timestamp("2026-04-01 00:00:00")

SENSORS = [
    "temperature_c",
    "vibration_rms",
    "pressure_bar",
    "rpm",
    "oil_quality_idx",
    "power_kw",
    "acoustic_db",
]


@dataclass(frozen=True)
class MachineProfile:
    """Nominal operating point and wear characteristics for a class of machine."""

    name: str
    baseline: Dict[str, float]
    noise: Dict[str, float]
    load_sensitivity: float      # how strongly sensor levels track the duty cycle
    base_failure_rate: float     # expected failures per 1000 operating hours
    rated_power_kw: float


PROFILES: List[MachineProfile] = [
    MachineProfile(
        name="Centrifugal Pump",
        baseline={"temperature_c": 55.0, "vibration_rms": 2.0, "pressure_bar": 6.5,
                  "rpm": 1480.0, "oil_quality_idx": 95.0, "power_kw": 18.0, "acoustic_db": 72.0},
        noise={"temperature_c": 1.3, "vibration_rms": 0.12, "pressure_bar": 0.18,
               "rpm": 9.0, "oil_quality_idx": 0.6, "power_kw": 0.7, "acoustic_db": 0.9},
        load_sensitivity=0.22, base_failure_rate=1.5, rated_power_kw=22.0,
    ),
    MachineProfile(
        name="Screw Compressor",
        baseline={"temperature_c": 70.0, "vibration_rms": 3.2, "pressure_bar": 9.0,
                  "rpm": 2950.0, "oil_quality_idx": 92.0, "power_kw": 45.0, "acoustic_db": 82.0},
        noise={"temperature_c": 1.8, "vibration_rms": 0.20, "pressure_bar": 0.25,
               "rpm": 14.0, "oil_quality_idx": 0.8, "power_kw": 1.6, "acoustic_db": 1.1},
        load_sensitivity=0.30, base_failure_rate=2.3, rated_power_kw=55.0,
    ),
    MachineProfile(
        name="Belt Conveyor",
        baseline={"temperature_c": 42.0, "vibration_rms": 1.4, "pressure_bar": 2.0,
                  "rpm": 950.0, "oil_quality_idx": 97.0, "power_kw": 11.0, "acoustic_db": 65.0},
        noise={"temperature_c": 1.1, "vibration_rms": 0.10, "pressure_bar": 0.10,
               "rpm": 7.0, "oil_quality_idx": 0.5, "power_kw": 0.5, "acoustic_db": 0.8},
        load_sensitivity=0.18, base_failure_rate=1.1, rated_power_kw=15.0,
    ),
    MachineProfile(
        name="Gas Turbine",
        baseline={"temperature_c": 85.0, "vibration_rms": 2.6, "pressure_bar": 12.0,
                  "rpm": 3600.0, "oil_quality_idx": 90.0, "power_kw": 120.0, "acoustic_db": 88.0},
        noise={"temperature_c": 2.2, "vibration_rms": 0.16, "pressure_bar": 0.32,
               "rpm": 18.0, "oil_quality_idx": 0.9, "power_kw": 3.5, "acoustic_db": 1.3},
        load_sensitivity=0.35, base_failure_rate=2.8, rated_power_kw=150.0,
    ),
]

# Relative sensor deviation at the moment of failure, per failure mode. Positive values
# push the reading up, negative push it down; they are scaled by the degradation ramp.
FAILURE_MODES: Dict[str, Dict[str, float]] = {
    "bearing_wear": {"vibration_rms": 1.80, "temperature_c": 0.22, "acoustic_db": 0.12,
                     "power_kw": 0.08},
    "seal_leak": {"pressure_bar": -0.35, "oil_quality_idx": -0.30, "temperature_c": 0.08},
    "overheating": {"temperature_c": 0.45, "power_kw": 0.18, "oil_quality_idx": -0.15},
    "motor_fault": {"power_kw": 0.30, "acoustic_db": 0.20, "rpm": -0.12, "temperature_c": 0.15},
    "lubrication_failure": {"oil_quality_idx": -0.45, "vibration_rms": 0.90,
                            "temperature_c": 0.25},
}
MODE_WEIGHTS = [0.30, 0.20, 0.18, 0.17, 0.15]

# A fifth of failures give almost no warning. Keeping them in the data is the honest
# choice: it caps recall at a realistic level instead of inventing a perfect model.
SUDDEN_FAILURE_SHARE = 0.20
GRADUAL_ONSET_HOURS = (48, 220)
SUDDEN_ONSET_HOURS = (2, 8)

PREVENTIVE_INTERVAL_HOURS = 720   # scheduled service roughly every 30 days
MIN_FAILURE_SPACING_HOURS = 180


def _duty_cycle(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Load factor in [0, 1]: daily shift pattern, lighter at weekends, plus drift."""
    hours = timestamps.hour.to_numpy()
    weekday = timestamps.day_of_week.to_numpy()

    shift = 0.55 + 0.35 * np.sin((hours - 6) / 24 * 2 * np.pi)       # peaks mid-shift
    weekend = np.where(weekday >= 5, 0.60, 1.0)
    drift = 1.0 + 0.05 * np.sin(np.arange(len(timestamps)) / 400.0)
    jitter = rng.normal(1.0, 0.06, len(timestamps))
    return np.clip(shift * weekend * drift * jitter, 0.08, 1.25)


def _ambient_temperature(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Plant ambient temperature -- a confounder the model must look past."""
    hours = np.arange(len(timestamps))
    daily = 4.5 * np.sin((timestamps.hour.to_numpy() - 4) / 24 * 2 * np.pi)
    seasonal = 3.0 * np.sin(hours / (24 * 45) * 2 * np.pi)
    return 22.0 + daily + seasonal + rng.normal(0, 0.8, len(timestamps))


def _ar1_noise(n: int, sigma: float, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Autocorrelated noise -- real sensor noise is not independent hour to hour."""
    innovations = rng.normal(0, sigma * np.sqrt(1 - rho**2), n)
    series = np.empty(n)
    series[0] = rng.normal(0, sigma)
    for i in range(1, n):
        series[i] = rho * series[i - 1] + innovations[i]
    return series


def _schedule_failures(
    rng: np.random.Generator, n_hours: int, profile: MachineProfile, age_years: float
) -> List[dict]:
    """Pick failure times, modes and onset lead times for one machine.

    Older machines fail more often (a simple linear age multiplier). Failures are spaced
    so that degradation episodes do not overlap, which keeps the labels unambiguous.
    """
    age_multiplier = 1.0 + 0.11 * age_years
    expected = profile.base_failure_rate * age_multiplier * n_hours / 1000.0
    n_failures = rng.poisson(expected)
    if n_failures == 0:
        return []

    earliest = GRADUAL_ONSET_HOURS[1] + 24
    candidates = rng.uniform(earliest, n_hours - 12, size=n_failures * 4)
    chosen: List[float] = []
    for candidate in np.sort(candidates):
        if all(abs(candidate - existing) >= MIN_FAILURE_SPACING_HOURS for existing in chosen):
            chosen.append(float(candidate))
        if len(chosen) == n_failures:
            break

    failures = []
    for failure_hour in chosen:
        sudden = rng.random() < SUDDEN_FAILURE_SHARE
        low, high = SUDDEN_ONSET_HOURS if sudden else GRADUAL_ONSET_HOURS
        onset_hours = float(rng.integers(low, high + 1))
        failures.append(
            {
                "failure_hour": int(round(failure_hour)),
                "mode": str(rng.choice(list(FAILURE_MODES), p=MODE_WEIGHTS)),
                "onset_hours": onset_hours,
                "sudden": bool(sudden),
                "severity": float(np.clip(rng.normal(1.0, 0.22), 0.45, 1.8)),
                "downtime_hours": int(rng.integers(4, 25)),
            }
        )
    return failures


def _degradation_signatures(
    n_hours: int, failures: List[dict]
) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    """Build the per-sensor relative deviation caused by every degradation episode.

    Also returns the true health index (1.0 = healthy, 0.0 = at failure), which is kept
    out of the modelling features and used only for diagnostics in the report.
    """
    signatures = {sensor: np.zeros(n_hours) for sensor in SENSORS}
    health = np.ones(n_hours)

    for failure in failures:
        failure_hour = failure["failure_hour"]
        onset = int(failure["onset_hours"])
        start = max(failure_hour - onset, 0)
        if failure_hour <= start:
            continue

        span = np.arange(start, failure_hour)
        progress = (span - start) / max(failure_hour - start, 1)
        # Accelerating ramp: faint early, obvious late. Sudden failures ramp linearly
        # over a handful of hours, so there is almost nothing to detect in advance.
        shape = progress if failure["sudden"] else progress**1.8
        ramp = shape * failure["severity"]

        for sensor, magnitude in FAILURE_MODES[failure["mode"]].items():
            signatures[sensor][span] += magnitude * ramp
        health[span] = np.minimum(health[span], 1.0 - shape)

    return signatures, health


def _simulate_machine(
    rng: np.random.Generator,
    machine_id: str,
    profile: MachineProfile,
    age_years: float,
    n_hours: int,
) -> tuple[pd.DataFrame, List[dict]]:
    """Generate the full hourly history for one machine."""
    timestamps = pd.date_range(START_TIME, periods=n_hours, freq="h")
    load = _duty_cycle(timestamps, rng)
    ambient = _ambient_temperature(timestamps, rng)

    failures = _schedule_failures(rng, n_hours, profile, age_years)
    signatures, health = _degradation_signatures(n_hours, failures)

    # --- machine state: running, or down for repair after a failure ------------------
    state = np.full(n_hours, "running", dtype=object)
    for failure in failures:
        down_from = failure["failure_hour"]
        down_to = min(down_from + failure["downtime_hours"], n_hours)
        state[down_from:down_to] = "down"

    # --- maintenance events reset the wear clock -------------------------------------
    maintenance_hours = list(range(PREVENTIVE_INTERVAL_HOURS, n_hours, PREVENTIVE_INTERVAL_HOURS))
    repair_hours = [
        min(f["failure_hour"] + f["downtime_hours"], n_hours - 1) for f in failures
    ]
    reset_points = sorted(set(maintenance_hours + repair_hours))

    hours_since_maintenance = np.zeros(n_hours)
    counter = float(rng.integers(0, PREVENTIVE_INTERVAL_HOURS))
    reset_set = set(reset_points)
    for t in range(n_hours):
        if t in reset_set:
            counter = 0.0
        hours_since_maintenance[t] = counter
        counter += 1.0

    # Slow wear between services: a small drift that the model can learn to discount.
    wear = (hours_since_maintenance / PREVENTIVE_INTERVAL_HOURS) * (1.0 + 0.1 * age_years)

    # --- sensor readings ---------------------------------------------------------------
    readings: Dict[str, np.ndarray] = {}
    for sensor in SENSORS:
        baseline = profile.baseline[sensor]
        load_effect = 1.0 + profile.load_sensitivity * (load - 0.75)
        # Temperature additionally tracks the ambient, which is a genuine confounder.
        ambient_effect = 0.55 * (ambient - 22.0) if sensor == "temperature_c" else 0.0
        wear_effect = 1.0 + np.array(
            {"vibration_rms": 0.06, "temperature_c": 0.02, "oil_quality_idx": -0.03}.get(sensor, 0.0)
        ) * wear

        value = (
            baseline * load_effect * wear_effect * (1.0 + signatures[sensor])
            + ambient_effect
            + _ar1_noise(n_hours, profile.noise[sensor], 0.72, rng)
        )
        readings[sensor] = value

    frame = pd.DataFrame(
        {
            "machine_id": machine_id,
            "timestamp": timestamps,
            **{sensor: np.round(readings[sensor], 3) for sensor in SENSORS},
            "ambient_temp_c": np.round(ambient, 2),
            "load_pct": np.round(load * 100, 2),
            "machine_state": state,
            "hours_since_maintenance": hours_since_maintenance.astype(int),
            "_health_index": np.round(np.clip(health, 0.0, 1.0), 4),
        }
    )

    # Operating hours accumulate only while the machine is actually running.
    frame["operating_hours"] = (frame["machine_state"] == "running").cumsum() + int(
        age_years * 2400
    )

    # While a machine is down the sensors read near-zero, not their nominal values.
    down = frame["machine_state"] == "down"
    for sensor in SENSORS:
        frame.loc[down, sensor] = np.round(
            frame.loc[down, sensor] * rng.uniform(0.02, 0.08, int(down.sum())), 3
        )

    event_rows = [
        {
            "machine_id": machine_id,
            "failure_time": timestamps[f["failure_hour"]],
            "failure_mode": f["mode"],
            "onset_hours": f["onset_hours"],
            "is_sudden": f["sudden"],
            "severity": round(f["severity"], 3),
            "downtime_hours": f["downtime_hours"],
            "repair_cost": round(
                float(profile.rated_power_kw * 38 * f["severity"] + f["downtime_hours"] * 420), 2
            ),
        }
        for f in failures
    ]
    return frame, event_rows


def _inject_sensor_faults(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Dropouts, stuck sensors and spikes -- the three faults every historian contains."""
    frame = frame.copy()
    n = len(frame)

    for sensor in SENSORS:
        # 1. Random dropouts.
        frame.loc[rng.random(n) < 0.004, sensor] = np.nan

        # 2. Stuck sensor: holds its last value for a stretch of hours.
        for start in np.flatnonzero(rng.random(n) < 0.00035):
            length = int(rng.integers(4, 14))
            end = min(start + length, n)
            if start > 0:
                frame.iloc[start:end, frame.columns.get_loc(sensor)] = frame.iloc[start - 1][sensor]

        # 3. Transient spikes that are instrument faults, not machine faults.
        spikes = rng.random(n) < 0.0008
        frame.loc[spikes, sensor] = frame.loc[spikes, sensor] * rng.uniform(1.8, 3.2, int(spikes.sum()))

    return frame


def generate(
    n_machines: int = DEFAULT_MACHINES, n_days: int = DEFAULT_DAYS, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate the fleet. Returns (machines, readings, failures)."""
    rng = np.random.default_rng(seed)
    n_hours = n_days * 24

    machine_rows, reading_frames, failure_rows = [], [], []
    for index in range(n_machines):
        machine_id = f"MCH-{index + 1:03d}"
        profile = PROFILES[index % len(PROFILES)]
        age_years = float(np.round(rng.uniform(0.5, 14.0), 1))

        frame, events = _simulate_machine(rng, machine_id, profile, age_years, n_hours)
        reading_frames.append(_inject_sensor_faults(frame, rng))
        failure_rows.extend(events)

        machine_rows.append(
            {
                "machine_id": machine_id,
                "machine_type": profile.name,
                "age_years": age_years,
                "rated_power_kw": profile.rated_power_kw,
                "site": f"Plant-{chr(65 + index % 4)}",
                "criticality": str(rng.choice(["High", "Medium", "Low"], p=[0.25, 0.5, 0.25])),
                "commissioned": (START_TIME - pd.Timedelta(days=age_years * 365)).date(),
            }
        )

    machines = pd.DataFrame(machine_rows)
    readings = pd.concat(reading_frames, ignore_index=True)
    failures = pd.DataFrame(failure_rows)
    return machines, readings, failures


def main() -> None:
    here = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Simulate industrial fleet telemetry.")
    parser.add_argument("--machines", type=int, default=DEFAULT_MACHINES)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=here / "data" / "raw")
    args = parser.parse_args()

    machines, readings, failures = generate(args.machines, args.days, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    machines.to_csv(args.out / "machines.csv", index=False)
    readings.to_csv(args.out / "sensor_readings.csv", index=False)
    failures.to_csv(args.out / "failure_events.csv", index=False)

    sudden = int(failures["is_sudden"].sum()) if len(failures) else 0
    print(f"machines        : {len(machines)}")
    print(f"sensor readings : {len(readings):,} rows ({args.days} days hourly)")
    print(f"failures        : {len(failures)} ({sudden} sudden / near-unpredictable)")
    if len(failures):
        print(f"modes           : {failures['failure_mode'].value_counts().to_dict()}")
        print(f"mean onset      : {failures['onset_hours'].mean():.1f} h")
        print(f"downtime        : {failures['downtime_hours'].sum()} h total")
    print(f"missing cells   : {int(readings[SENSORS].isna().sum().sum()):,}")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
