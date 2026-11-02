"""Step 3 of the approach: define the failure prediction window and label every row.

The label answers one question, asked once per machine-hour:

    "Will this machine fail within the next H hours?"

Three correctness rules are enforced here, because each one silently inflates the
reported score if it is skipped:

1. **Rows where the machine is already down are dropped.** Their sensors read near zero,
   which is trivially separable. A model trained on them scores brilliantly and predicts
   nothing useful -- it has learned to detect that a machine has *already* failed.

2. **The last H hours of every machine's history are dropped.** Their future window
   extends past the end of the data, so a "no failure" label there is not an observation,
   it is an assumption. This is right-censoring, and keeping those rows biases the
   negative class.

3. **An optional lead-time gap.** Predicting a failure one minute before it happens is
   useless -- nobody can act on it. Setting `--gap` to, say, 4 hours labels only failures
   that are between 4 and H hours away, so every alert leaves time to respond.

The module also records `time_to_failure` and whether the upcoming failure was a sudden
one. Neither is a model input; both are needed in evaluation to measure how much warning
the model actually bought.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HORIZON_HOURS = 24
DEFAULT_GAP_HOURS = 0


@dataclass
class LabelSummary:
    """Audit trail for the labelling step."""

    horizon_hours: int
    gap_hours: int
    rows_in: int
    rows_dropped_machine_down: int
    rows_dropped_censored: int
    rows_out: int
    positives: int
    positive_rate: float
    failures_covered: int
    failures_total: int

    def as_dict(self) -> dict:
        return {
            "horizon_hours": self.horizon_hours,
            "gap_hours": self.gap_hours,
            "rows_in": self.rows_in,
            "rows_dropped_machine_down": self.rows_dropped_machine_down,
            "rows_dropped_censored": self.rows_dropped_censored,
            "rows_out": self.rows_out,
            "positives": self.positives,
            "positive_rate": round(self.positive_rate, 5),
            "imbalance_ratio": round((self.rows_out - self.positives) / max(self.positives, 1), 1),
            "failures_covered": self.failures_covered,
            "failures_total": self.failures_total,
        }


def label_readings(
    readings: pd.DataFrame,
    failures: pd.DataFrame,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    gap_hours: int = DEFAULT_GAP_HOURS,
) -> tuple[pd.DataFrame, LabelSummary]:
    """Attach the binary failure-window label plus time-to-failure diagnostics."""
    readings = readings.copy()
    readings["timestamp"] = pd.to_datetime(readings["timestamp"])
    rows_in = len(readings)

    if len(failures):
        failures = failures.copy()
        failures["failure_time"] = pd.to_datetime(failures["failure_time"])

    horizon = pd.Timedelta(hours=horizon_hours)
    gap = pd.Timedelta(hours=gap_hours)

    labelled_parts = []
    failures_covered = 0

    for machine_id, machine_rows in readings.groupby("machine_id", sort=False):
        machine_rows = machine_rows.sort_values("timestamp").reset_index(drop=True)
        series_end = machine_rows["timestamp"].iloc[-1]

        machine_failures = (
            failures[failures["machine_id"] == machine_id].sort_values("failure_time")
            if len(failures)
            else pd.DataFrame(columns=["failure_time", "is_sudden", "failure_mode"])
        )

        if len(machine_failures):
            failure_times = machine_failures["failure_time"].to_numpy()
            sudden_flags = machine_failures["is_sudden"].to_numpy()
            modes = machine_failures["failure_mode"].to_numpy()

            # Index of the first failure at or after each reading.
            position = np.searchsorted(
                failure_times, machine_rows["timestamp"].to_numpy(), side="left"
            )
            has_future_failure = position < len(failure_times)
            safe_position = np.clip(position, 0, len(failure_times) - 1)

            next_failure = pd.Series(
                np.where(
                    has_future_failure,
                    failure_times[safe_position],
                    np.datetime64("NaT", "ns"),
                ),
                index=machine_rows.index,
            )
            machine_rows["next_failure_time"] = next_failure
            machine_rows["next_failure_is_sudden"] = np.where(
                has_future_failure, sudden_flags[safe_position], False
            )
            machine_rows["next_failure_mode"] = np.where(
                has_future_failure, modes[safe_position], "none"
            )
        else:
            machine_rows["next_failure_time"] = pd.NaT
            machine_rows["next_failure_is_sudden"] = False
            machine_rows["next_failure_mode"] = "none"

        time_to_failure = machine_rows["next_failure_time"] - machine_rows["timestamp"]
        machine_rows["hours_to_failure"] = time_to_failure.dt.total_seconds() / 3600.0

        # Rule 3: the alert window is (gap, horizon].
        in_window = (time_to_failure > gap) & (time_to_failure <= horizon)
        machine_rows["label"] = in_window.fillna(False).astype(int)

        # Rule 2: a negative label is only observed if the whole window fits in the data.
        window_observed = (machine_rows["timestamp"] + horizon) <= series_end
        machine_rows["_censored"] = (~in_window) & (~window_observed)

        failures_covered += int(
            machine_rows.loc[machine_rows["label"] == 1, "next_failure_time"].nunique()
        )
        labelled_parts.append(machine_rows)

    labelled = pd.concat(labelled_parts, ignore_index=True)

    # Rule 1: drop rows where the machine was not running.
    down_mask = labelled["machine_state"] != "running"
    dropped_down = int(down_mask.sum())
    labelled = labelled[~down_mask]

    censored_mask = labelled["_censored"]
    dropped_censored = int(censored_mask.sum())
    labelled = labelled[~censored_mask].drop(columns=["_censored"]).reset_index(drop=True)

    summary = LabelSummary(
        horizon_hours=horizon_hours,
        gap_hours=gap_hours,
        rows_in=rows_in,
        rows_dropped_machine_down=dropped_down,
        rows_dropped_censored=dropped_censored,
        rows_out=len(labelled),
        positives=int(labelled["label"].sum()),
        positive_rate=float(labelled["label"].mean()) if len(labelled) else 0.0,
        failures_covered=failures_covered,
        failures_total=int(len(failures)),
    )
    return labelled, summary


def main() -> None:
    here = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Label sensor readings with a failure window.")
    parser.add_argument("--raw", type=Path, default=here / "data" / "raw")
    parser.add_argument(
        "--out", type=Path, default=here / "data" / "processed" / "labelled.parquet"
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument("--gap", type=int, default=DEFAULT_GAP_HOURS)
    args = parser.parse_args()

    readings = pd.read_csv(args.raw / "sensor_readings.csv")
    failures = pd.read_csv(args.raw / "failure_events.csv")

    labelled, summary = label_readings(readings, failures, args.horizon, args.gap)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_parquet(args.out, index=False)

    info = summary.as_dict()
    print(f"prediction window : failure within {args.horizon}h (gap {args.gap}h)")
    print(f"rows              : {info['rows_in']:,} -> {info['rows_out']:,}")
    print(f"  dropped (down)  : {info['rows_dropped_machine_down']:,} machine not running")
    print(f"  dropped (censor): {info['rows_dropped_censored']:,} future window not observed")
    print(f"positives         : {info['positives']:,} ({info['positive_rate']:.2%})")
    print(f"imbalance ratio   : {info['imbalance_ratio']}:1")
    print(f"failures covered  : {info['failures_covered']}/{info['failures_total']}")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
