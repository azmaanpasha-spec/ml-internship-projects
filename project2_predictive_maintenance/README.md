# Project 2 — Predictive Maintenance for Industrial Equipment

Predict whether a machine will fail within the next 24 hours from sensor telemetry and
maintenance history, and turn that into an alerting policy a maintenance team can
actually run.

> **This project is submitted as the internship report.**
> The full write-up is in **[`REPORT.md`](REPORT.md)**. This README is the technical
> quick reference.

---

## Results

Held-out **time period** (the last 30% of the calendar, 48,598 machine-hours, 5.20%
positive). No row in training postdates any row in test.

| Model | PR-AUC | ROC-AUC | Precision | Recall | Brier | ECE |
|---|---|---|---|---|---|---|
| Logistic regression | 0.5652 | 0.9109 | 0.290 | 0.847 | 0.1151 | 0.2597 |
| **Random forest** | **0.6894** | **0.9568** | 0.462 | 0.843 | 0.0819 | 0.2155 |
| LightGBM | 0.6876 | 0.9502 | 0.481 | 0.777 | 0.0410 | 0.0566 |
| XGBoost | 0.6702 | 0.9490 | 0.357 | 0.839 | 0.0636 | 0.1138 |
| Random forest + isotonic calibration | 0.6716 | 0.9506 | 0.693 | 0.539 | **0.0268** | **0.0161** |

### The headline numbers

At the **cost-optimal alert threshold of 0.15**:

| | |
|---|---|
| **Failures caught** | **104 of 107 — 97.2%** |
| Median warning | 24.0 hours |
| Alert burden | **0.53 alert episodes per machine per week** |
| Expected cost | 594,000 vs 1,926,000 reactive — **1,332,000 saved** |

### The finding that matters most

Detection rate alone is a misleading summary. Split by how fast the failure developed:

| Onset type | Failures | Caught | Rate | Mean lead time | Median |
|---|---|---|---|---|---|
| Gradual (48–220 h onset) | 89 | 87 | 97.8% | **23.3 h** | 24 h |
| Sudden (2–8 h onset) | 18 | 17 | 94.4% | **2.4 h** | 2 h (max 5 h) |

Sudden failures are caught at *almost the same rate* — but two hours before the event
instead of a full day. The alert window runs right up to the failure, so a model can
catch a sudden failure late and still score it as "detected". **Detection rate hides the
entire difference; lead time is the metric that reflects whether anyone could have
acted.** This is why `evaluate.py` reports lead time per group, and why the report leads
with it.

By failure mode (all at threshold 0.15):

| Mode | Caught | Rate | Mean lead | Median lead |
|---|---|---|---|---|
| bearing_wear | 29/29 | 100.0% | 21.4 h | 24 h |
| lubrication_failure | 13/13 | 100.0% | 17.2 h | 24 h |
| seal_leak | 28/29 | 96.5% | 19.6 h | 24 h |
| motor_fault | 19/20 | 95.0% | 20.3 h | 24 h |
| overheating | 15/16 | 93.8% | 19.1 h | 24 h |

Bearing wear and lubrication failure are caught completely — both drive vibration, the
sensor with the largest relative deviation. Overheating is the weakest at 93.8%, which is
consistent with temperature being the one sensor that also tracks a confounder (plant
ambient), so a genuine thermal fault has to clear a noisier baseline.

### Generalisation to unseen machines

A second, independent split holds out 20 whole machines rather than a time period:
**PR-AUC 0.6344** (vs 0.6894 on the time split), ROC-AUC 0.9220. An ~8% relative drop,
which is the honest answer to "will this work on a machine the model has never seen?"

### Imbalance ablation — a result that contradicted the usual advice

Both strategies, same tuned hyperparameters, same test set:

| Strategy | Rows used | PR-AUC | Brier |
|---|---|---|---|
| `class_weight="balanced"` | 113,596 | 0.6894 | 0.0819 |
| Random undersampling (1:3) | 36,540 | 0.6853 | **0.0393** |

The common claim is that class weighting preserves calibration where resampling destroys
it. **That is not what happened.** PR-AUC was effectively tied, and the undersampled model
was *better* calibrated — weighting skews probabilities towards 0.5 just as surely.
Weighting was kept because it uses all 113,596 rows rather than discarding 68% of the
negatives, and because calibration is restored afterwards anyway: isotonic calibration cut
the Brier score from 0.0819 to 0.0268 and ECE from 0.2155 to 0.0161, a **13× improvement
in calibration error**.

---

## How to run

```bash
pip install -r ../requirements.txt

python -m src.generate_data   # 80 machines x 90 days hourly = 172,800 readings
python -m src.labeling        # failure-window labels + censoring rules
python -m src.features        # 152 causal features (+ empirical leakage check)
python -m src.train           # 4 models, time-aware CV, ablation, calibration
python -m src.evaluate        # PR/ROC, lead time, cost analysis, SHAP
python -m src.monitor         # fleet risk view + alerts

streamlit run app.py          # monitoring dashboard
```

Roughly 10 minutes end to end; `src.train` dominates.

---

## Approach

### 1. Data (`generate_data.py`)

A physics-flavoured simulator, not noise with labels attached. 80 machines across four
types (pump, compressor, conveyor, turbine), 90 days of hourly telemetry on seven sensors,
486 failures across five modes.

What makes the problem non-trivial by construction:

- **Mode-specific signatures.** A bearing failure drives vibration (+180%) and temperature
  (+22%); a seal leak drives pressure (−35%) and oil quality (−30%). A model must use the
  sensors *jointly*; single-sensor thresholds are provably insufficient.
- **Accelerating degradation ramps.** The signature grows as `progress^1.8`, so the early
  warning is faint and the horizon is a real design choice.
- **20% sudden failures** with 2–8 hour onsets — the honest ceiling on lead time.
- **Ambient temperature as a confounder**, which is why `excess_temp_over_ambient` exists
  as a feature: an absolute temperature threshold fires on hot afternoons.
- **Dirty telemetry** — dropouts, stuck sensors holding their last value, and instrument
  spikes that are *not* machine faults.
- **Machines are offline during repair**, with sensors reading near zero.

### 2. Labelling (`labeling.py`) — three rules that each inflate the score if skipped

| Rule | Why | Effect here |
|---|---|---|
| Drop rows where the machine is **down** | Sensors read ~0 during repair; trivially separable. A model trained on them learns to detect failures that *already happened* | −6,746 rows |
| Drop the **last H hours** of each machine's history | Their future window extends past the data, so "no failure" is an assumption, not an observation (right-censoring) | −1,860 rows |
| Optional **lead-time gap** | An alert one hour before failure is useless; `--gap 6` labels only failures 6–24 h away | available, default 0 |

Result: 164,194 rows, 11,664 positives (**7.10%**, a 13.1:1 imbalance), all 486 failures
covered.

### 3. Features (`features.py`) — 152 features, every one causal

Per sensor: trailing means (6 h, 24 h), trailing standard deviations, 24 h range, EWMA,
coefficient of variation, lags at 1/6/24 h, deltas against those lags, deviation from the
24 h mean, short-vs-long mean ratio, and a z-score against **that machine's own prior
history** (expanding, shifted by one hour).

Plus cross-sensor physics (`excess_temp_over_ambient`, `power_per_rpm`,
`vibration_temp_product`, `efficiency_proxy`), a multivariate alarm count
(`sensors_abnormal_2sigma` — one drifting sensor is often noise, three at once rarely is),
and operating context (hours since service, duty cycle, machine metadata).

**Leakage rules enforced:** forward-fill only (never `interpolate()` or `bfill()`, both of
which pull values backwards from the future); trailing windows only (never
`center=True`); expanding baselines shifted by one hour, never whole-series statistics;
warm-up rows dropped rather than back-filled.

**And verified, not asserted.** `verify_causality()` rebuilds every feature from a series
truncated at time *t* and checks the row at *t* is identical to the one built from the full
series:

```
verifying feature causality (no future leakage)...
  [PASS] 3 cut-points, 456 feature values compared, 0 mismatches
```

The pipeline aborts if any mismatch appears. A unit test also injects a 50× spike at the
final hour and asserts no earlier row's features change.

### 4. Models (`train.py`)

Logistic regression, random forest, XGBoost, LightGBM — tuned with `RandomizedSearchCV`
over `TimeSeriesSplit` folds, scored on **average precision**.

- **The split is by time, never random.** A random split puts 09:00 in training and 10:00
  in test; they share a 24-hour rolling window, and the resulting score is fiction.
- **Selection on PR-AUC.** With 5.2% positives, ROC-AUC stays flattering (every model here
  exceeds 0.91) while precision varies from 0.29 to 0.48. PR-AUC separates them.
- **Hyperparameters are searched on the most recent 50,000 training rows, then the winner
  is refitted on all 113,596.** Searching on the full matrix made each of the parallel
  workers materialise a ~150 MB dense copy; the run spent its time swapping rather than
  fitting.
- **Two independent robustness checks**: the undersampling ablation and the held-out-machine
  split described above.

### 5. Evaluation (`evaluate.py`)

The module answers the three questions a planner actually asks:

1. **Of the real failures, how many did we catch?** — `event_level_detection()`. One alert
   anywhere in the window is a catch. Row-level recall counts machine-hours and reads
   higher; nobody schedules maintenance by the hour.
2. **How much warning?** — lead time of the first alert, reported per onset type and per
   failure mode.
3. **How many false alarms per week?** — consecutive alerts are collapsed into **episodes**
   (a 24 h cooldown), because ten alerts in ten consecutive hours is one maintenance
   decision, not ten. Counting raw alerting rows overstates the burden roughly tenfold.

Then it prices the trade-off (unplanned failure 18,000; inspection 450; 75% of caught
failures averted) and picks the cost-minimising threshold. Operating points at fixed
recall are also tabulated:

| Target row recall | Threshold | Row precision | Event recall | Mean lead | Alerts/machine/week |
|---|---|---|---|---|---|
| 0.70 | 0.337 | 0.611 | 91.6% | 19.3 h | 0.37 |
| 0.80 | 0.153 | 0.474 | 97.2% | 19.9 h | 0.53 |
| 0.90 | 0.074 | 0.298 | 100.0% | 22.1 h | 0.97 |

### 6. Monitoring (`monitor.py`)

Scores every machine's latest reading, bands it against the cost-optimal threshold,
computes a 24-hour risk trend (a machine at 0.4 and climbing is not a machine at 0.4 and
falling), and translates per-machine SHAP contributions into plain symptoms —
"unstable vibration (rising variability); running hot relative to ambient" — with a
recommended action. Outputs `reports/fleet_status.csv` and `reports/alerts.json`.

---

## Files

```
src/generate_data.py   fleet simulator: modes, ramps, sensor faults, downtime
src/labeling.py        failure-window labels, censoring, downtime exclusion
src/features.py        152 causal features + verify_causality()
src/train.py           4 models, time-aware CV, ablation, robustness, calibration
src/evaluate.py        event-level recall, lead time, alert episodes, cost curve
src/monitor.py         fleet risk view, alerts, symptom translation
app.py                 Streamlit monitoring dashboard
REPORT.md              the internship report
reports/               evaluation.json, fleet_status.csv, alerts.json, figures/
```

---

## Honest limitations

- **The top two features are maintenance-schedule features**, not sensor features
  (`hours_since_maintenance`, mean |SHAP| 0.049; `maintenance_overdue_ratio`, 0.047).
  This is partly an artefact of the simulator: the wear clock resets after every repair,
  and failures are spaced at least 180 hours apart, so "recently serviced" genuinely
  implies "unlikely to fail soon" in this data. The effect is real in industry too, but on
  real telemetry this feature would need auditing — it can easily become a proxy for
  maintenance *bookkeeping* rather than machine condition.
- **The data is synthetic.** These numbers measure recovery of a known degradation
  process. 97.2% event-level recall should not be read as a forecast of real-world
  performance.
- **24 hours is the only horizon evaluated.** The horizon is a parameter (`--horizon`), and
  a serious study would sweep it — 6 h, 24 h, 72 h — because lead time and achievable
  recall trade off directly against each other.
- **Binary classification, not remaining-useful-life.** A regression on time-to-failure, or
  a survival model, would give the planner a schedule rather than a flag.
- **The cost model uses flat constants.** Real plants have per-machine criticality
  (already in the data, unused in the cost function), spare-part lead times, and
  production schedules that make some downtime windows far cheaper than others.
- **No concept-drift handling.** The model is fitted once. In production, sensor
  recalibration and equipment ageing shift the feature distribution, so this needs
  scheduled retraining and drift monitoring on the input distribution.
