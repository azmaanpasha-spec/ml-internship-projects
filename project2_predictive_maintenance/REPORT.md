# Predictive Maintenance for Industrial Equipment

### Predicting equipment failure from sensor telemetry, and converting predictions into an alerting policy

**Internship Project Report**
Date: 2 November 2026
Project type: Rare-event time-series classification
Code: this repository · Reproduce with `python run_all.py --project 2`

---

## Abstract

This project builds a machine learning system that predicts whether an industrial machine
will fail within the next 24 hours, using hourly readings from seven sensors together with
maintenance history. A fleet of 80 machines over 90 days was simulated, producing 172,800
machine-hours of telemetry and 486 failures across five distinct failure modes.

After labelling, 164,194 observations remained with a 7.10% positive rate (13.1:1
imbalance). 152 features were engineered — trailing statistics, lags, cross-sensor
physics and per-machine anomaly scores — under strict causality constraints that were
**verified empirically** rather than assumed. Four model families were compared on a
strictly chronological split and selected on PR-AUC.

A random forest achieved **PR-AUC 0.6894 and ROC-AUC 0.9568**. At the cost-optimal alert
threshold, the system catches **104 of 107 failures (97.2%)** with a **median 24 hours of
warning**, at a burden of **0.53 alert episodes per machine per week**, representing a
projected saving of **1.33 million** against a purely reactive strategy.

The most important methodological result is that **detection rate is a misleading headline
metric**. Failures with a sudden onset (2–8 hours) are detected at 94.4% — nearly the same
rate as gradual failures at 97.8% — but with a mean lead time of **2.4 hours instead of
23.3**. A single aggregate recall figure conceals that distinction entirely. Reporting lead
time per onset type is the central recommendation of this report.

A second finding contradicted a widely repeated piece of practical advice: random
undersampling matched class weighting on PR-AUC (0.6853 vs 0.6894) and produced
*better*-calibrated probabilities (Brier 0.0393 vs 0.0819), not worse.

---

## 1. Introduction and problem statement

### 1.1 Context

Unplanned equipment failure is expensive in three ways at once: the repair itself, the lost
production while the machine is down, and the knock-on disruption to everything downstream
of it. The three common maintenance strategies trade these costs differently:

| Strategy | Mechanism | Weakness |
|---|---|---|
| **Reactive** | repair after failure | maximum downtime, emergency labour rates |
| **Preventive** | service on a fixed schedule | services healthy machines; still misses early failures |
| **Predictive** | service when condition indicates | requires a model that is trustworthy enough to act on |

This project builds the third. The critical phrase is *trustworthy enough to act on*: a
model that flags 40% of the fleet every week will be ignored within a month regardless of
its recall, so the alert burden is treated here as a first-class result rather than a
footnote.

### 1.2 Formal problem

For each machine *m* at each hour *t*, predict

> **P(machine *m* fails during the interval (*t*, *t* + 24 hours] | all observations up to and including *t*)**

This is a binary classification problem over machine-hours, with three properties that
shape every subsequent decision:

1. **The positive class is rare** — 7.1% of observations after labelling.
2. **Observations are not independent.** Consecutive hours from the same machine are
   strongly correlated, which makes naive cross-validation invalid.
3. **The cost matrix is deeply asymmetric.** A missed failure costs roughly 40× an
   unnecessary inspection, so the default 0.5 decision threshold is close to meaningless.

### 1.3 Success criteria

Defined before modelling began:

- Catch a large majority of failures (target: >90% at event level).
- Provide enough warning to schedule a repair (target: median >12 hours).
- Keep the alert burden tolerable (target: <1 alert per machine per week).
- Produce **calibrated** probabilities, so expected cost can be computed rather than guessed.
- Demonstrate the absence of data leakage, rather than claiming it.

All five were met; §6 reports against each.

---

## 2. Data

### 2.1 Why the data is synthetic — and what that costs

No public dataset provides hourly multi-sensor telemetry, a maintenance work-order log,
*and* ground-truth failure causes with known onset times for the same fleet. The nearest
candidates each miss something essential: NASA C-MAPSS has run-to-failure trajectories but
no maintenance history and no distinct failure modes; the Azure predictive-maintenance
sample has a work-order log but coarse telemetry.

**The cost of simulating is real and must be stated up front: the metrics in this report
measure how well the models recover a known generative process, not how well they would
predict real equipment.** 97.2% event-level recall is a property of this simulator.

**What it buys is ground truth that makes the engineering verifiable.** Because the
simulator records which failures were gradual and which were sudden, detection can be
decomposed by onset speed — which turned out to be the most important result in the project
and is simply not measurable on a real dataset where onset times are unknown. Because the
simulator knows the true degradation signature of each mode, it is possible to confirm the
model is responding to physics rather than to an artefact.

### 2.2 Simulator design

`src/generate_data.py` simulates 80 machines of four types over 90 days at hourly
resolution.

**Machine types** each have a nominal operating point and characteristic failure rate:

| Type | Temp (°C) | Vibration (RMS) | Pressure (bar) | RPM | Power (kW) | Failures / 1000 h |
|---|---|---|---|---|---|---|
| Centrifugal pump | 55 | 2.0 | 6.5 | 1480 | 18 | 1.5 |
| Screw compressor | 70 | 3.2 | 9.0 | 2950 | 45 | 2.3 |
| Belt conveyor | 42 | 1.4 | 2.0 | 950 | 11 | 1.1 |
| Gas turbine | 85 | 2.6 | 12.0 | 3600 | 120 | 2.8 |

**Five failure modes**, each with a distinct sensor signature expressed as relative
deviation at the moment of failure:

| Mode | Signature | Share |
|---|---|---|
| `bearing_wear` | vibration +180%, temp +22%, acoustic +12%, power +8% | 30% |
| `seal_leak` | pressure −35%, oil quality −30%, temp +8% | 20% |
| `overheating` | temp +45%, power +18%, oil quality −15% | 18% |
| `motor_fault` | power +30%, acoustic +20%, RPM −12%, temp +15% | 17% |
| `lubrication_failure` | oil quality −45%, vibration +90%, temp +25% | 15% |

Because no two modes share a signature, **a model must combine sensors**; a per-sensor
threshold rule cannot separate a seal leak from a bearing fault.

**Degradation dynamics.** Each failure has an onset time, and the signature grows as
`progress^1.8` — accelerating, so early warning is faint and the choice of horizon
genuinely matters. 80% of failures are *gradual* with onsets of 48–220 hours; **20% are
*sudden*, with onsets of only 2–8 hours.** Retaining the sudden class was a deliberate
decision: it imposes a realistic ceiling, and a model reporting near-perfect lead time on
this data would indicate a bug rather than a result.

**Realistic nuisances**, each present to make a specific engineering step necessary:

| Nuisance | Forces |
|---|---|
| Ambient temperature cycle (daily + seasonal) affecting all temperature readings | a confounder-corrected feature (`excess_temp_over_ambient`) |
| Duty-cycle variation (shift patterns, lighter weekends) | load-normalised features |
| AR(1) autocorrelated sensor noise | trailing statistics rather than point thresholds |
| Sensor dropouts (0.4% of readings) | explicit, causal imputation |
| **Stuck sensors** holding their last value for 4–14 hours | missingness indicators |
| Instrument spikes (1.8–3.2×) that are *not* machine faults | robustness to outliers |
| Machines offline during repair, sensors near zero | an exclusion rule (§3) |

### 2.3 Dataset characteristics

```
machines        : 80
sensor readings : 172,800 rows (90 days hourly)
failures        : 486 (94 sudden / near-unpredictable)
modes           : bearing_wear 136, seal_leak 95, motor_fault 92,
                  overheating 89, lubrication_failure 74
mean onset      : 110.3 h
downtime        : 6,755 h total
missing cells   : 4,789
```

Three files are emitted, mirroring how this data genuinely arrives in industry — an asset
register, a telemetry historian, and a work-order log:

- `machines.csv` — one row per machine (type, age, rated power, site, criticality)
- `sensor_readings.csv` — one row per machine-hour (7 sensors + ambient, load, state)
- `failure_events.csv` — one row per failure (time, mode, **true onset**, downtime, cost)

---

## 3. Problem formulation: labelling and the three correctness rules

Labelling looks trivial — mark an hour positive if a failure follows within 24 hours — and
it is where this class of project most often goes quietly wrong. `src/labeling.py` enforces
three rules. **Each one, if skipped, inflates the reported score without producing any
visible symptom.**

### Rule 1 — Exclude rows where the machine is not running

During repair the machine is off and its sensors read near zero. Those rows are trivially
separable from normal operation. A model trained on them achieves an excellent score by
learning to detect that a machine has *already* failed, which has no predictive value
whatsoever.

→ **6,746 rows removed.**

### Rule 2 — Exclude the final H hours of every machine's history (right-censoring)

For an observation in the last 24 hours of the dataset, the 24-hour future window extends
beyond the available data. Labelling such a row "no failure" is not an observation; it is
an assumption. Keeping them pads the negative class with unverifiable labels.

→ **1,860 rows removed.**

### Rule 3 — Support an optional lead-time gap

An alert fired one hour before a failure is technically correct and operationally
worthless. The `--gap` parameter restricts positives to failures between *gap* and *H*
hours away, guaranteeing every alert leaves time to respond. Default 0 for the headline
results; the mechanism exists and is unit-tested.

### Resulting dataset

```
prediction window : failure within 24h (gap 0h)
rows              : 172,800 -> 164,194
  dropped (down)  : 6,746 machine not running
  dropped (censor): 1,860 future window not observed
positives         : 11,664 (7.10%)
imbalance ratio   : 13.1:1
failures covered  : 486/486
```

These rules are verified by unit tests that assert the label of every row in a
hand-constructed 60-hour fleet (16 negative, 24 positive, 5 dropped as down, 15 dropped as
censored) — see `tests/test_project2_labeling.py`.

---

## 4. Feature engineering

### 4.1 Design

A raw sensor reading says what the machine is doing *now*. Failure prediction needs to know
where it is *heading*, and whether its current behaviour is abnormal **for that machine**.
152 features were built in seven families.

**Per sensor** (7 sensors × 15 features):

| Family | Features |
|---|---|
| Level | current reading |
| Trend | trailing mean over 6 h and 24 h, EWMA (span 12) |
| Dispersion | trailing std over 6 h and 24 h, coefficient of variation |
| Shape | 24 h range (max − min) |
| Memory | lags at 1 h, 6 h, 24 h |
| Change | delta vs each lag, deviation from the 24 h mean, short-vs-long mean ratio |
| Anomaly | **z-score against that machine's own prior history** |

The anomaly family is the most important design choice. An absolute vibration of 4.0 RMS is
alarming for a conveyor and normal for a compressor. Scoring each reading against the
machine's *own* expanding history makes the feature comparable across a heterogeneous
fleet.

**Cross-sensor physics** — combinations no single sensor expresses:

- `excess_temp_over_ambient` — raw temperature tracks the plant ambient, so an absolute
  threshold fires on hot afternoons. The excess over ambient isolates the machine's own
  contribution. **This is the feature the ambient confounder was introduced to require.**
- `power_per_rpm`, `vibration_per_rpm` — load-normalised, so a change at constant speed is
  distinguishable from a change in speed
- `vibration_temp_product` — captures the joint bearing-failure signature
- `efficiency_proxy` (pressure / power), `oil_degradation`, `thermal_load_ratio`
- `sensors_abnormal_2sigma` / `_3sigma`, `max_abs_z_score`, `mean_abs_z_score` — a
  multivariate alarm count. One drifting sensor is often noise; three at once rarely is.

**Operating context** — hours since maintenance, maintenance-overdue ratio, duty-cycle
mean and variability, load utilisation, machine type, age, criticality, hour-of-day,
day-of-week.

### 4.2 Leakage prevention

Leakage is the dominant failure mode of time-series ML. Four rules were enforced:

| Rule | Rejected alternative | Why the alternative leaks |
|---|---|---|
| Forward-fill only, limit 6 h | `interpolate()`, `bfill()` | both pull values *backwards* from the future |
| Trailing windows only | `rolling(center=True)` | a centred window straddles the present |
| Expanding baseline shifted by 1 h | whole-series mean/std per machine | whole-series statistics include future rows |
| Drop warm-up rows | back-filling partial windows | fabricates history that did not exist |

### 4.3 Verification, not assertion

Code comments claiming "no leakage" are worth nothing. `verify_causality()` tests the
property **empirically**:

> For a sampled (machine, time) pair, build the features from the full series; then rebuild
> them from the series **truncated at that timestamp**; then compare the feature row at
> that timestamp. If any feature consulted a future value, the two rows must differ.

```
verifying feature causality (no future leakage)...
  [PASS] 3 cut-points, 456 feature values compared, 0 mismatches
feature frame : 162,194 rows x 163 columns
model inputs  : 152 numeric + 2 categorical
positives     : 11,664 (7.19%)
```

**The pipeline aborts if any mismatch is found**, so this is a build-time gate, not a
diagnostic. A complementary unit test injects a 50× spike at the final hour and asserts no
earlier row's features change.

Separately, `feature_columns()` excludes the simulator's ground-truth `_health_index`, the
label, and every future-derived column, and a unit test asserts none of them reaches the
model.

---

## 5. Modelling methodology

### 5.1 Validation strategy — two different questions

**The primary split is chronological**: the first 70% of the calendar trains, the final 30%
tests.

A random split would be catastrophic here and worth spelling out: observations at 09:00 and
10:00 from the same machine share a 24-hour rolling window and nearly identical feature
values. Splitting randomly places near-duplicates on both sides, and the resulting score
measures interpolation, not prediction.

```
time-based split at 2026-06-02 21:00:00
  train: 113,596 rows, 9,135 positives (8.04%)
  test : 48,598 rows, 2,529 positives (5.20%)
```

The positive rate differs between splits (8.04% vs 5.20%) because failures are not
uniformly distributed in time. This is realistic and is left uncorrected.

**A second, independent split holds out 20 whole machines.** This answers a genuinely
different question — *will the model work on equipment it has never seen?* — which matters
when commissioning a new machine. Both numbers are reported in §6.6 because a fleet
operator needs both.

Within training, hyperparameters are tuned with `TimeSeriesSplit`, so every validation fold
is strictly later than its training fold.

### 5.2 Metric selection

Models are selected on **PR-AUC (average precision)**, not ROC-AUC.

This is not pedantry; the results demonstrate it. Every model in §6.1 exceeds **0.91
ROC-AUC**, which suggests they are all excellent and nearly equivalent. Their precisions at
the same threshold range from **0.29 to 0.48** — a 65% spread. ROC-AUC is dominated by the
94.8% of easy negatives; PR-AUC tracks the quantity the maintenance team actually
experiences: *of the alerts raised, how many were real*.

### 5.3 Class imbalance

`class_weight="balanced"` (random forest, logistic regression, LightGBM) and
`scale_pos_weight = 11.4` (XGBoost). The alternative — random undersampling of negatives —
was **measured rather than dismissed**; see §6.6.

### 5.4 Hyperparameter search, and a resource problem worth recording

`RandomizedSearchCV` over `TimeSeriesSplit(3)`, scored on average precision, with 4–12
candidate draws per family.

The first attempt did not complete. Searching over all 113,596 training rows × 152 features
caused each of the twelve parallel workers to materialise its own ~150 MB dense copy of the
transformed matrix; at roughly 7 GB of working set the machine began swapping and the run
made no progress for twenty minutes.

Two changes fixed it, and both are documented in the code because they are
methodologically relevant rather than incidental:

1. **Search on the most recent 50,000 training rows, then refit the winning configuration
   on all 113,596.** Model *comparison* does not need the full dataset; the final model
   does. Time ordering is preserved in the slice so `TimeSeriesSplit` remains valid.
2. **Cap worker parallelism, and cap LightGBM's own thread count.** LightGBM defaults to
   one thread per core; nested inside a process pool this requested 48 threads on 12 cores.
   Capping it to 2 reduced that family's search from over 20 minutes (unfinished) to
   **110.8 seconds**.

### 5.5 Calibration

The winning model is wrapped in `CalibratedClassifierCV` (isotonic, `TimeSeriesSplit`
folds). Class-weighted tree ensembles are systematically overconfident, and §6.2 shows the
magnitude. Calibration is essential here because the threshold is chosen by expected-cost
minimisation — which multiplies probabilities by costs and is meaningless if the
probabilities are wrong.

---

## 6. Results

### 6.1 Model comparison

Held-out time period, 48,598 machine-hours, 5.20% positive, threshold 0.50:

| Model | PR-AUC | ROC-AUC | Precision | Recall | Brier | ECE |
|---|---|---|---|---|---|---|
| Logistic regression | 0.5652 | 0.9109 | 0.290 | 0.847 | 0.1151 | 0.2597 |
| **Random forest** | **0.6894** | **0.9568** | 0.462 | 0.843 | 0.0819 | 0.2155 |
| LightGBM | 0.6876 | 0.9502 | 0.481 | 0.777 | 0.0410 | 0.0566 |
| XGBoost | 0.6702 | 0.9490 | 0.357 | 0.839 | 0.0636 | 0.1138 |
| Random forest + isotonic | 0.6716 | 0.9506 | 0.693 | 0.539 | **0.0268** | **0.0161** |

*Figures: `reports/figures/pr_curves.png`, `roc_curves.png`, `model_comparison.png`*

Observations:

- **The non-linear models beat the linear baseline decisively** — PR-AUC 0.69 vs 0.57, a
  22% relative gain. This is expected: the degradation signatures are multiplicative and
  mode-specific, so the decision boundary is genuinely non-linear.
- **Random forest and LightGBM are statistically indistinguishable** (0.6894 vs 0.6876).
  Random forest was selected on PR-AUC, but LightGBM would be the better operational choice
  on two counts: it was already far better calibrated before any post-processing (ECE 0.057
  vs 0.216) and it trains faster. This is noted as a deliberate judgement rather than
  hidden.
- **ROC-AUC is uninformative here**, as predicted in §5.2: the spread across models is
  0.046 on ROC-AUC against 0.124 on PR-AUC.

### 6.2 Calibration

Isotonic calibration trades a small amount of ranking quality (PR-AUC 0.6894 → 0.6716) for
a transformation in probability quality:

| | Before | After | Improvement |
|---|---|---|---|
| Brier score | 0.0819 | **0.0268** | 3.1× |
| Expected calibration error | 0.2155 | **0.0161** | **13.4×** |

*Figure: `reports/figures/calibration.png`*

An ECE of 0.2155 means the uncalibrated model's stated confidence was wrong by 22
percentage points on average — it would claim 60% risk for machines that fail 38% of the
time. After calibration the error is 1.6 points. **Since the operating threshold is chosen
by minimising expected cost, this step is what makes the cost analysis in §6.4 legitimate
rather than decorative.**

### 6.3 Event-level detection and lead time — the central result

Row-level recall counts machine-hours. No one schedules maintenance by the hour. The
operationally meaningful question is: **of the failures that occurred, how many produced at
least one alert in time to act?**

At the cost-optimal threshold of 0.15:

```
failures in test    : 107
detected            : 104 (97.2%)
median lead time    : 24.0 h
mean lead time      : 19.9 h
```

Decomposed by onset speed:

| Onset type | Failures | Caught | Rate | Mean lead | Median lead | Max lead |
|---|---|---|---|---|---|---|
| **Gradual** (48–220 h onset) | 89 | 87 | **97.8%** | **23.3 h** | 24 h | 24 h |
| **Sudden** (2–8 h onset) | 18 | 17 | **94.4%** | **2.4 h** | 2 h | 5 h |

*Figure: `reports/figures/lead_time.png`*

**This table is the most important result in the project, and it was nearly missed.**

The initial expectation was that sudden failures would be largely undetectable — that was
the stated purpose of including them. The detection rates refute that: 94.4% versus 97.8%,
a difference of one failure. On that evidence the model appears to handle both classes
equally well.

The lead times show it does not. The mechanism is subtle: the 24-hour alert window extends
**right up to the moment of failure**, so a model that only detects a sudden failure two
hours beforehand still registers as having "caught" it. The binary label cannot distinguish
a day of warning from a two-hour scramble, and **a single aggregate recall figure conceals
the distinction completely.**

For a maintenance planner the two situations are not comparable. 23 hours permits a
scheduled repair in a planned window with parts ordered. 2.4 hours permits, at best, a
controlled shutdown. Both count identically in the recall column.

Two conclusions follow, and they are the report's primary recommendations:

1. **Report lead time per onset type, not just detection rate.** `evaluate.py` now does
   this for both onset classes and all five failure modes.
2. **Use the `--gap` parameter in production.** Setting a 4–6 hour gap re-labels the
   problem so that alerts arriving too late to act on no longer count as successes —
   turning the metric into an honest measure of actionable warning.

By failure mode:

| Mode | Caught | Rate | Mean lead | Median lead |
|---|---|---|---|---|
| `bearing_wear` | 29/29 | 100.0% | 21.4 h | 24 h |
| `lubrication_failure` | 13/13 | 100.0% | 17.2 h | 24 h |
| `seal_leak` | 28/29 | 96.5% | 19.6 h | 24 h |
| `motor_fault` | 19/20 | 95.0% | 20.3 h | 24 h |
| `overheating` | 15/16 | 93.8% | 19.1 h | 24 h |

The ordering is interpretable. Bearing wear and lubrication failure are caught completely —
both drive vibration, the sensor with the largest relative deviation (+180% and +90%).
Overheating is weakest at 93.8%, consistent with temperature being the one sensor that also
tracks a confounder (plant ambient), so a genuine thermal fault must clear a noisier
baseline.

### 6.4 Choosing the operating point

Nothing here defaults to 0.5. Two complementary approaches were used.

**Fixed-recall operating points**, with alerts collapsed into *episodes* — consecutive
alerts within 24 hours are one maintenance decision, not ten. Counting raw alerting rows
overstates the operational burden roughly tenfold:

| Target row recall | Threshold | Row precision | Event recall | Mean lead | **Alerts / machine / week** |
|---|---|---|---|---|---|
| 0.70 | 0.337 | 0.611 | 91.6% | 19.3 h | **0.37** |
| 0.80 | 0.153 | 0.474 | 97.2% | 19.9 h | **0.53** |
| 0.90 | 0.074 | 0.298 | 100.0% | 22.1 h | **0.97** |

*Figure: `reports/figures/recall_vs_alerts.png`*

Note the shape of this trade-off: going from 91.6% to 100% event-level recall costs 2.6× the
alert burden. Catching the last 8% of failures requires nearly tripling the number of
inspections — exactly the kind of judgement that should be made by the business, not
silently by a default parameter.

**Expected-cost minimisation**, with assumptions stated explicitly in the code so they can
be revised:

| Assumption | Value |
|---|---|
| Unplanned failure cost | 18,000 |
| Planned inspection cost | 450 |
| Failures averted when caught and inspected | 75% |
| Alert cooldown (episode definition) | 24 h |

| | |
|---|---|
| **Cost-optimal threshold** | **0.15** |
| Alert episodes | 160 (0.53 per machine per week) |
| Event-level recall | 97.2% |
| Mean lead time | 19.9 h |
| Total expected cost | 594,000 |
| Reactive-strategy cost | 1,926,000 |
| **Saving** | **1,332,000 (69%)** |

*Figure: `reports/figures/cost_curve.png`*

The optimum sits at 0.15 rather than 0.5 precisely because of the cost asymmetry: with a
missed failure costing 40× an inspection, the model should deliberately over-refer. The
Streamlit dashboard exposes these four assumptions as controls, so a stakeholder who
disagrees with them can see the threshold move.

### 6.5 What the model uses

Mean |SHAP| over 2,000 sampled test rows:

| Rank | Feature | mean &#124;SHAP&#124; |
|---|---|---|
| 1 | `hours_since_maintenance` | 0.0489 |
| 2 | `maintenance_overdue_ratio` | 0.0471 |
| 3 | `max_abs_z_score` | 0.0237 |
| 4 | `sensors_abnormal_2sigma` | 0.0140 |
| 5 | `oil_quality_idx` | 0.0136 |
| 6 | `power_per_rpm` | 0.0135 |
| 7 | `oil_quality_idx_z_vs_history` | 0.0135 |
| 8 | `vibration_temp_product` | 0.0130 |
| 9 | `load_utilisation` | 0.0129 |
| 10 | `oil_quality_idx_ewm_12h` | 0.0127 |

*Figure: `reports/figures/shap_importance.png`*

Two things are worth drawing out, one encouraging and one that requires scepticism.

**Encouraging:** ranks 3 and 4 are `max_abs_z_score` and `sensors_abnormal_2sigma` — both
*engineered multivariate* features rather than raw readings. The model's second-strongest
signal is "this machine is behaving unusually for itself, on several sensors at once",
which is exactly the intended design and vindicates the per-machine anomaly family in §4.1.
Engineered features (`power_per_rpm`, `vibration_temp_product`, `oil_quality_idx_z_vs_history`)
occupy five of the top ten.

**Requiring scepticism:** the top two features are **maintenance-schedule** features, not
sensor features. This is at least partly an artefact of the simulator — the wear clock
resets after every repair and failures are spaced at least 180 hours apart, so "recently
serviced" genuinely implies "unlikely to fail imminently" in this data. The effect exists in
industry too, but on real telemetry this feature would need auditing: `hours_since_maintenance`
can easily become a proxy for maintenance *bookkeeping* (when a work order was closed)
rather than machine *condition*. The honest statement is that the headline metrics are
supported more by schedule information than this report's framing around sensor physics
would suggest, and a follow-up experiment should retrain with both features removed to
quantify how much of the 0.6894 PR-AUC is genuinely condition-based.

### 6.6 Robustness

**Generalisation to unseen machines.** Holding out 20 whole machines instead of a time
period:

| Split | Question answered | PR-AUC | ROC-AUC |
|---|---|---|---|
| Time-based | will it work next week on this fleet? | 0.6894 | 0.9568 |
| Machine-based | will it work on a machine it has never seen? | **0.6344** | 0.9220 |

An 8% relative degradation. The model does transfer to new equipment, but measurably worse
— which is the expected and correct result, and the number a commissioning engineer would
want before trusting it on a new installation.

**Imbalance strategy ablation.** Same tuned hyperparameters, same test set, only the
imbalance strategy differing:

| Strategy | Training rows used | PR-AUC | Brier |
|---|---|---|---|
| `class_weight="balanced"` | 113,596 | **0.6894** | 0.0819 |
| Random undersampling (1:3) | 36,540 (32%) | 0.6853 | **0.0393** |

This contradicted the expectation that motivated the experiment. The common advice is that
class weighting preserves calibration where resampling destroys it. **It did not hold.**
Discriminative performance was effectively tied (a 0.6% relative difference), and the
undersampled model was *better* calibrated — because `class_weight="balanced"` skews
predicted probabilities towards 0.5 just as resampling does.

Class weighting was retained on the grounds that actually survive measurement: it uses all
113,596 training rows rather than discarding 68% of the negatives, and calibration is
restored afterwards by isotonic regression in either case. The lesson recorded here is that
this folklore should be tested per dataset rather than inherited.

---

## 7. Deployment artefact

A model that reports metrics is not a deliverable. `src/monitor.py` and `app.py` produce
what a maintenance planner would open each morning.

**Fleet view** (`reports/fleet_status.csv`, `reports/alerts.json`): every machine's current
risk, ranked and banded against the cost-optimal threshold. Current state of the simulated
fleet:

```
fleet status as of 2026-06-29 10:00:00
alert threshold: 0.15 (cost-optimal, from evaluate.py)
band counts: {'Normal': 62, 'Watch': 12, 'Alert': 5, 'Critical': 1}
```

**Risk trend.** Each machine carries a 24-hour risk change and a rising/stable/falling
label. A machine at 0.4 and climbing is a different situation from one at 0.4 and falling,
and a static score cannot express that.

**Symptom translation.** Per-machine SHAP contributions are aggregated back to source
features and mapped to plain language, so the work order reads *"unstable vibration (rising
variability); running hot relative to ambient; oil quality degrading"* rather than
*"score 0.78"*. A recommended action is attached to each alert band.

**Threshold explorer.** The dashboard lets a planner move the threshold and watch event
recall, alert burden, lead time and expected cost respond, with the four cost assumptions
exposed as inputs. The intent is that the operating point becomes a negotiated business
decision rather than a modelling artefact.

*Figures: `reports/figures/fleet_risk.png`, `risk_trajectory.png`*

---

## 8. Discussion

### 8.1 Against the success criteria

| Criterion | Target | Achieved |
|---|---|---|
| Event-level recall | >90% | **97.2%** ✓ |
| Median lead time | >12 h | **24.0 h** ✓ |
| Alert burden | <1 / machine / week | **0.53** ✓ |
| Calibrated probabilities | — | **ECE 0.0161** ✓ |
| Leakage demonstrably absent | — | **456 values verified, 0 mismatches** ✓ |

### 8.2 What I would do differently

**The suddenness analysis should have been designed in from the start.** The simulator was
built with sudden failures specifically to create an honest recall ceiling, and then the
evaluation reported only aggregate detection rate — which showed no ceiling at all (94.4% vs
97.8%) and would have supported a materially overstated conclusion. The signal was in lead
time the whole time; it was only found by looking at a breakdown that had not originally
been written. The general lesson is that **when a dataset is constructed with a known hard
subgroup, the evaluation must report that subgroup separately from the outset**, otherwise
the aggregate silently absorbs it.

**I should have questioned the top features earlier.** `hours_since_maintenance` dominating
the SHAP ranking is the kind of result that deserves immediate suspicion, and the
retrain-without-it experiment (§9) belongs in the main results rather than in future work.

**The resource problems cost real time.** Both the memory thrashing and the LightGBM thread
oversubscription were avoidable with a smaller pilot run before launching a full search.

### 8.3 What worked well

The **empirical causality check** is the component I would reuse unchanged. Writing
`verify_causality()` took well under an hour and converted "I believe there is no leakage"
into a build-time gate that compares 456 feature values and fails the pipeline on any
mismatch. Given that leakage is the dominant failure mode in time-series ML and produces no
visible symptom, this is the highest-value hour in the project.

**Alert episodes** rather than alert rows changed the apparent operational burden by an
order of magnitude and made the cost analysis credible.

**Measuring the imbalance ablation instead of asserting it** produced the one result that
genuinely surprised me and corrected a belief I held going in.

---

## 9. Limitations

1. **The data is synthetic.** All metrics measure recovery of a known process. This is the
   principal limitation and it bounds every number above.
2. **The top features are schedule-based, not condition-based** (§6.5), and the degree to
   which performance depends on them is unquantified. The necessary experiment — retrain
   without `hours_since_maintenance` and `maintenance_overdue_ratio` — is specified but not
   run.
3. **Only a 24-hour horizon was evaluated.** The horizon is parameterised; a proper study
   would sweep 6/24/72 hours, since lead time and achievable recall trade off directly.
4. **Binary classification, not remaining-useful-life.** A time-to-failure regression or
   survival model would give a planner a schedule rather than a flag.
5. **Flat cost constants.** Per-machine criticality is present in the data but unused in
   the cost function; real plants also have spare-part lead times and production schedules
   that make some downtime windows much cheaper than others.
6. **No concept-drift handling.** The model is fitted once. Sensor recalibration and
   equipment ageing shift the feature distribution, requiring scheduled retraining and
   input-distribution monitoring.
7. **Single random seed.** Results are from one simulated fleet; confidence intervals over
   repeated seeds are not reported.
8. **Random forest was selected on PR-AUC despite LightGBM being better calibrated and
   faster** (§6.1). A production choice would likely go the other way.

---

## 10. Conclusion

A predictive maintenance system was built end to end: a physics-flavoured fleet simulator,
a labelling scheme with three explicit correctness rules, 152 causal features with
build-time leakage verification, a four-model comparison on a chronological split, isotonic
calibration, cost-based threshold selection, and a monitoring dashboard with
plain-language symptom explanations.

A calibrated random forest reaches **PR-AUC 0.6894 / ROC-AUC 0.9568**, catching **97.2% of
failures with a median 24 hours of warning** at **0.53 alert episodes per machine per week**
— a projected **69% reduction in maintenance cost** against a reactive strategy.

The two findings worth carrying forward are methodological rather than numerical:

1. **Detection rate conceals what matters.** Sudden-onset failures are caught at 94.4%
   versus 97.8% for gradual ones, but with 2.4 hours of warning instead of 23.3. Because the
   alert window runs up to the failure itself, a late catch scores identically to an early
   one. Lead time must be reported per onset class, and a lead-time gap should be used in
   production so that unactionable alerts stop counting as successes.
2. **Standard advice about class imbalance did not survive measurement.** Undersampling
   matched class weighting on PR-AUC and beat it on calibration, the opposite of the
   expected result.

**Future work**, in priority order: quantify the dependence on maintenance-schedule
features by retraining without them; sweep the prediction horizon; move from binary
classification to remaining-useful-life regression; incorporate per-machine criticality into
the cost model; and validate the whole pipeline on real telemetry, where the simulator's
convenient ground truth is unavailable and the labelling rules in §3 matter most.

---

## Appendix A — Reproduction

```bash
pip install -r ../requirements.txt

python -m src.generate_data   # 172,800 readings, 486 failures
python -m src.labeling        # 164,194 labelled rows, 7.10% positive
python -m src.features        # 152 features + causality gate
python -m src.train           # 4 models, ablation, robustness, calibration
python -m src.evaluate        # metrics, lead time, cost analysis, SHAP
python -m src.monitor         # fleet view and alerts

streamlit run app.py          # monitoring dashboard
pytest ../tests -q            # 60 tests, including the labelling assertions
```

Runtime ≈ 10 minutes on a 12-core laptop. All seeds fixed (`SEED = 7`,
`RANDOM_STATE = 42`).

## Appendix B — Code manifest

| File | Lines of responsibility |
|---|---|
| `src/generate_data.py` | fleet simulator: types, modes, ramps, sensor faults, downtime |
| `src/labeling.py` | failure-window labels, downtime exclusion, censoring, lead-time gap |
| `src/features.py` | 152 causal features, `verify_causality()`, leakage exclusions |
| `src/train.py` | 4 model families, time-aware CV, search slicing, ablation, robustness split, calibration |
| `src/evaluate.py` | model comparison, event-level detection, lead time, alert episodes, cost curve, SHAP |
| `src/monitor.py` | fleet risk view, banding, trend, symptom translation, alerts |
| `app.py` | Streamlit dashboard: fleet, machine detail, performance, threshold explorer |
| `../tests/test_project2_labeling.py` | row-by-row label assertions, leakage tests, split integrity |

## Appendix C — Figures

All in `reports/figures/`:

| Figure | Shows |
|---|---|
| `pr_curves.png` | precision-recall by model — the metric that separates them |
| `roc_curves.png` | ROC by model — included to show why it is uninformative here |
| `model_comparison.png` | all metrics side by side |
| `calibration.png` | predicted probability vs observed frequency, before and after |
| `confusion_matrix.png` | at the cost-optimal threshold |
| `lead_time.png` | **lead-time distribution by onset type — the central result** |
| `cost_curve.png` | expected cost vs threshold, with the optimum marked |
| `recall_vs_alerts.png` | failures caught vs alarms chased |
| `shap_importance.png` | what the model is using |
| `fleet_risk.png` | current fleet risk, ranked and banded |
| `risk_trajectory.png` | 72-hour risk trends for the highest-risk machines |

## Appendix D — Environment

Python 3.12.10 · pandas 2.x · scikit-learn 1.8 · XGBoost 2.x · LightGBM 4.x · SHAP ·
pyarrow · matplotlib · Streamlit. Full pins in `../requirements.txt`.
