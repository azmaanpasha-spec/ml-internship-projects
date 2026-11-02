# Project 1 — Customer Churn Prediction

Predict which subscription customers are likely to leave, so retention effort can be
aimed at the accounts where it pays for itself.

The deliverable is not a model score. It is a **ranked, costed worklist**: who to call,
why they are at risk, what to offer them, and how many customers it is worth contacting
before the campaign stops making money.

---

## Results

Measured on a stratified 20% hold-out set (1,600 customers) that is touched exactly once,
in `src/evaluate.py`.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.8647 | 0.7062 | 0.624 | 0.711 | 0.665 | 0.1519 | 0.1461 |
| Random forest | 0.8573 | 0.6805 | 0.606 | 0.679 | 0.640 | 0.1476 | 0.1233 |
| LightGBM | 0.8680 | 0.7013 | 0.636 | 0.682 | 0.658 | 0.1371 | 0.0967 |
| **XGBoost** | **0.8734** | **0.7117** | 0.651 | 0.695 | 0.672 | 0.1403 | 0.1229 |
| XGBoost + isotonic calibration | 0.8740 | **0.7154** | 0.731 | 0.554 | 0.630 | **0.1215** | **0.0194** |

**Headline:** ROC-AUC 0.874, PR-AUC 0.715 against a 26.5% base churn rate.

Three results worth drawing attention to:

1. **Calibration matters more than the AUC here.** Isotonic calibration barely moves
   ranking quality (ROC-AUC 0.8734 → 0.8740) but cuts expected calibration error by
   **6×** (0.123 → 0.019) and the Brier score from 0.140 to 0.122. That is the difference
   between a score that ranks customers and a probability you can multiply by revenue to
   build a budget. The risk bands show it working:

   | Band | Customers | Mean predicted | Actual churn rate |
   |---|---|---|---|
   | Critical | 925 | 0.840 | 0.885 |
   | High | 757 | 0.589 | 0.668 |
   | Medium | 923 | 0.397 | 0.422 |
   | Low | 5,395 | 0.100 | 0.074 |

2. **The profit-optimal threshold is 0.20, not 0.50.** Missing a churner costs far more
   than an unnecessary retention offer, so the model should over-refer. At 0.20 the
   campaign contacts 43% of the base, reaches **85% of all churners**, and returns a
   **2.58× ROI** (net 89,747 on the test set's economics). The F1-optimal threshold (0.37)
   looks better statistically and makes less money.

3. **Tree models earn their place, but not by much.** XGBoost beats the logistic baseline
   by 0.009 ROC-AUC. Anyone reporting a huge boosting win on a churn problem of this shape
   is probably leaking something.

**Top drivers** (permutation importance — the drop in ROC-AUC when a feature is shuffled):
`contract` (0.051), `tenure_months` (0.044), `satisfaction_score` (0.023),
`risk_flag_count` (0.015), `usage_trend_pct` (0.011), `age` (0.008).

That `usage_trend_pct` ranks above every static billing field is the whole argument for
the feature engineering stage: *where the customer is heading* beats *what the customer
looks like*.

---

## How to run

```bash
pip install -r ../requirements.txt

python -m src.generate_data      # 8,000 customers with 6 months of history
python -m src.preprocess         # cleaning + data-quality report
python -m src.features           # behavioural feature engineering
python -m src.train              # 4 models, tuned, then calibrated
python -m src.evaluate           # metrics, figures, threshold selection
python -m src.score_customers    # ranked worklist with reasons

streamlit run app.py             # interactive retention console
```

Total runtime is roughly three minutes on a laptop.

---

## Approach

### 1. Data

No public dataset carries demographics, tenure, usage history, support interactions *and*
billing history for the same customers with a known ground truth, so `generate_data.py`
simulates 8,000 customers with six months of per-month usage, billing, ticket and login
history.

Churn is drawn from a latent logistic model with three parts: additive main effects,
**interaction terms**, and an irreducible noise term tuned so a good model lands near
0.87 ROC-AUC rather than a meaningless 0.99.

The interactions are deliberate. A price rise on a month-to-month contract is far more
dangerous than the same rise on a two-year term, a bad first year is rarely recovered, and
the age effect is U-shaped. Without them the data is perfectly log-linear, the logistic
baseline is optimal by construction, and the model comparison is theatre. *(This was
measured, not assumed: on an additive-only version of the generator, logistic regression
beat XGBoost on PR-AUC, 0.7062 vs 0.7117 reversed.)*

The raw CSV is also deliberately dirty, so the cleaning stage has real work:

- `total_charges` arrives as a **blank string** for never-billed customers
- `gender` and `region` carry spelling variants from multiple source systems
  (1,646 and 627 rows fixed respectively)
- `avg_resolution_hours` is **structurally missing** where no ticket was raised
- usage history is absent for months before a customer joined
- 1.2% of usage readings are meter-glitch spikes; 32 rows are duplicated

### 2. Cleaning (`preprocess.py`)

Distinguishes three different kinds of "missing", because they need different treatment:

| Kind | Example | Treatment |
|---|---|---|
| Structurally zero | `total_charges` for a day-one customer | set to 0.0 — it is known, not unknown |
| Structurally absent | resolution time with no ticket | 0 plus an explicit `_missing` indicator flag |
| Genuinely missing | a dropped usage reading | fill from that customer's own history, then global median |

Outliers are **winsorised at the 99.5th percentile rather than dropped**, because the
customers they belong to are real and dropping rows would bias the sample. Class imbalance
(2.78:1) is handled at model level rather than by resampling — see below.

Every decision is recorded in a `CleaningReport` and printed, so the cleaning is auditable
instead of silent.

### 3. Features (`features.py`)

Six months of raw history becomes behavioural signal:

- **Usage dynamics** — least-squares slope, slope normalised by the customer's own level,
  volatility, recent-3-months vs prior-3-months ratio, a sharp-drop flag
- **Billing dynamics** — recent price change %, a price-increase flag, charge trend
- **Support dynamics** — ticket escalation (recent minus prior quarter), tickets per month,
  escalation rate
- **Engagement** — login slope and recent-vs-prior ratio
- **Value and tenure** — revenue per month, charges per GB, tenure buckets
- **Product footprint** — service count, protective add-ons, streaming count
- **Payment friction** — late-payment rate, manual-payment flag
- **`risk_flag_count`** — a composite of six independent binary risks; cheap, fully
  explainable, and one of the strongest single predictors

The per-month raw columns are then **dropped**. The engineered trends carry the same
information in a form that generalises; keeping both invites the model to memorise
month-specific noise.

### 4. Models (`train.py`)

Logistic regression (baseline), random forest, XGBoost and LightGBM, each tuned with
`RandomizedSearchCV` over stratified folds of the **training split only**.

**Class imbalance is handled with class weights rather than resampling**, because
weighting uses every row while undersampling discards most of the majority class. It is
worth being precise about what this does *not* buy: class weighting distorts predicted
probabilities just as resampling does — it pushes them towards 0.5 — which is exactly why
the calibration step below is not optional. Project 2 measures this trade-off directly on
the same test set, and found undersampling gave an almost identical PR-AUC with a *better*
raw Brier score; weighting was still preferred there for using all the data, with
calibration applied afterwards either way.

The winner is selected on **PR-AUC**, not ROC-AUC, because with a 26% positive class
PR-AUC reflects minority-class ranking quality more faithfully. It is then wrapped in
`CalibratedClassifierCV` (isotonic), because weighted tree ensembles are confident and
badly calibrated.

### 5. Evaluation (`evaluate.py`)

Beyond the standard metrics: decile lift and cumulative gains, calibration curves,
expected calibration error, permutation importance, SHAP attributions, and a **retention
campaign simulation** that prices every threshold from 0.05 to 0.95 and picks the one that
maximises net profit.

The campaign assumptions are stated explicitly in the module so the recommendation can be
re-derived when the business changes them — and the Streamlit app lets you do exactly that
with sliders.

### 6. Scoring (`score_customers.py`)

Produces the actual deliverable. Each flagged customer gets a calibrated probability, a
risk band, revenue at risk, **up to three plain-language reasons** derived from their own
SHAP contributions translated against their own values, and a suggested next action.

Example output:

```
CUST-006805  p=0.995  Critical  at-risk=1,619
    - Low satisfaction score (3.3/10)
    - On a month-to-month contract, so can leave with no penalty
    -> Service-recovery call from a retention specialist
```

Probabilities are clipped to [0.001, 0.995]: isotonic regression maps its top bin to
exactly 1.0, and telling a retention agent a customer is "100% certain to churn" destroys
trust in the tool for no ranking benefit.

---

## Files

```
src/generate_data.py     synthetic customer base with interaction-rich churn process
src/preprocess.py        cleaning, imputation, outlier capping, audit report
src/features.py          behavioural feature engineering from 6-month history
src/train.py             4 models, tuning, imbalance handling, calibration
src/evaluate.py          metrics, figures, lift, SHAP, campaign simulation
src/score_customers.py   ranked worklist with reason codes and actions
app.py                   Streamlit retention console
reports/                 evaluation.json, worklists, figures/
```

---

## Honest limitations

- **The data is synthetic.** Metrics measure recovery of a known generative process, not
  real customer behaviour. The pipeline is the transferable artefact; the numbers are not.
- **Churn is modelled as a point-in-time label**, not a survival problem. A
  time-to-event model (Cox, or a discrete-time hazard) would make better use of tenure and
  is the natural next step.
- **The campaign simulation assumes a constant 32% offer success rate** across all
  customers. In reality, responsiveness to an offer is itself a prediction problem, and
  targeting *persuadable* customers rather than *likely-to-churn* customers is the proper
  uplift-modelling formulation.
- **No fairness audit.** The model uses age and gender. Before deployment it would need
  per-group calibration and error-rate checks, and most likely those features removed.
