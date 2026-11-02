# Machine Learning Internship Projects

Three end-to-end machine learning projects, each taken from raw data through to a
deliverable somebody could actually use — a worklist, a monitoring dashboard, a serving
API — rather than stopping at a notebook with an accuracy score.

| # | Project | Problem type | Headline result | Deliverable |
|---|---|---|---|---|
| 1 | [Customer Churn Prediction](project1_customer_churn/) | Binary classification, 26.5% positive | ROC-AUC **0.874**, PR-AUC **0.715**, calibration error cut 6× | Ranked retention worklist with reasons + campaign simulator |
| 2 | [Predictive Maintenance](project2_predictive_maintenance/) | Rare-event time-series classification, 5.2% positive | PR-AUC **0.689**, **97.2%** of failures caught with 24 h median warning at 0.53 alerts/machine/week | Fleet monitoring dashboard with cost-optimal alerting |
| 3 | [Product Recommendation](project3_recommendation/) | Implicit-feedback ranking, 98.6% sparse | NDCG@10 **0.1128**, **+78.7%** over popularity | Recommendation API + interactive explorer |

**Project 2 is submitted as the internship report** — see
[`project2_predictive_maintenance/REPORT.md`](project2_predictive_maintenance/REPORT.md).

---

## Quick start

```bash
pip install -r requirements.txt

python run_all.py                  # reproduce all three projects end to end
python run_all.py --project 2      # just one
python run_all.py --skip-training  # reuse saved models, regenerate reports
```

Then explore any project interactively:

```bash
cd project1_customer_churn       && streamlit run app.py
cd project2_predictive_maintenance && streamlit run app.py
cd project3_recommendation       && streamlit run app.py
cd project3_recommendation       && uvicorn src.api:app --port 8000   # + /docs
```

Total reproduction time is roughly 15–20 minutes on a 12-core laptop, dominated by
hyperparameter search in projects 1 and 2.

---

## What these projects have in common

Each follows the same six-step structure — data, cleaning, features, models, evaluation,
deployment artefact — and each is built around a small number of decisions that are
defended rather than assumed.

**1. The split matches how the model would be used.**
Project 1 uses a stratified hold-out. Project 2 splits **by time** (a random split would
put 09:00 in training and 10:00 in test, sharing a 24-hour rolling window, and the score
would be fiction) and additionally reports generalisation to **machines never seen in
training**. Project 3 splits **chronologically**, because recommending backwards in time is
not the task.

**2. The metric matches the class balance.**
Models are selected on **PR-AUC**, not ROC-AUC, wherever the positive class is rare — with
5% positives, ROC-AUC is dominated by easy negatives and stays flattering while precision
is useless. Project 3 reports NDCG/MAP/Hit@K against an explicit popularity baseline,
because a recommender that merely rediscovers best-sellers has added nothing.

**3. Class imbalance is handled with weights, and the alternative is measured.**
`class_weight="balanced"` / `scale_pos_weight` rather than resampling, because weighting
uses every row where undersampling throws most of the majority class away. Project 2
trains the winning model both ways on the same test set, and the result is a useful
corrective to the usual folklore: undersampling scored an almost identical PR-AUC
(0.6853 vs 0.6894) with a **better** raw Brier score (0.039 vs 0.082). Class weighting
does not protect calibration — it skews probabilities towards 0.5 — which is why every
model here is calibrated afterwards regardless. The choice is evidenced, not asserted.

**4. Probabilities are calibrated, and calibration is reported.**
Weighted tree ensembles are confident and badly calibrated. Isotonic calibration barely
changes ranking but transforms usability: in project 1 it cuts expected calibration error
from 0.123 to 0.019. Brier score and ECE are reported alongside AUC throughout.

**5. The threshold is a business decision, priced explicitly.**
Nothing here defaults to 0.5. Project 1 simulates the retention campaign and selects the
profit-maximising threshold (0.20, ROI 2.58×). Project 2 prices unplanned failures against
inspections and selects the cost-minimising threshold, reporting the
**alerts-per-machine-per-week** that a technician would actually live with. Both expose the
assumptions as sliders so a stakeholder can disagree with them.

**6. Predictions are explained in the language of the domain.**
Per-row SHAP attributions are aggregated back to source features (so the explanation says
`contract`, not `contract_Month-to-month`) and translated into plain statements —
"usage down 34% over the last 3 months", "running hot relative to ambient" — with a
suggested action attached.

**7. The correctness-critical parts have tests.**

```bash
pytest tests -q      # 60 tests, ~20s
```

Not coverage for its own sake — the tests target the things that fail *silently*. A
ranking metric that is subtly wrong still returns plausible numbers, so NDCG and MAP are
checked against hand-computed values. A labelling bug that leaks the future just looks
like a very good model, so the labels for a 60-hour toy fleet are asserted row by row
(16 negative, 24 positive, 5 dropped as the machine is down, 15 dropped as censored) and
a 50× spike injected at the final hour is asserted to change no earlier row's features.
The tests also assert that ground-truth columns never reach the feature set, that no
recommender re-recommends an item the user has already seen, and that the ALS objective
actually decreases.

**8. Leakage is tested, not asserted.**
Project 2's feature pipeline includes `verify_causality()`, which rebuilds every feature
from a series truncated at time *t* and asserts the row at *t* is bit-for-bit identical to
the one built from the full series. If any feature peeked forward, the values would
differ. It compares 456 feature values across sampled cut-points and the pipeline aborts
if any mismatch is found.

---

## Repository layout

```
common/                  shared across projects
  evaluation.py            classification + ranking metrics, threshold selection, lift
  plotting.py              ROC/PR/calibration/importance/lift figures
  explain.py               model-agnostic SHAP (tree + linear), one-hot aggregation
  io_utils.py              paths, JSON, model persistence

project1_customer_churn/
  src/                     generate_data, preprocess, features, train, evaluate, score_customers
  app.py                   Streamlit retention console
  README.md                results, approach, limitations
  reports/                 metrics JSON, worklists, figures

project2_predictive_maintenance/
  src/                     generate_data, labeling, features, train, evaluate, monitor
  app.py                   Streamlit fleet monitor
  README.md                results, approach, limitations
  REPORT.md                >>> the internship report <<<
  reports/                 metrics JSON, fleet status, alerts, figures

project3_recommendation/
  src/                     generate_data, data_prep, recommenders, evaluate, api
  app.py                   Streamlit recommendation explorer
  README.md                results, approach, limitations
  reports/                 metrics JSON, figures

tests/                   60 tests: metric correctness, labelling, leakage, API
  conftest.py              isolates the three same-named `src` packages
  test_metrics.py          hand-computed NDCG/MAP/lift/ECE checks
  test_project1_pipeline.py   cleaning and feature-engineering correctness
  test_project2_labeling.py   label correctness, causality, split integrity
  test_project3_recommenders.py  split, cold start, ALS convergence
  test_project3_api.py     FastAPI endpoint contract

run_all.py               reproduce everything
requirements.txt
```

---

## Stack

Python 3.12 · pandas · NumPy · scikit-learn · XGBoost · LightGBM · SHAP · SciPy ·
matplotlib · Streamlit · FastAPI

Matrix factorisation (implicit ALS) is implemented directly in
`project3_recommendation/src/recommenders.py` rather than imported, so the alternating
least-squares mechanics are explicit and inspectable.

---

## A note on the data

All three datasets are **synthetic**, generated by simulators in each project's
`generate_data.py`. This was a deliberate choice, and it has a real cost that is worth
stating up front: **the reported metrics measure how well each model recovers a known
generative process, not how well it would predict real customers or real machines.**

What it buys is a ground truth that makes the engineering verifiable. Project 2's
simulator knows which failures were gradual and which were sudden, so detection rates can
be reported separately and the ~20% of failures with 2–8 hours of warning explain exactly
why recall has a ceiling. Project 3's simulator knows which items are cold-start, so the
content-based fallback can be measured on precisely the items it exists to serve. Project
1's simulator contains known interaction terms, which is what makes the
linear-vs-tree comparison a fair test rather than a foregone conclusion.

The simulators are also built to be *hard*. Noise terms are tuned so a good model lands
near 0.87 ROC-AUC rather than a meaningless 0.99; the raw data arrives with blank strings,
inconsistent category spellings, stuck sensors, duplicate rows and structurally missing
fields; and each generator's docstring states which properties it was designed to create.

The pipelines, the evaluation protocols and the deployment artefacts transfer directly to
real data. The numbers do not, and each README says so in its own limitations section.
