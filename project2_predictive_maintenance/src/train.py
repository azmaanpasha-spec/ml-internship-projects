"""Step 4 of the approach: train classifiers and handle the rare-failure class.

Three decisions define this module, and each is the one a reviewer will question:

**1. The split is by time, never at random.**
A random split puts hour 09:00 in training and hour 10:00 in test. Those rows share a
24-hour rolling window, so the test set is contaminated and the score is fiction. Here
the first 70% of the calendar trains and the last 30% tests. A second, independent
*machine-based* split is also evaluated, which answers a different question: does the
model work on a machine it has never seen? Both numbers are reported, because a fleet
operator needs the answer to both.

**2. Selection is on PR-AUC, not ROC-AUC.**
Failures are ~7% of rows. ROC-AUC is dominated by the 93% of easy negatives and stays
flattering even when precision is useless; PR-AUC tracks exactly the quantity the
maintenance team feels -- of the alerts raised, how many were real.

**3. Class imbalance is handled with weights, and the alternative is measured.**
`class_weight="balanced"` / `scale_pos_weight` is the default here, but the module also
trains the winning family on randomly undersampled negatives so the two can be compared
on the same test set instead of the choice being asserted.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

from common.evaluation import classification_report  # noqa: E402
from common.io_utils import save_json, save_model  # noqa: E402
from src.features import feature_columns  # noqa: E402

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_FRACTION = 0.30
MACHINE_TEST_FRACTION = 0.25
CV_SPLITS = 3

# Search budget per family. Random forest gets fewer draws because each fit is an order
# of magnitude slower than a histogram-based booster's.
SEARCH_ITERATIONS = {
    "logistic_regression": 8,
    "random_forest": 4,
    "xgboost": 12,
    "lightgbm": 8,
}

# Hyperparameters are searched on the most recent slice of the training period, then the
# winning configuration is refitted on the *full* training set. Searching on all 113k
# rows x 152 features meant every parallel worker materialised a ~150 MB dense copy of
# the transformed matrix; twelve of those exhausted memory and the run spent its time
# swapping rather than fitting. The recent slice keeps the comparison between
# configurations fair while making the search affordable, and the final model still sees
# every training row.
SEARCH_SAMPLE_ROWS = 50_000
SEARCH_N_JOBS = 4


def time_based_split(
    frame: pd.DataFrame, test_fraction: float = TEST_FRACTION
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split on a single calendar cut-point shared by every machine.

    Every training row precedes every test row, which is the only split that reflects how
    the model would be used: fitted on history, applied to the hours that follow.
    """
    timestamps = pd.to_datetime(frame["timestamp"])
    cutoff = timestamps.quantile(1.0 - test_fraction)
    train = frame[timestamps <= cutoff]
    test = frame[timestamps > cutoff]
    return train, test, cutoff


def machine_based_split(
    frame: pd.DataFrame, test_fraction: float = MACHINE_TEST_FRACTION, seed: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole machines, to test generalisation to unseen equipment."""
    rng = np.random.default_rng(seed)
    machine_ids = np.sort(frame["machine_id"].unique())
    n_test = max(int(round(len(machine_ids) * test_fraction)), 1)
    held_out = set(rng.choice(machine_ids, size=n_test, replace=False))

    is_test = frame["machine_id"].isin(held_out)
    return frame[~is_test], frame[is_test]


def build_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    """Median imputation for the sensor features; one-hot for machine metadata."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def model_space(scale_pos_weight: float) -> Dict[str, dict]:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "logistic_regression": {
            "scale": True,
            "estimator": LogisticRegression(
                max_iter=1200, class_weight="balanced", random_state=RANDOM_STATE
            ),
            "grid": {"model__C": np.logspace(-3, 1, 12)},
        },
        "random_forest": {
            "scale": False,
            "estimator": RandomForestClassifier(
                class_weight="balanced_subsample",
                max_samples=0.6,           # bagging on a fraction keeps fits affordable
                random_state=RANDOM_STATE,
                # One core per forest: the search parallelises across folds instead, which
                # avoids nesting a thread pool inside a process pool.
                n_jobs=1,
            ),
            "grid": {
                "model__n_estimators": [120, 200],
                "model__max_depth": [8, 12],
                "model__min_samples_leaf": [50, 100],
                "model__max_features": ["sqrt"],
            },
        },
        "xgboost": {
            "scale": False,
            "estimator": XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            ),
            "grid": {
                "model__n_estimators": [300, 500, 700],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__subsample": [0.7, 0.9],
                "model__colsample_bytree": [0.5, 0.7, 0.9],
                "model__min_child_weight": [1, 5, 20],
                "model__reg_lambda": [1.0, 3.0],
            },
        },
        "lightgbm": {
            "scale": False,
            "estimator": LGBMClassifier(
                objective="binary",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                # Capped deliberately. LightGBM defaults to one thread per core, and
                # nesting that inside the search's process pool oversubscribes the CPU
                # badly -- 12 threads x 4 workers on 12 cores made this family run an
                # order of magnitude slower than XGBoost on the same grid.
                n_jobs=2,
                verbose=-1,
            ),
            "grid": {
                "model__n_estimators": [300, 500],
                "model__num_leaves": [31, 63],
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__min_child_samples": [20, 50, 100],
                "model__subsample": [0.7, 0.9],
                "model__colsample_bytree": [0.5, 0.7, 0.9],
            },
        },
    }


def undersample(
    X: pd.DataFrame, y: pd.Series, negative_ratio: float = 3.0, seed: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.Series]:
    """Keep every positive and `negative_ratio` negatives per positive."""
    rng = np.random.default_rng(seed)
    positive_index = y.index[y == 1]
    negative_index = y.index[y == 0]
    keep_negatives = rng.choice(
        negative_index,
        size=min(int(len(positive_index) * negative_ratio), len(negative_index)),
        replace=False,
    )
    keep = np.concatenate([positive_index.to_numpy(), keep_negatives])
    return X.loc[keep], y.loc[keep]


def train_all(frame: pd.DataFrame, out_dir: Path) -> dict:
    numeric, categorical = feature_columns(frame)
    train, test, cutoff = time_based_split(frame)

    X_train, y_train = train[numeric + categorical], train["label"].astype(int)
    X_test, y_test = test[numeric + categorical], test["label"].astype(int)
    y_test_array = y_test.to_numpy()

    # The search slice is the most recent SEARCH_SAMPLE_ROWS hours of the training period,
    # kept in time order so TimeSeriesSplit still sees a valid chronology.
    train_ordered = train.sort_values("timestamp")
    search_slice = train_ordered.tail(min(SEARCH_SAMPLE_ROWS, len(train_ordered)))
    X_search = search_slice[numeric + categorical]
    y_search = search_slice["label"].astype(int)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"time-based split at {cutoff}")
    print(f"  train: {len(X_train):,} rows, {y_train.sum():,} positives ({y_train.mean():.2%})")
    print(f"  test : {len(X_test):,} rows, {y_test.sum():,} positives ({y_test.mean():.2%})")
    print(f"  search slice: {len(X_search):,} most recent training rows "
          f"({y_search.mean():.2%} positive); winner refitted on all {len(X_train):,}")
    print(f"features: {len(numeric)} numeric + {len(categorical)} categorical")
    print(f"scale_pos_weight: {scale_pos_weight:.1f}\n")

    # TimeSeriesSplit keeps every validation fold strictly after its training fold.
    cv = TimeSeriesSplit(n_splits=CV_SPLITS)
    results, fitted = {}, {}

    for name, spec in model_space(scale_pos_weight).items():
        started = time.time()
        pipeline = Pipeline(
            [
                ("prep", build_preprocessor(numeric, categorical, scale=spec["scale"])),
                ("model", spec["estimator"]),
            ]
        )
        search = RandomizedSearchCV(
            pipeline,
            spec["grid"],
            n_iter=SEARCH_ITERATIONS[name],
            scoring="average_precision",      # PR-AUC, the right metric for rare events
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=SEARCH_N_JOBS,
            refit=False,                      # refit happens below, on the full training set
        )
        search.fit(X_search, y_search)

        # Refit the winning configuration on every training row.
        best = Pipeline(
            [
                ("prep", build_preprocessor(numeric, categorical, scale=spec["scale"])),
                ("model", spec["estimator"]),
            ]
        )
        best.set_params(**search.best_params_)
        best.fit(X_train, y_train)

        scores = best.predict_proba(X_test)[:, 1]
        report = classification_report(y_test_array, scores, model=name, threshold=0.5)
        report.extra["cv_pr_auc"] = float(search.best_score_)
        report.extra["fit_seconds"] = round(time.time() - started, 1)

        results[name] = report.as_dict()
        results[name]["best_params"] = {
            key.replace("model__", ""): value for key, value in search.best_params_.items()
        }
        fitted[name] = best

        print(
            f"{name:<20} cv_pr_auc={search.best_score_:.4f}  test_pr_auc={report.pr_auc:.4f}  "
            f"roc_auc={report.roc_auc:.4f}  ({report.extra['fit_seconds']}s)",
            flush=True,
        )
        save_model(best, out_dir / f"{name}.joblib")

    best_name = max(results, key=lambda name: results[name]["pr_auc"])
    print(f"\nbest by PR-AUC: {best_name}")

    # --- ablation: class weights vs. undersampling -----------------------------------
    print("\nablation: class weighting vs. random undersampling")
    X_under, y_under = undersample(X_train, y_train)
    unweighted = model_space(1.0)[best_name]
    under_pipeline = Pipeline(
        [
            ("prep", build_preprocessor(numeric, categorical, scale=unweighted["scale"])),
            ("model", unweighted["estimator"]),
        ]
    )
    # Apply the tuned hyperparameters so only the imbalance strategy differs.
    under_pipeline.set_params(
        **{f"model__{key}": value for key, value in results[best_name]["best_params"].items()}
    )
    if hasattr(under_pipeline.named_steps["model"], "class_weight"):
        under_pipeline.set_params(model__class_weight=None)
    under_pipeline.fit(X_under, y_under)
    under_scores = under_pipeline.predict_proba(X_test)[:, 1]
    under_report = classification_report(
        y_test_array, under_scores, model=f"{best_name}_undersampled", threshold=0.5
    )
    results[f"{best_name}_undersampled"] = under_report.as_dict()
    print(f"  class weights : pr_auc={results[best_name]['pr_auc']:.4f}  "
          f"brier={results[best_name]['brier']:.4f}")
    print(f"  undersampled  : pr_auc={under_report.pr_auc:.4f}  brier={under_report.brier:.4f}  "
          f"(trained on {len(X_under):,} of {len(X_train):,} rows)")

    # --- robustness: generalisation to machines never seen in training ----------------
    print("\nrobustness: held-out machines (different question, different split)")
    machine_train, machine_test = machine_based_split(frame)
    machine_pipeline = Pipeline(
        [
            ("prep", build_preprocessor(numeric, categorical, scale=model_space(1.0)[best_name]["scale"])),
            ("model", model_space(
                float((machine_train["label"] == 0).sum() / max((machine_train["label"] == 1).sum(), 1))
            )[best_name]["estimator"]),
        ]
    )
    machine_pipeline.set_params(
        **{f"model__{key}": value for key, value in results[best_name]["best_params"].items()}
    )
    machine_pipeline.fit(
        machine_train[numeric + categorical], machine_train["label"].astype(int)
    )
    machine_scores = machine_pipeline.predict_proba(machine_test[numeric + categorical])[:, 1]
    machine_report = classification_report(
        machine_test["label"].astype(int).to_numpy(),
        machine_scores,
        model=f"{best_name}_unseen_machines",
        threshold=0.5,
    )
    results[f"{best_name}_unseen_machines"] = machine_report.as_dict()
    print(f"  unseen machines: pr_auc={machine_report.pr_auc:.4f}  "
          f"roc_auc={machine_report.roc_auc:.4f}  "
          f"({machine_test['machine_id'].nunique()} machines held out)")

    # --- calibration ------------------------------------------------------------------
    calibrated = CalibratedClassifierCV(fitted[best_name], method="isotonic", cv=cv)
    calibrated.fit(X_train, y_train)
    calibrated_scores = calibrated.predict_proba(X_test)[:, 1]
    calibrated_report = classification_report(
        y_test_array, calibrated_scores, model=f"{best_name}_calibrated", threshold=0.5
    )
    results[f"{best_name}_calibrated"] = calibrated_report.as_dict()
    print(
        f"\nafter isotonic calibration: pr_auc={calibrated_report.pr_auc:.4f}  "
        f"brier={calibrated_report.brier:.4f} (was {results[best_name]['brier']:.4f})"
    )

    save_model(calibrated, out_dir / "best_model.joblib")
    save_json(
        {
            "best_model": best_name,
            "calibrated": True,
            "split": "time_based",
            "split_cutoff": str(cutoff),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "train_positive_rate": round(float(y_train.mean()), 5),
            "test_positive_rate": round(float(y_test.mean()), 5),
            "numeric_features": numeric,
            "categorical_features": categorical,
            "random_state": RANDOM_STATE,
        },
        out_dir / "model_card.json",
    )
    return results


def main() -> None:
    here = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Train predictive maintenance models.")
    parser.add_argument(
        "--features", type=Path, default=here / "data" / "processed" / "features.parquet"
    )
    parser.add_argument("--models", type=Path, default=here / "models")
    parser.add_argument("--reports", type=Path, default=here / "reports")
    args = parser.parse_args()

    frame = pd.read_parquet(args.features)
    results = train_all(frame, args.models)
    save_json(results, args.reports / "training_results.json")
    print(f"\nwrote metrics -> {args.reports / 'training_results.json'}")


if __name__ == "__main__":
    main()
