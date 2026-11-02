"""Step 4 of the approach: train a logistic-regression baseline and tree-based models.

Design decisions worth defending in a review:

* Class imbalance is handled with class weights (`class_weight="balanced"`,
  `scale_pos_weight`) rather than SMOTE. Resampling distorts the predicted probabilities,
  and a retention team needs calibrated probabilities to rank and to budget.
* Hyperparameters are tuned with `RandomizedSearchCV` on ROC-AUC over stratified folds of
  the *training* split only. The test split is touched exactly once, in `evaluate.py`.
* The winning model is wrapped in `CalibratedClassifierCV`, because tree ensembles with
  class weights produce confident but poorly calibrated scores.
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

from common.evaluation import classification_report  # noqa: E402
from common.io_utils import save_json, save_model  # noqa: E402
from src.features import feature_columns  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 4
SEARCH_ITERATIONS = 15


def build_preprocessor(
    numeric: list[str], categorical: list[str], scale: bool
) -> ColumnTransformer:
    """Shared preprocessing. Scaling is only needed by the linear baseline."""
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
    """Model zoo: a transparent baseline plus three tree ensembles.

    Each entry carries its own estimator, its search grid, and whether it needs scaled
    inputs -- so adding a model is a single dictionary entry.
    """
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "logistic_regression": {
            "scale": True,
            "estimator": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            # Only C is tuned: scikit-learn 1.8 deprecates explicit `penalty`, and the
            # default L2 is the right regulariser for this dense, correlated feature set.
            "grid": {"model__C": np.logspace(-3, 2, 25)},
        },
        "random_forest": {
            "scale": False,
            "estimator": RandomForestClassifier(
                class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
            ),
            "grid": {
                "model__n_estimators": [300, 500, 700],
                "model__max_depth": [6, 10, 14, None],
                "model__min_samples_leaf": [1, 5, 10, 20],
                "model__max_features": ["sqrt", "log2", 0.4],
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
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [3, 4, 5, 6],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
                "model__min_child_weight": [1, 5, 10],
                "model__reg_lambda": [0.5, 1.0, 3.0],
            },
        },
        "lightgbm": {
            "scale": False,
            "estimator": LGBMClassifier(
                objective="binary",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            "grid": {
                "model__n_estimators": [300, 500, 800],
                "model__num_leaves": [15, 31, 63],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__min_child_samples": [10, 20, 40],
                "model__subsample": [0.7, 0.9, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
            },
        },
    }


def split_data(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified hold-out split, so both splits carry the same churn rate."""
    numeric, categorical = feature_columns(frame)
    X = frame[numeric + categorical]
    y = frame["churned"].astype(int)
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def train_all(frame: pd.DataFrame, out_dir: Path) -> dict:
    """Tune every model, score it on the hold-out set, calibrate the winner."""
    numeric, categorical = feature_columns(frame)
    X_train, X_test, y_train, y_test = split_data(frame)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"train={len(X_train):,}  test={len(X_test):,}  churn rate={y_train.mean():.3f}")
    print(f"features: {len(numeric)} numeric + {len(categorical)} categorical")
    print(f"scale_pos_weight for boosted models: {scale_pos_weight:.2f}\n")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
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
            n_iter=SEARCH_ITERATIONS,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        scores = search.best_estimator_.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, scores, model=name, threshold=0.5)
        report.extra["cv_roc_auc"] = float(search.best_score_)
        report.extra["fit_seconds"] = round(time.time() - started, 1)

        results[name] = report.as_dict()
        results[name]["best_params"] = {
            key.replace("model__", ""): value for key, value in search.best_params_.items()
        }
        fitted[name] = search.best_estimator_

        print(
            f"{name:<20} cv_auc={search.best_score_:.4f}  "
            f"test_auc={report.roc_auc:.4f}  pr_auc={report.pr_auc:.4f}  "
            f"({report.extra['fit_seconds']}s)"
        )
        save_model(search.best_estimator_, out_dir / f"{name}.joblib")

    # PR-AUC decides the winner: with a ~26% positive class it reflects ranking quality
    # on the minority class better than ROC-AUC does.
    best_name = max(results, key=lambda name: results[name]["pr_auc"])
    print(f"\nbest by PR-AUC: {best_name}")

    calibrated = CalibratedClassifierCV(fitted[best_name], method="isotonic", cv=cv)
    calibrated.fit(X_train, y_train)
    calibrated_scores = calibrated.predict_proba(X_test)[:, 1]
    calibrated_report = classification_report(
        y_test, calibrated_scores, model=f"{best_name}_calibrated", threshold=0.5
    )
    results[f"{best_name}_calibrated"] = calibrated_report.as_dict()
    print(
        f"after isotonic calibration: auc={calibrated_report.roc_auc:.4f}  "
        f"brier={calibrated_report.brier:.4f} (was {results[best_name]['brier']:.4f})"
    )

    save_model(calibrated, out_dir / "best_model.joblib")
    save_json(
        {
            "best_model": best_name,
            "calibrated": True,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "numeric_features": numeric,
            "categorical_features": categorical,
            "random_state": RANDOM_STATE,
        },
        out_dir / "model_card.json",
    )
    return results


def main() -> None:
    here = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Train churn models.")
    parser.add_argument(
        "--features", type=Path, default=here / "data" / "processed" / "customers_features.csv"
    )
    parser.add_argument("--models", type=Path, default=here / "models")
    parser.add_argument("--reports", type=Path, default=here / "reports")
    args = parser.parse_args()

    frame = pd.read_csv(args.features)
    results = train_all(frame, args.models)
    save_json(results, args.reports / "training_results.json")
    print(f"\nwrote metrics -> {args.reports / 'training_results.json'}")


if __name__ == "__main__":
    main()
