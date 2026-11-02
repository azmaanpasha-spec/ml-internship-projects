"""Model-agnostic SHAP helper.

Both churn modules need per-feature attributions, and which explainer applies depends on
which model won the comparison -- `TreeExplainer` for the ensembles, `LinearExplainer`
for the logistic baseline. This keeps that choice in one place and degrades to `None`
rather than raising, so an explanation problem never breaks a pipeline.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def shap_values_for_pipeline(
    pipeline, X: pd.DataFrame, background_size: int = 200
) -> Tuple[np.ndarray, List[str]] | Tuple[None, None]:
    """SHAP values in the pipeline's *transformed* feature space.

    Returns `(values, feature_names)` with one row per input row, or `(None, None)` if
    SHAP is unavailable or cannot handle this estimator.
    """
    try:
        import shap
    except ImportError:
        return None, None

    try:
        prep = pipeline.named_steps["prep"]
        model = pipeline.named_steps["model"]
        transformed = np.asarray(prep.transform(X), dtype=float)
        names = [name.split("__", 1)[-1] for name in prep.get_feature_names_out()]

        if hasattr(model, "coef_"):
            # Linear / logistic: explain against a sample of the data as the background.
            background = transformed[: min(background_size, len(transformed))]
            explainer = shap.LinearExplainer(model, background)
        else:
            explainer = shap.TreeExplainer(model)

        values = explainer.shap_values(transformed)
        if isinstance(values, list):          # one array per class in some versions
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:                  # (n_rows, n_features, n_classes)
            values = values[:, :, -1]
        return values, names
    except Exception:  # noqa: BLE001 - explanations are supporting evidence, not critical path
        return None, None


def aggregate_to_source_columns(
    values: np.ndarray, names: List[str], source_columns: List[str]
) -> pd.DataFrame:
    """Sum one-hot SHAP columns back into the original feature they came from.

    So the explanation reads `contract`, not `contract_Month-to-month`.
    """
    mapped: List[str] = []
    for name in names:
        match = next(
            (
                column
                for column in source_columns
                if name == column or name.startswith(f"{column}_")
            ),
            name,
        )
        mapped.append(match)

    frame = pd.DataFrame(values, columns=mapped)
    aggregated = frame.T.groupby(level=0).sum().T
    return aggregated.reindex(columns=source_columns, fill_value=0.0)
