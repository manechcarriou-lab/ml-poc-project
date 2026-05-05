"""Feature engineering & preprocessing pipeline for the Online Shoppers dataset.

Goal
----
Centralize every transformation applied to features so that train and test get
*exactly* the same treatment (no leakage). The returned object is a
``sklearn.pipeline.Pipeline`` so it can be ``fit`` on the train set and
``transform`` on the test set without re-deriving statistics on the test data.

Why a single pipeline
---------------------
- ``StandardScaler`` learns mean/std on train only, applies them to test.
- ``OneHotEncoder`` learns categories on train only.
- Any future ``TargetEncoder`` / ``PCA`` / dropping step plugs into the same
  pipeline, fit once, transformed twice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Column groups (single source of truth for the rest of the project).
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: list[str] = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

# Long-tail / heavily skewed numeric columns: log1p before scaling stabilizes them.
SKEWED_NUMERIC_FEATURES: list[str] = [
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated_Duration",
    "PageValues",
]

CATEGORICAL_FEATURES: list[str] = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

TARGET: str = "Revenue"


# ---------------------------------------------------------------------------
# Hand-crafted feature engineering done before the sklearn pipeline.
# These are deterministic row-wise transforms — no fit needed, so leakage-safe.
# ---------------------------------------------------------------------------


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add new features derived from existing ones.

    All transforms here are row-wise and stateless, so they can be applied
    identically to train and test without leakage.
    """
    df = df.copy()

    total_pages = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    total_duration = (
        df["Administrative_Duration"]
        + df["Informational_Duration"]
        + df["ProductRelated_Duration"]
    )

    df["TotalPages"] = total_pages
    df["TotalDuration"] = total_duration
    df["AvgTimePerPage"] = np.where(total_pages > 0, total_duration / total_pages, 0.0)
    df["ProductRelatedRatio"] = np.where(
        total_pages > 0, df["ProductRelated"] / total_pages, 0.0
    )

    df["HighPageValue"] = (df["PageValues"] > 0).astype(int)
    df["IsHighBounce"] = (df["BounceRates"] > df["BounceRates"].quantile(0.75)).astype(int)
    df["IsSpecialDay"] = (df["SpecialDay"] > 0).astype(int)

    return df


ENGINEERED_NUMERIC_FEATURES: list[str] = [
    "TotalPages",
    "TotalDuration",
    "AvgTimePerPage",
    "ProductRelatedRatio",
]

ENGINEERED_BINARY_FEATURES: list[str] = [
    "HighPageValue",
    "IsHighBounce",
    "IsSpecialDay",
]


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------


def _log1p_transformer() -> FunctionTransformer:
    """Stateless log1p — safe for both train and test."""
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one", validate=False)


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that handles all feature types.

    - Skewed numeric columns: log1p -> StandardScaler
    - Other numeric columns: StandardScaler
    - Categorical columns: OneHotEncoder (handle_unknown='ignore' for robustness)
    - Engineered binary flags: passed through as-is
    """
    skewed_pipeline = Pipeline(
        steps=[
            ("log1p", _log1p_transformer()),
            ("scale", StandardScaler()),
        ]
    )

    plain_numeric = [c for c in NUMERIC_FEATURES if c not in SKEWED_NUMERIC_FEATURES]

    preprocessor = ColumnTransformer(
        transformers=[
            ("skewed_num", skewed_pipeline, SKEWED_NUMERIC_FEATURES),
            ("num", StandardScaler(), plain_numeric + ENGINEERED_NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("binary_passthrough", "passthrough", ENGINEERED_BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_feature_pipeline() -> Pipeline:
    """Full preprocessing pipeline (engineered features + ColumnTransformer).

    Note: the row-wise feature engineering (``add_engineered_features``) is
    stateless, so applying it before fitting the pipeline is leakage-safe.
    The ``Pipeline`` object itself only contains the stateful transformers.
    """
    return Pipeline(steps=[("preprocessor", build_preprocessor())])
