"""Feature engineering & preprocessing pipeline for the Online Shoppers dataset.

The single entry point is ``build_preprocessor(encoder=...)``. Pick the encoder
that matches your model family (justified empirically in
``notebooks/encoding_comparison.ipynb``):

- ``onehot`` (default) — best for linear models and the most robust overall.
- ``ordinal`` — best for tree-based models. Compact (24 features vs 82) and
  trees handle non-monotonic splits regardless of the integer ordering.
- ``target`` — leakage-safe via ``TargetEncoder``'s internal cross-validation.

Every transformation is wrapped in a ``Pipeline`` so it can be ``fit`` on the
train split and ``transform`` on the test split — no leakage by construction.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

EncoderName = Literal["onehot", "ordinal", "target"]


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

TARGET: str = "Revenue"


# ---------------------------------------------------------------------------
# Hand-crafted feature engineering done before the sklearn pipeline.
# Row-wise stateless transforms — fit-free, so leakage-safe.
# ---------------------------------------------------------------------------


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Row-wise stateless → leakage-safe."""
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


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------


def _log1p_transformer() -> FunctionTransformer:
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one", validate=False)


def _categorical_block(encoder: EncoderName):
    """Return the (transformer, columns) pair for the categorical block."""
    if encoder == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if encoder == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if encoder == "target":
        # CV-internal — no leakage. cv=5 is the sklearn default.
        return TargetEncoder(target_type="binary", random_state=42, cv=5)
    raise ValueError(f"Unknown encoder: {encoder!r}. Use 'onehot', 'ordinal', or 'target'.")


def build_preprocessor(encoder: EncoderName = "onehot") -> ColumnTransformer:
    """Return a ColumnTransformer that handles every feature type.

    Parameters
    ----------
    encoder
        Strategy for ``CATEGORICAL_FEATURES``. See module docstring for guidance.

    Numeric processing is identical across encoders:
    - Skewed numeric columns: ``log1p`` then ``StandardScaler``
    - Other numeric columns + engineered aggregates: ``StandardScaler``
    - Engineered binary flags: passthrough.
    """
    skewed_pipeline = Pipeline(
        steps=[("log1p", _log1p_transformer()), ("scale", StandardScaler())]
    )

    plain_numeric = [c for c in NUMERIC_FEATURES if c not in SKEWED_NUMERIC_FEATURES]

    return ColumnTransformer(
        transformers=[
            ("skewed_num", skewed_pipeline, SKEWED_NUMERIC_FEATURES),
            ("num", StandardScaler(), plain_numeric + ENGINEERED_NUMERIC_FEATURES),
            ("cat", _categorical_block(encoder), CATEGORICAL_FEATURES),
            ("binary_passthrough", "passthrough", ENGINEERED_BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline(encoder: EncoderName = "onehot") -> Pipeline:
    """Convenience wrapper exposing the preprocessor as a Pipeline."""
    return Pipeline(steps=[("preprocessor", build_preprocessor(encoder))])
