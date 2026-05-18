"""Student-owned dataset loading contract.

Loads the Online Shoppers Purchasing Intention dataset, applies leakage-safe
row-wise feature engineering, and returns a stratified train/test split.

The actual stateful preprocessing (scaling, encoding, ...) is **not** applied
here. It lives in ``features.build_feature_pipeline()`` and is meant to be
fitted on the train split inside the model training script — fitting it here
would leak test statistics into the train set.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR
from features import TARGET, add_engineered_features

DATASET_FILE = DATA_DIR / "online_shoppers_intention.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return ``(X_train, X_test, y_train, y_test)`` for the conversion task.

    - Reads the raw CSV from ``data/``.
    - Applies stateless row-wise feature engineering
      (see ``features.add_engineered_features``).
    - Splits 80/20, stratified on the target to preserve the ~85/15 class balance.

    Returns
    -------
    X_train, X_test : pandas.DataFrame
        Feature frames; same columns, ready to be fed to
        ``features.build_feature_pipeline()`` for fitting / transforming.
    y_train, y_test : pandas.Series of int
        Target ``Revenue`` cast to ``{0, 1}``.
    """
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_FILE}. "
            "Download it from "
            "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset"
        )

    df = pd.read_csv(DATASET_FILE)
    df = add_engineered_features(df)

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    # train_test_split returns a list — main.py validates against a tuple,
    # so we cast explicitly to match the documented contract.
    return tuple(
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )
