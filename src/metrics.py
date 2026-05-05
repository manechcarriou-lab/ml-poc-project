"""Metrics for the conversion-prediction task.

The dataset is heavily imbalanced (~85/15), so accuracy alone is misleading.
We compute a small panel of metrics that all measure something different about
the positive class (Revenue == True).
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return the metric panel used to compare models.

    All metrics target the positive class (purchase). ``roc_auc`` falls back
    to ``float('nan')`` when ``y_pred`` only contains class labels and not
    probability scores from the same dataset (still computable here because we
    pass binary predictions, but caller should prefer probabilities when
    available).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred),
    }
