"""Generate all the static visualizations used in the deck and the doc.

Outputs go to ``plots/`` and ``results/``:

- ``plots/eda_target_balance.png``
- ``plots/eda_conversion_by_segment.png``
- ``plots/feature_importance_xgb.png``
- ``plots/confusion_matrix_<model>.png`` (3 files)
- ``plots/roc_curves.png``
- ``plots/pr_curves.png``
- ``plots/threshold_tuning_xgb.png``
- ``results/model_metrics.csv``
- ``results/test_predictions.csv``  (raw probas for the Streamlit demo)

Run with:
    python scripts/generate_plots.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PLOTS = ROOT / "plots"
RESULTS = ROOT / "results"
DATA_FILE = ROOT / "data" / "online_shoppers_intention.csv"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


config = _load("config", SRC / "config.py"); sys.modules["config"] = config
features = _load("features", SRC / "features.py"); sys.modules["features"] = features
data_mod = _load("data", SRC / "data.py")

PLOTS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# 1. EDA snapshots (use the raw CSV — independent from the trained models)
# ---------------------------------------------------------------------------


def plot_target_balance() -> None:
    df = pd.read_csv(DATA_FILE)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    counts = df["Revenue"].value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    bars = ax.bar(
        ["No purchase", "Purchase"],
        counts.values,
        color=["#4c72b0", "#dd8452"],
    )
    for bar, p in zip(bars, pct.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            f"{p}%",
            ha="center", fontsize=12, fontweight="bold",
        )
    ax.set_title("Distribution of the target — heavy class imbalance", fontsize=13)
    ax.set_ylabel("Number of sessions")
    plt.tight_layout()
    fig.savefig(PLOTS / "eda_target_balance.png", dpi=150)
    plt.close(fig)


def plot_conversion_by_segment() -> None:
    df = pd.read_csv(DATA_FILE)
    overall = df["Revenue"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    cr_month = df.groupby("Month")["Revenue"].mean().reindex(
        ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    sns.barplot(x=cr_month.index, y=cr_month.values, ax=axes[0], color="#4c72b0")
    axes[0].axhline(overall, color="red", ls="--", lw=1, label=f"overall {overall:.1%}")
    axes[0].set_title("Conversion rate by month")
    axes[0].set_ylabel("Conversion rate")
    axes[0].legend()

    cr_vt = df.groupby("VisitorType")["Revenue"].mean().sort_values(ascending=False)
    sns.barplot(x=cr_vt.index, y=cr_vt.values, ax=axes[1], color="#dd8452")
    axes[1].axhline(overall, color="red", ls="--", lw=1, label=f"overall {overall:.1%}")
    axes[1].set_title("Conversion rate by visitor type")
    axes[1].set_ylabel("Conversion rate")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(PLOTS / "eda_conversion_by_segment.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Model evaluation plots (use trained pipelines from models/)
# ---------------------------------------------------------------------------


def _load_models(model_keys=("logreg", "random_forest", "xgboost")):
    return {k: joblib.load(config.MODELS[k]["path"]) for k in model_keys}


def plot_confusion_matrices(models: dict, X_test, y_test) -> None:
    for key, pipe in models.items():
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        disp = ConfusionMatrixDisplay(cm, display_labels=["No buy", "Buy"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"Confusion matrix — {config.MODELS[key]['name']}")
        plt.tight_layout()
        fig.savefig(PLOTS / f"confusion_matrix_{key}.png", dpi=150)
        plt.close(fig)


def plot_roc_curves(models: dict, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for key, pipe in models.items():
        y_score = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        ax.plot(fpr, tpr, lw=2, label=f"{config.MODELS[key]['name']} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — held-out test set")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(PLOTS / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_pr_curves(models: dict, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    baseline = float(np.mean(y_test))
    for key, pipe in models.items():
        y_score = pipe.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_score)
        ax.plot(rec, prec, lw=2, label=config.MODELS[key]["name"])
    ax.axhline(baseline, ls="--", color="gray", lw=1, label=f"baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curves — held-out test set")
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(PLOTS / "pr_curves.png", dpi=150)
    plt.close(fig)


def _inner_pipeline(model):
    """Unwrap TunedThresholdClassifierCV to get the underlying sklearn Pipeline."""
    return getattr(model, "estimator_", model)


def plot_xgb_feature_importance(models: dict) -> None:
    pipe = _inner_pipeline(models["xgboost"])
    pre = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["clf"]
    names = list(pre.get_feature_names_out())
    imp = clf.feature_importances_
    df = (
        pd.DataFrame({"feature": names, "importance": imp})
        .sort_values("importance", ascending=False)
        .head(15)
        .iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(df["feature"], df["importance"], color="#1f77b4")
    ax.set_title("Top 15 features — XGBoost (gain importance)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(PLOTS / "feature_importance_xgb.png", dpi=150)
    plt.close(fig)


def plot_threshold_tuning(models: dict, X_test, y_test) -> None:
    pipe = models["xgboost"]
    y_score = pipe.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
            }
        )
    df = pd.DataFrame(rows)
    best_t = df.loc[df["f1"].idxmax(), "threshold"]
    best_f1 = df["f1"].max()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(df["threshold"], df["precision"], label="precision", color="#4c72b0")
    ax.plot(df["threshold"], df["recall"], label="recall", color="#dd8452")
    ax.plot(df["threshold"], df["f1"], label="F1", color="#27ae60", lw=2)
    ax.axvline(0.5, ls=":", color="gray", lw=1, label="default threshold")
    ax.axvline(best_t, ls="--", color="red", lw=1, label=f"best F1 @ {best_t:.2f} = {best_f1:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric")
    ax.set_title("XGBoost — precision / recall / F1 vs decision threshold")
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(PLOTS / "threshold_tuning_xgb.png", dpi=150)
    plt.close(fig)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# 3. CSV deliverables for the Streamlit app + main.py
# ---------------------------------------------------------------------------


def write_metrics_csv(models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for key, pipe in models.items():
        y_pred = pipe.predict(X_test)
        y_score = pipe.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model_key": key,
                "model_name": config.MODELS[key]["name"],
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_score),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "model_metrics.csv", index=False)
    return df


def write_test_predictions(models: dict, X_test, y_test) -> None:
    df = X_test.copy()
    df["y_true"] = y_test.values
    for key, pipe in models.items():
        df[f"proba_{key}"] = pipe.predict_proba(X_test)[:, 1]
    df.to_csv(RESULTS / "test_predictions.csv", index=False)


# ---------------------------------------------------------------------------


def main() -> None:
    print("Generating EDA snapshots…")
    plot_target_balance()
    plot_conversion_by_segment()

    print("Loading trained models + test split…")
    models = _load_models()
    _, X_test, _, y_test = data_mod.load_dataset_split()

    print("Plotting confusion matrices, ROC, PR, feature importance, threshold tuning…")
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_pr_curves(models, X_test, y_test)
    plot_xgb_feature_importance(models)
    best_t, best_f1 = plot_threshold_tuning(models, X_test, y_test)
    print(f"  XGBoost best F1={best_f1:.4f} at threshold {best_t:.2f}")

    print("Writing model_metrics.csv + test_predictions.csv…")
    metrics_df = write_metrics_csv(models, X_test, y_test)
    write_test_predictions(models, X_test, y_test)

    print("\nResults summary:")
    print(metrics_df.to_string(index=False))
    print(f"\nAll outputs written to {PLOTS.relative_to(ROOT)} and {RESULTS.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
