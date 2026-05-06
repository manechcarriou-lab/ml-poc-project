"""Train and tune the model zoo with Optuna + MLflow tracking + threshold tuning.

Pipeline per family
-------------------
1. **Optuna** TPE study, 3-fold stratified CV, F1 (positive class) as objective.
   Each trial fits the full ``preprocessor → classifier`` pipeline and is logged
   as a nested MLflow run.

2. **Per-family encoder** — empirically validated in
   ``notebooks/encoding_comparison.ipynb``:
       - logreg          → OneHotEncoder (linear models need additive encoding)
       - random_forest   → OneHotEncoder (best on test, marginal vs ordinal)
       - xgboost         → OrdinalEncoder (compact 24-dim space, +2 F1 pts)

3. **Threshold tuning** — after the best params are found and the model is
   refit on the full train, we wrap it in
   ``TunedThresholdClassifierCV(scoring='f1', cv=5)``. The wrapper is fit on the
   train data only (its CV stays inside train), then exposes a ``predict()``
   that uses the optimal F1 threshold instead of the default 0.5.

4. **Final eval** — predictions on the held-out test set with the tuned
   threshold are logged to MLflow and the wrapped pipeline is saved as
   ``models/<family>.joblib``.

Inspect runs:
    mlflow ui --backend-store-uri ./mlruns

Retrain everything:
    python scripts/train.py [--trials 15] [--seed 42] [--families ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import joblib
import mlflow
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    TunedThresholdClassifierCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Module loading (mirrors scripts/main.py to make the project's src imports work)
# ---------------------------------------------------------------------------


def _load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module `{module_name}` from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

config = _load_module("config", SRC_DIR / "config.py")
sys.modules["config"] = config

features = _load_module("features", SRC_DIR / "features.py")
sys.modules["features"] = features

data = _load_module("data", SRC_DIR / "data.py")

build_preprocessor = features.build_preprocessor
load_dataset_split = data.load_dataset_split

MODELS_DIR = config.MODELS_DIR
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


# ---------------------------------------------------------------------------
# Default encoder per family — the result of the To-Do 4 comparison study.
# ---------------------------------------------------------------------------

DEFAULT_ENCODER_PER_FAMILY = {
    "logreg": "onehot",
    "random_forest": "onehot",
    "xgboost": "ordinal",
}


# ---------------------------------------------------------------------------
# Optuna search spaces
# ---------------------------------------------------------------------------


def _suggest_logreg(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l2"]),
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": 42,
    }


def _suggest_rf(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 24),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    }


def _suggest_xgb(trial: optuna.Trial, scale_pos_weight: float) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def _make_pipeline(estimator, encoder: str) -> Pipeline:
    return Pipeline(
        steps=[("preprocessor", build_preprocessor(encoder)), ("clf", estimator)]
    )


def _make_objective(
    family: str,
    encoder: str,
    suggest: Callable[[optuna.Trial], dict[str, Any]],
    estimator_cls,
    X_train,
    y_train,
    cv: StratifiedKFold,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        pipeline = _make_pipeline(estimator_cls(**params), encoder)

        with mlflow.start_run(run_name=f"{family}-trial-{trial.number}", nested=True):
            mlflow.log_param("encoder", encoder)
            mlflow.log_params({f"{family}__{k}": v for k, v in params.items()})
            mlflow.log_param("cv_folds", cv.get_n_splits())

            scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
            )
            mean_f1, std_f1 = float(scores.mean()), float(scores.std())
            mlflow.log_metric("cv_f1_mean", mean_f1)
            mlflow.log_metric("cv_f1_std", std_f1)

        return mean_f1

    return objective


def _fit_and_log_final(
    family: str,
    encoder: str,
    estimator_cls,
    best_params: dict[str, Any],
    X_train,
    X_test,
    y_train,
    y_test,
):
    pipeline = _make_pipeline(estimator_cls(**best_params), encoder)

    # Wrap in a CV threshold tuner — this fits the inner pipeline on train
    # *and* searches the optimal F1 threshold on the same train via 5-fold CV.
    # No test data ever touches it.
    tuned = TunedThresholdClassifierCV(
        pipeline, scoring="f1", cv=5, n_jobs=-1, random_state=42
    )

    with mlflow.start_run(run_name=f"{family}-final"):
        mlflow.log_param("encoder", encoder)
        mlflow.log_params({f"{family}__{k}": v for k, v in best_params.items()})

        tuned.fit(X_train, y_train)
        best_threshold = float(tuned.best_threshold_)
        mlflow.log_metric("best_threshold", best_threshold)

        y_pred = tuned.predict(X_test)
        y_score = tuned.predict_proba(X_test)[:, 1]

        # Report metrics at the tuned threshold (= what tuned.predict uses)
        # AND at the default 0.5 for transparent comparison.
        y_pred_default = (y_score >= 0.5).astype(int)
        test_metrics = {
            "test_f1": float(f1_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, y_score)),
            "test_f1_at_0p5": float(f1_score(y_test, y_pred_default)),
        }
        mlflow.log_metrics(test_metrics)

        save_path = MODELS_DIR / f"{family}.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(tuned, save_path)
        mlflow.log_artifact(str(save_path), artifact_path="model")

    return tuned, test_metrics, best_threshold, save_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run_family(
    family: str,
    encoder: str,
    suggest_fn: Callable[[optuna.Trial], dict[str, Any]],
    estimator_cls,
    X_train,
    X_test,
    y_train,
    y_test,
    n_trials: int,
    seed: int,
) -> dict[str, Any]:
    print(f"\n=== {family} ({encoder} encoding) - Optuna study ({n_trials} trials) ===")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=family)

    with mlflow.start_run(run_name=f"{family}-study"):
        mlflow.log_param("model_family", family)
        mlflow.log_param("encoder", encoder)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv_folds", cv.get_n_splits())

        objective = _make_objective(
            family, encoder, suggest_fn, estimator_cls, X_train, y_train, cv
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_trial
        mlflow.log_metric("best_cv_f1", float(best.value))
        mlflow.log_params({f"best_{family}__{k}": v for k, v in best.params.items()})

        print(f"  best CV F1 = {best.value:.4f} | params = {best.params}")

    tuned, test_metrics, best_threshold, save_path = _fit_and_log_final(
        family, encoder, estimator_cls, best.params, X_train, X_test, y_train, y_test
    )

    print(
        f"  best threshold = {best_threshold:.3f}  (vs 0.5 default)\n"
        f"  test F1={test_metrics['test_f1']:.4f}  "
        f"(@0.5 = {test_metrics['test_f1_at_0p5']:.4f})  "
        f"precision={test_metrics['test_precision']:.4f}  "
        f"recall={test_metrics['test_recall']:.4f}  "
        f"roc_auc={test_metrics['test_roc_auc']:.4f}"
    )
    print(f"  saved -> {save_path.relative_to(PROJECT_ROOT)}")

    return {
        "family": family,
        "encoder": encoder,
        "best_cv_f1": float(best.value),
        "best_params": best.params,
        "best_threshold": best_threshold,
        "test_metrics": test_metrics,
        "model_path": str(save_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=15, help="Optuna trials per model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["logreg", "random_forest", "xgboost"],
        choices=["logreg", "random_forest", "xgboost"],
    )
    parser.add_argument(
        "--encoder",
        default="auto",
        choices=["auto", "onehot", "ordinal", "target"],
        help="'auto' uses the empirically best encoder per family. "
             "Specify a single encoder to override for all families.",
    )
    args = parser.parse_args()

    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("online_shoppers_conversion")

    print("Loading dataset split...")
    X_train, X_test, y_train, y_test = load_dataset_split()
    print(f"  train={X_train.shape} test={X_test.shape}")

    pos_rate = float(y_train.mean())
    scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-9)

    family_dispatch = {
        "logreg": (_suggest_logreg, LogisticRegression),
        "random_forest": (_suggest_rf, RandomForestClassifier),
        "xgboost": (lambda t: _suggest_xgb(t, scale_pos_weight), XGBClassifier),
    }

    summaries = []
    for fam in args.families:
        encoder = (
            DEFAULT_ENCODER_PER_FAMILY[fam] if args.encoder == "auto" else args.encoder
        )
        suggest_fn, estimator_cls = family_dispatch[fam]
        summary = _run_family(
            fam,
            encoder,
            suggest_fn,
            estimator_cls,
            X_train, X_test, y_train, y_test,
            n_trials=args.trials,
            seed=args.seed,
        )
        summaries.append(summary)

    print("\n=== Summary ===")
    for s in summaries:
        m = s["test_metrics"]
        print(
            f"{s['family']:<14} encoder={s['encoder']:<8} "
            f"thr={s['best_threshold']:.3f}  "
            f"test_f1={m['test_f1']:.4f}  test_roc_auc={m['test_roc_auc']:.4f}  "
            f"-> {Path(s['model_path']).name}"
        )

    best = max(summaries, key=lambda s: s["test_metrics"]["test_f1"])
    print(
        f"\nBest family by test F1: {best['family']} "
        f"({best['test_metrics']['test_f1']:.4f}, threshold={best['best_threshold']:.3f}, "
        f"encoder={best['encoder']})"
    )
    print("Inspect runs with:  mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
