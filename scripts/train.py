"""Train and tune the model zoo with Optuna + MLflow tracking.

What this script does
---------------------
1. Loads the dataset split (``src/data.load_dataset_split``) — 80/20 stratified.
2. For each model family (Logistic Regression, Random Forest, XGBoost):
   - Runs an Optuna study that samples hyperparameters,
   - Each trial fits a *full pipeline* (preprocessor + classifier) on the train
     split, evaluates with 3-fold stratified CV using F1 (positive class) as
     the objective,
   - Logs every trial (params + metrics) to MLflow under one run per trial.
3. Refits the best pipeline of each family on the entire train split, evaluates
   on the held-out test split, logs the final test metrics + the saved
   pipeline as an MLflow artifact and as a ``.joblib`` file in ``models/``.
4. Writes a summary line per model family.

How to inspect the results
--------------------------
    mlflow ui --backend-store-uri ./mlruns

Then open http://localhost:5000.

How to retrain
--------------
    python scripts/train.py [--trials 20] [--seed 42]
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Module loading (mirrors scripts/main.py to make the project's `src` imports work)
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


def _make_pipeline(estimator) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("clf", estimator)])


# ---------------------------------------------------------------------------
# Optuna objective with MLflow logging per trial
# ---------------------------------------------------------------------------


def _make_objective(
    family: str,
    suggest: Callable[[optuna.Trial], dict[str, Any]],
    estimator_cls,
    X_train,
    y_train,
    cv: StratifiedKFold,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        pipeline = _make_pipeline(estimator_cls(**params))

        with mlflow.start_run(run_name=f"{family}-trial-{trial.number}", nested=True):
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


# ---------------------------------------------------------------------------
# Final fit + test eval + MLflow + joblib save
# ---------------------------------------------------------------------------


def _fit_and_log_final(
    family: str,
    estimator_cls,
    best_params: dict[str, Any],
    extra_params: dict[str, Any],
    X_train,
    X_test,
    y_train,
    y_test,
) -> tuple[Pipeline, dict[str, float], Path]:
    full_params = {**best_params, **extra_params}
    pipeline = _make_pipeline(estimator_cls(**full_params))

    with mlflow.start_run(run_name=f"{family}-final"):
        mlflow.log_params({f"{family}__{k}": v for k, v in full_params.items()})

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]

        test_metrics = {
            "test_f1": float(f1_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, y_score)),
        }
        mlflow.log_metrics(test_metrics)

        save_path = MODELS_DIR / f"{family}.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)
        mlflow.log_artifact(str(save_path), artifact_path="model")

    return pipeline, test_metrics, save_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run_family(
    family: str,
    suggest_fn: Callable[[optuna.Trial], dict[str, Any]],
    estimator_cls,
    X_train,
    X_test,
    y_train,
    y_test,
    n_trials: int,
    seed: int,
    extra_fit_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    print(f"\n=== {family} — Optuna study ({n_trials} trials) ===")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=family)

    with mlflow.start_run(run_name=f"{family}-study"):
        mlflow.log_param("model_family", family)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv_folds", cv.get_n_splits())

        objective = _make_objective(
            family, suggest_fn, estimator_cls, X_train, y_train, cv
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_trial
        mlflow.log_metric("best_cv_f1", float(best.value))
        mlflow.log_params({f"best_{family}__{k}": v for k, v in best.params.items()})

        print(f"  best CV F1 = {best.value:.4f} | params = {best.params}")

    pipeline, test_metrics, save_path = _fit_and_log_final(
        family,
        estimator_cls,
        best.params,
        extra_fit_params or {},
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print(
        f"  test F1={test_metrics['test_f1']:.4f} "
        f"precision={test_metrics['test_precision']:.4f} "
        f"recall={test_metrics['test_recall']:.4f} "
        f"roc_auc={test_metrics['test_roc_auc']:.4f}"
    )
    print(f"  saved -> {save_path.relative_to(PROJECT_ROOT)}")

    return {
        "family": family,
        "best_cv_f1": float(best.value),
        "best_params": best.params,
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
    args = parser.parse_args()

    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("online_shoppers_conversion")

    print("Loading dataset split…")
    X_train, X_test, y_train, y_test = load_dataset_split()
    print(f"  train={X_train.shape} test={X_test.shape}")

    pos_rate = float(y_train.mean())
    scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-9)

    family_dispatch = {
        "logreg": (_suggest_logreg, LogisticRegression, {}),
        "random_forest": (_suggest_rf, RandomForestClassifier, {}),
        "xgboost": (
            lambda trial: _suggest_xgb(trial, scale_pos_weight),
            XGBClassifier,
            {},
        ),
    }

    summaries = []
    for fam in args.families:
        suggest_fn, estimator_cls, extra = family_dispatch[fam]
        summary = _run_family(
            fam,
            suggest_fn,
            estimator_cls,
            X_train,
            X_test,
            y_train,
            y_test,
            n_trials=args.trials,
            seed=args.seed,
            extra_fit_params=extra,
        )
        summaries.append(summary)

    print("\n=== Summary ===")
    for s in summaries:
        print(
            f"{s['family']:<14} test_f1={s['test_metrics']['test_f1']:.4f} "
            f"test_roc_auc={s['test_metrics']['test_roc_auc']:.4f} "
            f"-> {Path(s['model_path']).name}"
        )

    best = max(summaries, key=lambda s: s["test_metrics"]["test_f1"])
    print(f"\nBest family by test F1: {best['family']} ({best['test_metrics']['test_f1']:.4f})")
    print("Inspect runs with:  mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
