from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for dir in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# Trained model registry. Each entry points to a full sklearn Pipeline
# (preprocessor + classifier) serialized as `.joblib` by `scripts/train.py`.
# Rebuild with:  python scripts/train.py --trials 15
MODELS = {
    "logreg": {
        "name": "Logistic Regression",
        "description": "Baseline linear classifier — Optuna-tuned C, class_weight='balanced'.",
        "path": MODELS_DIR / "logreg.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Tree ensemble — Optuna-tuned depth/leaves/features, class_weight='balanced'.",
        "path": MODELS_DIR / "random_forest.joblib",
    },
    "xgboost": {
        "name": "XGBoost",
        "description": "Gradient boosting — Optuna-tuned depth/lr/subsample, scale_pos_weight set from train balance.",
        "path": MODELS_DIR / "xgboost.joblib",
    },
}
