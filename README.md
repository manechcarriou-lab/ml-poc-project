# Online Shoppers Conversion Prediction

> A machine-learning proof of concept that predicts, from a single web session, whether
> a visitor will complete a purchase on an e-commerce site. Trained on the **UCI Online
> Shoppers Purchasing Intention** dataset (~12,330 sessions, 17 features, 15.5 %
> positive class). Best model: **XGBoost + Ordinal encoding + CV-tuned threshold 0.305**,
> reaching **F1 = 0.6731** and **ROC-AUC = 0.9292** on the held-out test set.
>
> *Author: Manech Carriou — Albert School — Machine Learning course (PoC).*

---

## What this project does

E-commerce sites convert 1 to 3 % of their visitors. The other 97 % leave without
buying. Growth and CRM teams have to decide, in real time, which sessions are worth
spending budget on (retargeting, pop-ups, coupons, live-chat triggers) and which
should be ignored.

This project frames that decision as a **binary classification problem**:

> *Given the features of a web session as it unfolds, can we predict whether it will
> result in a purchase?*

The deliverable is an interactive Streamlit dashboard structured in three parts:

1. **Problem & EDA** — the business context, the dataset, and the exploratory plots
   that motivated every preprocessing decision.
2. **Models & metrics** — three model families (Logistic Regression, Random Forest,
   XGBoost) trained inside an anti-leakage sklearn `Pipeline`, hyperparameter search
   with **Optuna**, experiment tracking with **MLflow**, automated **threshold tuning**
   with `TunedThresholdClassifierCV`, plus a *rigorous validation* section (calibration
   reliability diagram, error analysis by segment, comparison against non-ML baselines).
3. **Live demo** — a session-level simulator: adjust the inputs, see the probability
   of purchase predicted by the production XGBoost pipeline, and watch the ROI
   decomposition on the full test set update in real time as you move the decision
   threshold.

---

## Quick start

```bash
git clone git@github.com:manechcarriou-lab/ml-poc-project.git
cd ml-poc-project

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python scripts/main.py
```

`scripts/main.py` evaluates the saved models on the test split, writes
`results/model_metrics.csv`, then launches the Streamlit dashboard on
[http://localhost:8501](http://localhost:8501).

### Alternative entry points

| File | Purpose | Command |
|------|---------|---------|
| `src/app.py` | Main dashboard (ShopSignal design system) — flagship UI | `streamlit run src/app.py` |
| `src/presentation.py` | Multi-section presentation dashboard (sidebar nav, shadcn UI) | `streamlit run src/presentation.py` |
| `src/lifecycle_story.py` | Editorial scrollytelling — the project as a long-form story | `streamlit run src/lifecycle_story.py` |
| `scripts/train.py` | Retrain all three model families with Optuna + threshold tuning | `python scripts/train.py --trials 15` |
| `scripts/generate_plots.py` | Regenerate every PNG in `plots/` and `results/test_predictions.csv` | `python scripts/generate_plots.py` |
| `scripts/build_slides.py` | Rebuild `deliverables/process_overview.pptx` (16 slides) | `python scripts/build_slides.py` |
| `mlflow ui` | Inspect all training runs (params + metrics + artifacts) | `mlflow ui --backend-store-uri ./mlruns` |

### Tests

```bash
python -m unittest tests/test_pipeline.py -v
```

Seven smoke tests covering the split sizes, stratification, presence of the engineered
features, absence of target leakage, the leakage-safety of the preprocessor, the
metrics contract, and the round-trip prediction of every saved model.

---

## Repository structure

```
ml-poc-project/
├── data/                  # raw UCI dataset (gitignored, see "Getting the data")
├── deliverables/          # written deliverables for the course
│   ├── assignment1.md     # project proposal
│   ├── RAPPORT_COMPLET.md # full A-to-Z report
│   ├── process_overview.md  # condensed walkthrough
│   └── process_overview.pptx  # 16-slide deck
├── mlruns/                # MLflow tracking (gitignored)
├── models/                # trained joblib pipelines (gitignored, rebuild via train.py)
├── notebooks/
│   ├── data_exploration.ipynb     # the full EDA
│   ├── feature_engineering.ipynb  # PCA + feature comparison + RF importance
│   └── encoding_comparison.ipynb  # OneHot vs Ordinal vs TargetEncoder vs skrub
├── plots/                 # PNGs embedded in docs and slides (committed)
├── results/               # model_metrics.csv + test_predictions.csv (committed)
├── scripts/
│   ├── main.py            # evaluate + launch Streamlit (template entry point)
│   ├── train.py           # Optuna + MLflow + threshold tuning
│   ├── generate_plots.py  # regenerate every plot and the metrics CSVs
│   └── build_slides.py    # rebuild process_overview.pptx
├── src/
│   ├── app.py             # Streamlit dashboard (3 parts) — flagship
│   ├── presentation.py    # alternate presentation dashboard
│   ├── lifecycle_story.py # alternate scrollytelling
│   ├── config.py          # paths and MODELS registry
│   ├── data.py            # load_dataset_split (stratified 80/20)
│   ├── features.py        # ColumnTransformer pipeline (parametrable encoder)
│   ├── metrics.py         # compute_metrics — F1, precision, recall, ROC-AUC
│   ├── model_io.py        # joblib helpers (template)
│   ├── results.py         # write_metrics (template)
│   └── assets/logo/       # ShopSignal brand SVGs
├── tests/
│   └── test_pipeline.py   # 7 smoke tests
└── requirements.txt
```

---

## Methodology in one paragraph

The same anti-leakage `Pipeline` (sklearn `ColumnTransformer` + classifier) is used
for every model: numeric features go through `log1p` (when skewed) then
`StandardScaler`; categorical features use the encoder that empirically works best
for the model family (`OneHotEncoder` for the linear and tree-based models,
`OrdinalEncoder` for XGBoost, as validated in `notebooks/encoding_comparison.ipynb`).
Optuna runs 15 trials per family with 3-fold stratified cross-validation, optimising
F1 on the positive class. The fitted pipeline is then wrapped in
`TunedThresholdClassifierCV(scoring="f1", cv=5)`, so the decision threshold is
learned on the train split only (anti-leakage by construction). Every trial is logged
to MLflow; the best run per family is saved as `models/<family>.joblib`.

**Headline result:** XGBoost with Ordinal encoding and a threshold of 0.305 reaches
F1 = 0.6731, recall = 0.7356 (74 % of buyers caught), precision = 0.6203,
ROC-AUC = 0.9292 on the 2,466-session test set. That's +12 % above the PoC target
of F1 > 0.60.

---

## Getting the data

The raw CSV is not committed (only a `.gitkeep` placeholder lives in `data/`). To
materialise it locally, pick one of the two methods below.

### Method 1 — Direct download (recommended)

On Linux / macOS:

```bash
mkdir -p data
curl -L -o data/online_shoppers.zip \
  "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
cd data && unzip online_shoppers.zip && cd ..
ls -lh data/online_shoppers_intention.csv
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data | Out-Null
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip" `
                  -OutFile "data\online_shoppers.zip"
Expand-Archive -Path "data\online_shoppers.zip" -DestinationPath "data" -Force
```

You should now have `data/online_shoppers_intention.csv` (≈ 1 MB, 12,330 rows).

### Method 2 — Via the `ucimlrepo` Python package

```bash
pip install ucimlrepo
```

```python
from ucimlrepo import fetch_ucirepo
import pandas as pd

ds = fetch_ucirepo(id=468)
df = pd.concat([ds.data.features, ds.data.targets], axis=1)
df.to_csv("data/online_shoppers_intention.csv", index=False)
```

### Dataset specifications

| | |
|---|---|
| **Name** | Online Shoppers Purchasing Intention Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| **Licence** | CC BY 4.0 |
| **Rows** | 12,330 sessions |
| **Columns** | 18 (10 numeric + 7 categorical + 1 boolean target) |
| **Target** | `Revenue` (boolean) — did the session end with a purchase? |
| **Positive rate** | 15.5 % — heavy class imbalance |
| **File size** | ≈ 1 MB CSV |
| **Citation** | Sakar, C.O. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. |

### Once the data is in place

You have two paths:

1. Run `python scripts/train.py --trials 15` to retrain all three model families
   from scratch (≈ 5–8 minutes on a laptop CPU). This regenerates the joblibs in
   `models/`.
2. Or skip training and only use the cached predictions in
   `results/test_predictions.csv` to explore the dashboard — most of the UI works
   without the joblibs (the live-scoring section is the only one that requires them).

Either way, `python scripts/main.py` is the single command that brings everything
together: it evaluates the registered models on the test split, writes
`results/model_metrics.csv`, then launches Streamlit.

---

## Credits

Template repository by **Thomas Milcent** ([@thom1100](https://github.com/thom1100)),
Albert School Machine Learning course (2026). All ML work, deliverables, and the
ShopSignal design wrapper are by **Manech Carriou**.
