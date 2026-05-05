# Process Overview — du repo vide au modèle entraîné

> **Sujet :** Prédire la conversion d'un visiteur en acheteur sur un site e-commerce.
> **Auteur :** Manech Carriou — Albert School
> **Repo :** https://github.com/manechcarriou-lab/ml-poc-project

Ce document explique **chaque étape** du projet et **pourquoi** elle a été faite ainsi. Il accompagne la présentation `process_overview.pptx`.

---

## TL;DR

| Élément | Valeur |
|---|---|
| **Modèle retenu** | XGBoost (Optuna-tuned) |
| **Métrique principale** | F1 sur la classe positive (achat) |
| **Test F1** | **0.6542** (cible >0.60 → +9 %) |
| **Test ROC-AUC** | **0.9303** |
| **Bonus threshold-tuning** | F1 = 0.6706 à seuil 0.28 (vs 0.5 par défaut) |
| **Reproductibilité** | 4 commandes (`git clone` → `train.py`) |

> Stack : Python 3.13, scikit-learn (Pipeline + ColumnTransformer), Optuna (TPE), MLflow, XGBoost, Streamlit.

---

## 1. Définition du problème

### 1.1 Question business

> *Peut-on prédire en temps réel, dès le début d'une session, qu'un visiteur a une forte probabilité d'acheter ?*

### 1.2 Pourquoi c'est utile

Les sites e-commerce convertissent en moyenne **1 à 3 %** de leurs visiteurs. Identifier les sessions à fort potentiel permet de :

- prioriser les budgets de retargeting / coupons / pop-ups,
- réduire le coût d'acquisition (CAC),
- améliorer l'UX en ne sollicitant pas les visiteurs déjà acquis.

### 1.3 Type de problème ML

**Classification binaire supervisée.** La cible `Revenue ∈ {True, False}` indique si la session a abouti à un achat. Pas de série temporelle, pas de multi-classes — c'est une porte d'entrée propre vers le ML tabulaire.

---

## 2. Choix du dataset — *Online Shoppers Purchasing Intention*

### 2.1 D'où il vient

- **Source :** UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- **Licence :** CC BY 4.0 (utilisation libre, attribution).
- **Volume :** 12 330 sessions × 18 colonnes (10 numériques, 7 catégorielles, 1 cible).

### 2.2 Pourquoi celui-là plutôt qu'un autre

| Critère | Vérifié |
|---|---|
| Public et stable | ✅ UCI |
| Taille manageable (entraînement local) | ✅ ~1 MB |
| Cible business claire | ✅ achat / pas achat |
| Features réalistes (pages vues, durée, page values, mois…) | ✅ |
| Réaliste : dataset déséquilibré (85/15) | ✅ — vrai défi métrique |

### 2.3 Limites assumées

- Pas d'identifiant utilisateur → pas de parcours multi-session.
- Pas d'année → pas de saisonnalité multi-annuelle.
- Modalités catégorielles anonymisées (Region 1-9, OS 1-8, Browser 1-13).

---

## 3. Setup technique

### 3.1 Stack

| Composant | Choix | Raison |
|---|---|---|
| Versionning | Git + GitHub | Standard, fork du template du cours |
| Auth GitHub | Clé SSH ed25519 | Pas de token à gérer, sécurisé |
| CLI GitHub | `gh` | Forker, push, ouvrir une PR depuis le terminal |
| Python | 3.13 dans un `.venv` local | Isolation des dépendances |
| Tracking ML | **MLflow 3.11** | Persistance des essais, UI web, log des artefacts |
| Hyperparam search | **Optuna 4.8** | Recherche bayésienne (TPE) plus efficace que Grid |

### 3.2 Reproductibilité

```bash
git clone git@github.com:manechcarriou-lab/ml-poc-project.git
cd ml-poc-project
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/train.py --trials 15
```

---

## 4. Structure du repo

```
ml-poc-project/
├── data/                 # CSV brut (gitignoré)
├── deliverables/         # documents projet
│   ├── assignment1.md
│   ├── process_overview.md
│   └── process_overview.pptx
├── models/               # joblib des modèles entraînés (gitignorés, rebuild via train.py)
├── mlruns/               # tracking MLflow (gitignoré)
├── notebooks/
│   ├── data_exploration.ipynb
│   └── feature_engineering.ipynb
├── results/              # CSV de comparaison de modèles
├── scripts/
│   ├── main.py           # entrée fournie par le template
│   └── train.py          # ★ script Optuna + MLflow
├── src/
│   ├── config.py         # chemins + registry MODELS
│   ├── data.py           # load_dataset_split (split 80/20 stratifié)
│   ├── features.py       # ★ pipeline preprocessing (no leakage)
│   ├── metrics.py        # compute_metrics
│   ├── model_io.py       # load/save joblib (template)
│   └── results.py        # template
└── requirements.txt
```

---

## 5. EDA — `notebooks/data_exploration.ipynb`

### 5.1 Checks de qualité

| Check | Résultat | Décision |
|---|---|---|
| NaN globaux | 0 | Pas d'imputation |
| Lignes avec NaN | 0 (cible <10 % OK) | RAS |
| Features avec >5 % NaN | aucune | Pas de drop |
| Outliers IQR (>1.5 IQR) | dominent sur `PageValues`, `BounceRates`, `*_Duration` | **log1p + scaling** plutôt que suppression brute |
| Drift (1ère vs 2ème moitié) | `SpecialDay` n'apparaît qu'au début, durations dérivent | À surveiller |
| Class imbalance cible | 84.5 % / **15.5 %** | `stratify=y` au split + `class_weight='balanced'` |
| Modalités rares | `VisitorType=Other` (0.7 %), `Month=Feb` (1.5 %) | OneHotEncoder avec `handle_unknown='ignore'` |

### 5.2 Insights par segment

![Conversion par segment](../plots/eda_conversion_by_segment.png)

- **Mois forts** : Nov, Sep, Oct (saison Black Friday).
- **New_Visitor** convertit **2× plus** que Returning_Visitor — contre-intuitif mais signal fort.
- Certains `TrafficType` (8, 11) ont des taux de conversion jusqu'à 3× la moyenne.
- Effet `Weekend` marginal.

![Class imbalance](../plots/eda_target_balance.png)

### 5.3 Outils utilisés

`pandas`, `seaborn`, `matplotlib`, `plotly` (treemap interactif Month × VisitorType).

---

## 6. Feature engineering — `src/features.py`

### 6.1 Pipeline = anti-leakage par construction

```python
preprocessor = ColumnTransformer([
    ("skewed_num", Pipeline([("log1p", FunctionTransformer(np.log1p)),
                              ("scale", StandardScaler())]),
     ["Administrative_Duration", "Informational_Duration",
      "ProductRelated_Duration", "PageValues"]),
    ("num", StandardScaler(), [...autres numériques...]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     ["Month", "VisitorType", "OperatingSystems", "Browser",
      "Region", "TrafficType", "Weekend"]),
    ("binary_passthrough", "passthrough",
     ["HighPageValue", "IsHighBounce", "IsSpecialDay"]),
])
```

**Pourquoi un ColumnTransformer dans un Pipeline ?**

- `fit` est appelé **uniquement** sur le train (dans la cross-validation, sur les folds train).
- `transform` est appelé sur le test **avec les statistiques apprises sur le train**.
- Aucune fuite test → train possible. C'est la garantie standard sklearn.

### 6.2 Features ajoutées (transformations row-wise stateless → leakage-safe)

| Nouvelle feature | Définition | Intuition |
|---|---|---|
| `TotalPages` | somme des 3 compteurs | volume global d'engagement |
| `TotalDuration` | somme des durations | temps passé total |
| `AvgTimePerPage` | duration / pages | profondeur de lecture |
| `ProductRelatedRatio` | pages produit / total | focus produit |
| `HighPageValue` | `PageValues > 0` | flag binaire de session marchande |
| `IsHighBounce` | `BounceRates > Q3` | flag de session zappée |
| `IsSpecialDay` | `SpecialDay > 0` | jour spécial (ex. Saint-Valentin) |

### 6.3 Encodage retenu

| Type | Technique | Raison |
|---|---|---|
| Numérique skewed | `log1p` puis `StandardScaler` | longue traîne, dirac sur 0 |
| Numérique standard | `StandardScaler` | LogReg / SVMs / PCA exigent des features standardisées |
| Catégoriel | `OneHotEncoder(handle_unknown='ignore')` | non ordinal, robuste aux nouvelles modalités |
| Binaire engineerée | passthrough | déjà 0/1, pas de transformation utile |

### 6.4 PCA — testé puis écarté

Dans `notebooks/feature_engineering.ipynb`, on a testé une PCA à 95 % de variance et à 20 composantes. Conclusion : la perte en F1 (-2 à -4 points) ne justifie pas la compression sur ce dataset (82 features finales seulement). **Pas retenu** dans la pipeline finale, mais documenté dans le notebook.

---

## 7. Modèles — `scripts/train.py`

### 7.1 Familles testées

| Famille | Pourquoi |
|---|---|
| Logistic Regression | Baseline interprétable, rapide, donne le plancher de F1 |
| Random Forest | Robuste, gère les features mixtes, peu de tuning |
| **XGBoost** | State-of-the-art sur du tabulaire, gradient boosting |

Toutes les 3 sont entraînées **dans la même pipeline** : `preprocessor → classifieur`. Le fit-on-train-only est garanti par sklearn.

### 7.2 Hyperparameter search — Optuna

**Pourquoi Optuna plutôt que GridSearchCV ?**

- Recherche **bayésienne (TPE)** : exploite les bons coins de l'espace au lieu de quadriller à l'aveugle.
- Échantillonnage continu (`learning_rate` log-uniforme) — pas de discrétisation arbitraire.
- Beaucoup plus rapide à converger : 15 essais Optuna ≈ 100 essais GridSearch.

**Espace de recherche par modèle**

| Famille | Hyperparamètres clés |
|---|---|
| LogReg | `C` (log 1e-3..1e2) |
| RF | `n_estimators` (100-400), `max_depth` (4-24), `min_samples_*`, `max_features` |
| XGBoost | `n_estimators` (150-600), `max_depth` (3-10), `learning_rate` (log 1e-2..3e-1), `subsample`, `colsample_bytree`, `gamma`, `reg_lambda`, `min_child_weight` |

**Objectif de l'étude :** maximiser le F1 (classe positive) en **CV stratifiée 3-fold** sur le train uniquement.

### 7.3 Tracking — MLflow

**Pourquoi MLflow ?**

- Trace **persistante** des essais — on peut comparer un run d'aujourd'hui avec un run d'il y a 2 semaines.
- Log automatique des **params + métriques + artefacts** (le joblib du modèle).
- UI web pour comparer visuellement les essais.

**Ce qui est loggé** :

- 1 run parent par étude (`{family}-study`)
- N runs nestés (`{family}-trial-{i}`) → params + cv_f1_mean
- 1 run final (`{family}-final`) → test_f1, test_precision, test_recall, test_roc_auc, modèle joblib

**Inspecter** :

```bash
mlflow ui --backend-store-uri ./mlruns
# http://localhost:5000
```

---

## 8. Résultats

### 8.1 Tableau de comparaison (test set 2 466 sessions)

| Modèle | CV F1 | Test F1 | Test Precision | Test Recall | Test ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.6737 | 0.5994 | 0.7206 | 0.5131 | 0.9137 |
| Random Forest | 0.6843 | 0.6292 | 0.7500 | 0.5419 | 0.9202 |
| **XGBoost** | **0.6858** | **0.6542** | 0.7238 | 0.5969 | **0.9303** |

### 8.2 Courbes ROC et Precision-Recall

![ROC curves](../plots/roc_curves.png)
![PR curves](../plots/pr_curves.png)

XGBoost domine sur les deux courbes. Sur la PR curve, la précision reste >0.70 jusqu'à un recall de ~0.55 — c'est ce qui rend le modèle utilisable en pratique.

### 8.3 Matrice de confusion + features importantes (XGBoost)

![Confusion matrix XGBoost](../plots/confusion_matrix_xgboost.png)
![Feature importance XGBoost](../plots/feature_importance_xgb.png)

`PageValues` domine très largement l'importance — c'est la métrique qui agrège la « valeur » des pages vues, et c'est l'effet le plus prédictif d'une intention d'achat.

### 8.4 Critère de succès

> *Atteindre F1 > 0.60 sur la classe positive avec le meilleur modèle.*

✅ **XGBoost atteint F1 = 0.6542**, soit +**9 %** au-dessus du seuil cible.

### 8.5 Threshold tuning — gain « gratuit »

![Threshold tuning XGBoost](../plots/threshold_tuning_xgb.png)

Le seuil par défaut (0.5) n'est jamais le seuil optimal sur un dataset déséquilibré.
**À threshold = 0.28**, F1 monte à **0.6706** (vs 0.6542 à 0.5) — soit +2.5 points sans réentraîner. Le seuil idéal dépend du trade-off métier (coût d'une action marketing vs valeur d'une conversion). Implémenté dans la démo Streamlit (`src/app.py` section 5).

### 8.6 Lecture business

- Sur 100 sessions identifiées comme « probable achat » par le modèle, **~72 sont vraiment des acheteurs** (precision 0.72).
- Sur 100 acheteurs réels, on **en attrape ~60** (recall 0.60). On rate 40 % des conversions, mais on ne « gaspille » pas de budget marketing sur des non-acheteurs.
- Le seuil de décision est ajustable : si on veut plus de recall (capter plus d'acheteurs), on baisse le seuil et on accepte plus de faux positifs.

---

## 9. Ce qui reste à faire

1. **`src/app.py`** — Streamlit avec :
   - Contexte business
   - 2-3 graphes EDA
   - Tableau `results/model_metrics.csv`
   - Démo interactive : sliders sur PageValues / BounceRates / VisitorType → probabilité prédite par XGBoost.
2. **`tests/`** — au moins un test sur le pipeline de preprocessing.
3. **Threshold tuning** — choisir un seuil métier au lieu de 0.5 (en fonction du coût FP vs FN).
4. **PR-AUC** dans les métriques — plus informatif que ROC-AUC sur dataset déséquilibré.

---

## 10. Synthèse — pourquoi chaque choix

| Choix | Pourquoi |
|---|---|
| Classification binaire | Cible `Revenue` est booléenne |
| F1 (positive class) comme métrique principale | 85/15 imbalanced + coût FP ≈ FN |
| Stratify=y au split | Préserver la balance dans train et test |
| `class_weight='balanced'` / `scale_pos_weight` | Compenser le déséquilibre dans la fonction de perte |
| `log1p` sur durations | Distributions long-tail |
| OneHotEncoder | Catégorielles non ordinales |
| ColumnTransformer + Pipeline | Empêcher toute fuite test → train |
| Optuna TPE | Recherche bayésienne efficace, espaces continus |
| 3-fold CV (pas 5 ou 10) | Compromis temps / variance, suffisant à 12 k lignes |
| MLflow local | Pas besoin de serveur, suffisant pour un PoC solo |
| 3 familles différentes | Comparer linéaire vs ensemble vs boosting |
