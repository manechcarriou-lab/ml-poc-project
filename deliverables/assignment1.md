# Assignment 1 — Proposition de projet ML

## Mon projet

**Titre :** Prédire la conversion d'un visiteur en acheteur sur un site e-commerce

**Type de problème :** Classification binaire supervisée (acheté / pas acheté).

## Le business case

Les sites e-commerce convertissent en moyenne 1 à 3 % de leurs visiteurs. Identifier en temps réel les visiteurs à fort potentiel d'achat permet :

- de prioriser les budgets de retargeting / pop-ups / coupons sur les sessions à forte intention,
- de réduire le coût d'acquisition (CAC) en concentrant l'effort marketing,
- d'améliorer l'UX en évitant de spammer les visiteurs déjà engagés.

**Stakeholder cible :** équipe Growth / CRM d'un retailer en ligne.

**Décision opérationnelle :** déclencher (ou non) une action marketing ciblée pendant la session.

## Les sources de données

**Dataset principal :** *Online Shoppers Purchasing Intention Dataset* (UCI Machine Learning Repository).

- ~12 330 sessions web,
- 17 features (10 numériques + 7 catégorielles) : nombre de pages produit visitées, durée de session, bounce rate, exit rate, page values, mois, type de visiteur, weekend, OS, navigateur, région, type de trafic, etc.
- Variable cible : `Revenue` (booléen, achat ou non).
- Source : <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset>
- Licence : CC BY 4.0.
- Classes déséquilibrées (~15 % de positifs) → traitement à prévoir (class_weight, SMOTE, threshold tuning).

## Contexte ML

**Famille d'algorithmes envisagés (au moins 3 modèles à comparer comme demandé par le template) :**

1. **Logistic Regression** — baseline interprétable, features standardisées.
2. **Random Forest** — robuste aux features mixtes, peu de tuning nécessaire.
3. **Gradient Boosting (XGBoost ou LightGBM)** — généralement state-of-the-art sur tabulaire.

**Pipeline prévu :**

- Split train/test 80/20 stratifié sur `Revenue`.
- Préprocessing : `OneHotEncoder` pour les catégorielles, `StandardScaler` pour les numériques (via `ColumnTransformer`).
- Gestion du déséquilibre via `class_weight='balanced'` + tuning du seuil de décision.
- Validation croisée 5-fold stratifiée pendant le tuning.

## Objectif d'évaluation

**Métrique principale :** **F1-score** sur la classe positive (achat).
*Justification :* dataset déséquilibré + coût similaire entre faux positifs (action marketing inutile, peu coûteuse) et faux négatifs (opportunité ratée). F1 reflète l'équilibre précision / rappel.

**Métriques secondaires reportées dans `results/model_metrics.csv` :**

- `accuracy` (sanity check),
- `precision` (combien d'actions marketing sont pertinentes),
- `recall` (combien d'acheteurs potentiels on capte),
- `roc_auc` (qualité du ranking, utile pour ajuster le seuil business).

**Critère de succès du PoC :** dépasser un baseline naïf (toujours prédire la classe majoritaire ≈ 84 % d'accuracy mais 0 de recall sur la classe positive) et viser **F1 > 0.60** sur la classe positive avec le meilleur modèle.

## Livrable Streamlit

L'app `src/app.py` présentera :

- le contexte business et la décision visée,
- la distribution de la cible et 2-3 graphes EDA (page values, type de visiteur, mois),
- le tableau de comparaison des 3 modèles depuis `results/model_metrics.csv`,
- une démo interactive : l'utilisateur règle quelques features clés (PageValues, BounceRate, VisitorType...) et l'app affiche la probabilité de conversion prédite par le meilleur modèle.
