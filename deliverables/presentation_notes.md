# Soutenance — notes et anticipations

> 4 minutes de présentation · 2 minutes de questions.
> *Manech Carriou · Albert School · ML PoC final.*

---

## Plan de la présentation orale (4 minutes)

| Temps | Slide / écran | À dire (en une phrase forte) |
|---|---|---|
| **00:00 — 00:30** | Cover ShopSignal | « Sur 100 visiteurs d'un site e-commerce, 98 partent sans rien acheter. Mon projet : prédire les 2 qui vont acheter, avant qu'ils ne quittent la session. » |
| **00:30 — 01:00** | Slide problème / dataset | Présenter le dataset UCI (12 330 sessions, 17 features, 15.5 % positifs) — insister sur le **déséquilibre** comme le défi structurant. |
| **01:00 — 01:30** | Slide pipeline anti-leakage | « Tout passe dans un `Pipeline` sklearn — un seul `fit`, sur le train. Validé par 7 tests unitaires. C'est la décision technique la plus importante du projet. » |
| **01:30 — 02:15** | Slide 3 modèles + Optuna | LogReg / RF / XGBoost. Optuna TPE 15 trials par famille, F1 sur classe positive comme objectif. **Encoder par famille** (OneHot pour les linéaires, Ordinal pour XGBoost) — choix validé empiriquement. |
| **02:15 — 02:45** | Slide threshold tuning | « Le seuil 0.5 par défaut n'est jamais optimal sur un dataset déséquilibré. `TunedThresholdClassifierCV` apprend le seuil sur le train, anti-leakage. Gain : +6 à +7 points de F1 sans réentraîner. » |
| **02:45 — 03:30** | Slide résultats finaux | **XGBoost · F1 = 0.6731 · ROC-AUC = 0.9292 · recall = 74 %.** Cible 0.60 dépassée de +12 %. Sur 2 466 sessions test, on attrape 281 acheteurs sur 382. |
| **03:30 — 04:00** | Démo Streamlit live | Bouger le slider de seuil, montrer la matrice de confusion qui se met à jour, conclure : « le seuil est une décision business, pas une décision ML — c'est exactement le levier que ce dashboard met entre les mains du Growth team ». |

**Texte de transition à mémoriser** (à dire pendant les slides) :

> *« Trois niveaux d'optimisation cumulés : Optuna pour les hyperparamètres, encoder par famille de modèle, et threshold tuning automatique en cross-validation. C'est ce qui fait passer F1 de 0.59 à 0.67. »*

---

## Anticipation des questions du jury (2 minutes)

### Q1 — « Pourquoi ce modèle ? Pourquoi XGBoost et pas une régression logistique ? »

**Réponse courte (15 sec)** :
XGBoost capture les interactions non-linéaires entre features (par exemple : la combinaison `PageValues élevé × visiteur Returning × mois Nov` n'a pas le même poids que la somme des trois pris séparément). La régression logistique fait l'hypothèse d'additivité — elle est ma baseline, mais sur des features comportementales web qui se combinent fortement, XGBoost capte ~2-3 points de F1 supplémentaires.

**Réponse longue (si on insiste)** :
- **LogReg F1 = 0.6426** vs **XGBoost F1 = 0.6731** — l'écart vient des interactions.
- XGBoost gère mieux le déséquilibre via `scale_pos_weight`.
- Mais j'ai gardé LogReg comme baseline pour ne pas tomber dans le piège du « toujours XGBoost » — sur certains splits (cf. notebook encoding_comparison), LogReg gagne sur OneHot.
- Si on voulait du déployable interprétable, on prendrait LogReg avec une légère perte de performance. C'est un compromis.

---

### Q2 — « Pourquoi Random Forest est plus intéressant qu'une régression logistique ici ? »

**Réponse** :
Random Forest gagne sur trois plans précis :

1. **Robustesse aux features mixtes** : LogReg force à standardiser proprement et à encoder One-Hot toutes les catégorielles (82 features finales) ; RF accepte les Ordinal directement, donc moins de bruit.
2. **Pas d'hypothèse d'additivité** : un arbre peut apprendre « *si PageValues > 50 ET VisitorType=New, alors achat probable* » — règle conjonctive impossible pour LogReg sans feature engineering manuel.
3. **Meilleure gestion des outliers** : un arbre split sur un seuil, donc une valeur extrême n'influence pas le coefficient comme dans LogReg.

Sur ce dataset, **RF gagne +2.5 points de F1 sur LogReg** (0.6426 → 0.6683). C'est mesuré, pas théorique.

---

### Q3 — « Pourquoi avoir choisi `PageValues > 0` comme baseline naïf ? »

**Réponse** :
Parce que c'est la feature la plus corrélée à la cible (~0.49 de corrélation linéaire) — c'était l'heuristique métier la plus défendable contre laquelle confronter le ML. Et le résultat est honnête : cette règle simple atteint **F1 = 0.6485**, soit seulement **3.5 % en-dessous de XGBoost**. Ça nuance le gain ML : on n'apporte pas une révolution, on apporte un raffinement mesurable. Dans le notebook `encoding_comparison`, j'ai aussi testé une règle saisonnière (PageValues + mois Nov/Sep/Oct) qui donne F1 = 0.4545 — moins bonne car trop restrictive.

---

### Q4 — « Pourquoi avoir enlevé ou rajouté telle feature ? »

**Réponse sur les features ajoutées** :
J'ai ajouté **7 features engineerées** dans `src/features.py` :
- `TotalPages`, `TotalDuration`, `AvgTimePerPage`, `ProductRelatedRatio` (ratios/agrégats)
- `HighPageValue`, `IsHighBounce`, `IsSpecialDay` (binarisations)

Elles capturent des **interactions non-linéaires** que LogReg ne peut pas inférer seul. Toutes sont **row-wise stateless** (pas de `fit` à apprendre) — leakage-safe par construction.

**Réponse sur les features supprimées** :
Aucune feature de base n'a été supprimée. J'ai testé une PCA (notebook `feature_engineering.ipynb`) qui aurait compressé en 20 composantes, mais la perte F1 (-2 à -4 points) ne justifiait pas la compression sur 82 features finales — donc rejetée.

---

### Q5 — « Pourquoi le F1 sur la classe positive et pas l'accuracy ? »

**Réponse en 10 secondes** :
85/15 de déséquilibre. Un modèle qui prédit **toujours `False`** atteint **84.5 % d'accuracy** et **0 de recall sur la classe d'intérêt** — c'est inutilisable mais ça paraît bon. Le F1 (moyenne harmonique precision × recall) pénalise les modèles qui négligent la classe rare. ROC-AUC vient en deuxième pour évaluer le ranking indépendamment du seuil.

---

### Q6 — « Comment êtes-vous sûr qu'il n'y a pas de data leakage ? »

**Réponse** :
Trois garde-fous :

1. **`Pipeline` sklearn** : le `fit` du `StandardScaler` et de l'`OneHotEncoder` n'a accès qu'à `X_train`. Le `transform` du test utilise les statistiques apprises sur le train.
2. **TargetEncoder** : quand testé, j'ai utilisé sa CV interne (`cv=5`) pour éviter qu'une modalité du train voit sa propre moyenne.
3. **TunedThresholdClassifierCV** : le seuil optimal est cherché en 5-fold CV sur le train uniquement. Le test set n'est jamais vu pendant le tuning.

Et tout ça est **vérifié par un test unitaire** dans `tests/test_pipeline.py` : `test_preprocessor_is_leakage_safe`.

---

### Q7 — « Pourquoi pas un réseau de neurones ? »

**Réponse** :
Trois raisons :
1. **Taille du dataset** : 12 330 lignes, c'est trop petit pour qu'un MLP batte un gradient boosting sur du tabulaire — toutes les benchmarks Kaggle / Papers With Code sur des datasets de cette taille montrent que XGBoost/CatBoost/LightGBM gagnent ou égalent les NNs.
2. **Interprétabilité** : SHAP marche très bien sur XGBoost (TreeExplainer, exact, rapide). Pour un PoC business, c'est crucial : le Growth team doit comprendre pourquoi une session est ciblée.
3. **Coût opérationnel** : un MLP demanderait GPU, plus de maintenance, calibration différente. Pour un PoC, le marginal benefit ne le justifie pas.

---

### Q8 — « Que ferait votre modèle sur un autre site e-commerce ? »

**Réponse honnête** :
Pas grand-chose sans réentraînement. Le modèle est calibré sur les distributions du site qui a fourni le dataset UCI (Tunisie, retailer non précisé, période 2018). Les feature distributions vont drifter sur un autre site : autres `TrafficType` codes, autres mois forts, autre `PageValues`. **C'est pour ça qu'on a documenté la calibration et l'analyse d'erreur par segment** : ça permet de détecter rapidement si le modèle dérive sur un nouveau contexte. Le code et le pipeline sont génériques, le modèle ne l'est pas.

---

### Q9 — « Le threshold 0.305 — comment vous le justifiez en prod ? »

**Réponse** :
C'est une décision **technique** (maximise le F1) et **business** (maximise le compromis precision/recall).

- Côté technique : `TunedThresholdClassifierCV` a fait une 5-fold CV sur le train et a trouvé 0.305 comme optimum du F1. Honnête, anti-leakage.
- Côté business : à 0.305, on cible 30 % du trafic, on capte 74 % des acheteurs avec une précision de 62 %. Si le coût d'une action marketing est très faible (un pop-up gratuit), on peut **descendre encore** le seuil pour gagner du recall. Si le coût est élevé (un coupon de 20€), on monte le seuil pour la précision. Le slider dans la démo Streamlit existe précisément pour ça.

---

### Q10 — « Avez-vous comparé à des baselines plus sérieux ? »

**Réponse** :
Oui, trois baselines non-ML sont implémentés dans la section *Validation rigoureuse* de l'app :
- Toujours prédire « non-acheteur » → F1 = 0.0000 (accuracy 84.5 % mais inutile)
- Règle simple `PageValues > 0` → F1 = 0.6485
- Règle métier `PageValues > 0 ET mois Nov/Sep/Oct` → F1 = 0.4545 (trop restrictive)

XGBoost à F1 = 0.6731 bat la meilleure heuristique de **+3.5 %** — gain modeste mais consistant, et plus robuste hors saison.

---

## Ce qu'il faut absolument savoir par cœur

| Chiffre | Valeur | Comment c'est calculé |
|---|---|---|
| Volume dataset | 12 330 sessions | UCI Online Shoppers |
| Taux conversion | 15.5 % | `Revenue.mean()` |
| F1 XGBoost | 0.6731 | sur test set 2 466 sessions, threshold 0.305 |
| ROC-AUC XGBoost | 0.9292 | indépendant du seuil |
| Recall XGBoost | 0.7356 | 281/382 acheteurs captés |
| Precision XGBoost | 0.6203 | 281/(281+172) sessions ciblées vraies |
| Brier score | 0.0708 | bonne calibration (max = 0.25) |
| Optuna trials | 45 | 15 par famille × 3 familles |
| MLflow runs | 52 | studies + nested trials + finals |
| Threshold optimal | 0.305 | TunedThresholdClassifierCV, CV-tuned |
| Gain vs cible PoC | +12 % | (0.6731 - 0.60) / 0.60 |

---

## Pièges à éviter pendant la présentation

1. **Ne pas dire « accuracy »** comme si c'était la métrique principale. Si on vous demande la perf : F1, recall, ROC-AUC dans cet ordre. Accuracy en dernier (et seulement si on insiste).
2. **Ne pas confondre threshold tuning et tuning d'hyperparamètres.** Le seuil est appris après l'entraînement, pas pendant.
3. **Ne pas prétendre que XGBoost a battu LogReg de loin.** L'écart est de ~3 points de F1 — réel mais pas spectaculaire. Le vrai gain vient du threshold tuning (+6-7 points), pas du choix du modèle.
4. **Ne pas oublier de mentionner les limites** : pas d'identifiant utilisateur, pas de saisonnalité multi-annuelle, modalités catégorielles anonymisées. Les limites montrent qu'on a compris ce qu'on a fait.
5. **Ne pas se précipiter sur la démo.** Garder 30 secondes pour bouger le slider de seuil — c'est le moment le plus parlant pour un non-ML.

---

## Script d'ouverture (les 30 premières secondes)

> Bonjour. Je vais vous présenter un projet de classification supervisée appliqué à l'e-commerce : prédire, à partir d'une session web, si le visiteur va acheter ou non.
>
> Le contexte business est simple : sur 100 visiteurs qui arrivent sur un site marchand, 98 partent sans rien acheter. Les équipes Growth doivent décider — en temps réel — sur quelles sessions concentrer leur budget marketing : retargeting, pop-ups, coupons. Mon projet leur fournit le score qui rend cette décision automatique.
>
> Le dataset utilisé est l'**UCI Online Shoppers Purchasing Intention** : 12 330 sessions, 17 features, et un défi structurant : **seulement 15.5 % des sessions sont des achats**. Ce déséquilibre 85/15 oriente toutes mes décisions techniques en aval.

## Script de clôture (les 20 dernières secondes)

> En résumé : un pipeline sklearn anti-leakage validé par tests unitaires, trois familles de modèles benchmarkées avec Optuna et tracées dans MLflow, threshold tuning automatique en cross-validation. XGBoost gagne avec F1 = 0.6731 — 12 % au-dessus de la cible PoC.
>
> Et surtout : un dashboard qui transforme cette métrique en décision business. Le seuil de décision est un curseur, pas une constante. C'est ça, à mon sens, qui rend un modèle ML utilisable au-delà du PoC.

---

## Checklist 5 minutes avant de passer

- [ ] App Streamlit lancée et **ouverte dans un onglet** : `python scripts/main.py`
- [ ] Naviguer une fois sur les 3 onglets pour pré-charger les caches
- [ ] Bouger le slider de seuil une fois pour vérifier que la confusion matrix se met à jour
- [ ] Onglet GitHub ouvert : `github.com/manechcarriou-lab/ml-poc-project`
- [ ] Slides ouvertes : `deliverables/process_overview.pptx`
- [ ] Rapport complet ouvert si besoin de référence : `deliverables/RAPPORT_COMPLET.md`
- [ ] Téléphone éteint, fenêtres inutiles fermées
- [ ] Boire un verre d'eau

Bonne soutenance.
