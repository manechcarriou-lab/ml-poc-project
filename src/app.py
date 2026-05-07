"""Streamlit entry point — Online Shoppers Conversion Prediction.

Three-part dashboard (To-Do 5):

1. Le problème & EDA — contexte business, dataset, EDA plots clés.
2. Modèles & métriques — pourquoi le F1, pourquoi 3 familles, comparaison.
3. Démo monde réel — comment utiliser le modèle en production avec une
   prédiction live et une simulation de seuil.

Launch with:
    streamlit run src/app.py
    OR
    python scripts/main.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DATA_DIR, MODEL_METRICS_FILE, MODELS, PLOTS_DIR, RESULTS_DIR
from features import add_engineered_features

DATASET_PATH = DATA_DIR / "online_shoppers_intention.csv"
TEST_PRED_PATH = RESULTS_DIR / "test_predictions.csv"

PRIMARY = "#6366F1"
ACCENT = "#EC4899"
SUCCESS = "#10B981"
GREY = "#64748B"
DARK = "#0F172A"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data
def _load_dataset() -> pd.DataFrame | None:
    if not DATASET_PATH.exists():
        return None
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def _load_metrics() -> pd.DataFrame | None:
    if not MODEL_METRICS_FILE.exists():
        return None
    return pd.read_csv(MODEL_METRICS_FILE)


@st.cache_data
def _load_test_predictions() -> pd.DataFrame | None:
    if not TEST_PRED_PATH.exists():
        return None
    return pd.read_csv(TEST_PRED_PATH)


@st.cache_resource
def _load_xgb_pipeline():
    path = MODELS["xgboost"]["path"]
    if not Path(path).exists():
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Plotly theming
# ---------------------------------------------------------------------------


def _style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=DARK),
        margin=dict(t=50, b=30, l=30, r=30),
        legend=dict(bgcolor="rgba(255,255,255,0.95)"),
    )
    fig.update_xaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0", zeroline=False)
    fig.update_yaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# PART 1 — Le problème & EDA
# ---------------------------------------------------------------------------


def part1_problem_and_eda(df: pd.DataFrame | None) -> None:
    st.header("Partie 1 — Le problème & l'analyse exploratoire")

    st.markdown(
        """
        ### Contexte business

        Les sites e-commerce convertissent en moyenne **1 à 3 %** de leurs visiteurs.
        L'équipe Growth / CRM doit identifier en temps réel les sessions à fort
        potentiel d'achat pour **prioriser les budgets marketing** (retargeting,
        coupons, pop-ups personnalisés).

        > **Question ML :** *Peut-on prédire dès le début d'une session qu'un visiteur
        > va acheter ?*

        **Type de problème :** classification binaire supervisée
        — la cible `Revenue ∈ {True, False}` indique si la session a abouti à un achat.

        **Application business**
        - Déclencher (ou non) une action marketing pendant la session.
        - Réduire le coût d'acquisition (CAC) en concentrant l'effort sur les sessions à fort potentiel.
        - Améliorer l'UX en ne sollicitant pas les visiteurs déjà engagés.
        """
    )

    cols = st.columns(4)
    cols[0].metric("Sessions analysées", "12 330")
    cols[1].metric("Features", "17")
    cols[2].metric("Class imbalance", "85 / 15", help="84.5 % False / 15.5 % True")
    cols[3].metric("Source", "UCI ML Repo")

    st.markdown("---")
    st.markdown("### Le dataset — *Online Shoppers Purchasing Intention*")
    st.markdown(
        """
        - **Source :** UCI Machine Learning Repository, licence CC BY 4.0.
        - **17 features** : 10 numériques (page counts, durations, bounce/exit rates, PageValues, SpecialDay) et 7 catégorielles (Month, VisitorType, OperatingSystems, Browser, Region, TrafficType, Weekend).
        - **Cible** : `Revenue` (booléen).
        - **Limites** : pas d'identifiant utilisateur, pas d'année (donc pas de saisonnalité multi-annuelle), modalités catégorielles anonymisées.
        """
    )

    if df is None:
        st.warning("Dataset non trouvé. Lance `scripts/generate_plots.py` après le téléchargement.")
        return

    st.markdown("---")
    st.markdown("### Les graphiques EDA importants")

    eda_tab1, eda_tab2, eda_tab3 = st.tabs(
        ["1. Class imbalance", "2. Conversion par segment", "3. Top corrélations"]
    )

    with eda_tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            counts = df["Revenue"].value_counts()
            pct = (counts / counts.sum() * 100).round(1)
            fig = go.Figure(
                go.Bar(
                    x=["No purchase (84.5 %)", "Purchase (15.5 %)"],
                    y=counts.values,
                    text=[f"{v:,} sessions<br>{p} %" for v, p in zip(counts.values, pct.values)],
                    textposition="outside",
                    marker=dict(color=[PRIMARY, ACCENT], line=dict(color="white", width=2)),
                )
            )
            fig.update_layout(
                title="Distribution de la cible — fort déséquilibre",
                yaxis_title="Sessions", showlegend=False, height=400,
            )
            st.plotly_chart(_style(fig), use_container_width=True)
        with col2:
            st.info(
                "**Pourquoi c'est important**\n\n"
                "Avec 85 % de la majorité, un modèle qui prédit toujours `False` aurait "
                "**85 % d'accuracy mais 0 de recall** sur la classe d'intérêt.\n\n"
                "→ On choisit le **F1 sur la classe positive** comme métrique principale, "
                "pas l'accuracy."
            )

    with eda_tab2:
        seg_col = st.selectbox(
            "Segmenter par",
            ["Month", "VisitorType", "TrafficType", "Weekend"],
        )
        order = None
        if seg_col == "Month":
            order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        agg = (
            df.groupby(seg_col)["Revenue"]
              .agg(["mean", "count"])
              .rename(columns={"mean": "conversion_rate", "count": "n_sessions"})
              .reset_index()
        )
        if order:
            agg[seg_col] = pd.Categorical(agg[seg_col], categories=order, ordered=True)
            agg = agg.sort_values(seg_col)
        else:
            agg = agg.sort_values("conversion_rate", ascending=False)
        fig = px.bar(
            agg, x=seg_col, y="conversion_rate",
            text=agg["conversion_rate"].map(lambda x: f"{x:.1%}"),
            color="conversion_rate", color_continuous_scale="Viridis",
            hover_data=["n_sessions"],
        )
        fig.add_hline(
            y=df["Revenue"].mean(),
            line_dash="dash", line_color="red",
            annotation_text=f"Moyenne {df['Revenue'].mean():.1%}",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(title=f"Taux de conversion par {seg_col}", height=420,
                          coloraxis_showscale=False)
        st.plotly_chart(_style(fig), use_container_width=True)
        st.info(
            "**Insights clés** : Nov/Sep/Oct sont les mois forts (Black Friday). "
            "**New_Visitor convertit 2× plus** que Returning_Visitor — contre-intuitif "
            "mais signal très discriminant pour le modèle."
        )

    with eda_tab3:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        corrs = df[num_cols].corrwith(df["Revenue"].astype(int)).sort_values(key=abs, ascending=False)
        cdf = corrs.drop("Revenue", errors="ignore").reset_index()
        cdf.columns = ["feature", "corr"]
        fig = px.bar(
            cdf.head(10).iloc[::-1],
            x="corr", y="feature", orientation="h",
            color="corr", color_continuous_scale="RdBu", range_color=[-0.5, 0.5],
        )
        fig.update_layout(title="Top 10 corrélations linéaires avec Revenue",
                          height=440, coloraxis_showscale=False)
        st.plotly_chart(_style(fig), use_container_width=True)
        st.info(
            "**Lecture** : `PageValues` domine très largement (corr ≈ 0.49). C'est la feature "
            "qui agrège la valeur des pages vues — directement liée à l'intention d'achat."
        )


# ---------------------------------------------------------------------------
# PART 2 — Modèles & métriques
# ---------------------------------------------------------------------------


def part2_models_and_metrics(metrics_df: pd.DataFrame | None) -> None:
    st.header("Partie 2 — Choix de modèles & métriques")

    st.markdown(
        """
        ### Choix de la métrique : F1 sur la classe positive

        Le dataset est **déséquilibré 85/15**. L'accuracy est trompeuse — il faut une
        métrique qui pénalise les modèles qui négligent la classe rare.

        | Métrique | Pourquoi (ou pourquoi pas) |
        |---|---|
        | ❌ Accuracy | 85 % facile en prédisant tout `False` |
        | ✅ **F1 (positive class)** | Moyenne harmonique precision × recall, robuste à l'imbalance |
        | ✅ ROC-AUC | Qualité du *ranking*, indépendante du seuil |
        | ✅ Precision / Recall | Lecture business directe |

        ### Choix des modèles : 3 familles complémentaires
        """
    )

    cols = st.columns(3)
    with cols[0]:
        st.markdown("#### 📐 Logistic Regression")
        st.caption("Baseline interprétable, rapide, donne le plancher de F1.")
    with cols[1]:
        st.markdown("#### 🌳 Random Forest")
        st.caption("Robuste, gère features mixtes, peu de tuning.")
    with cols[2]:
        st.markdown("#### 🚀 XGBoost")
        st.caption("State-of-the-art tabulaire, gradient boosting.")

    st.markdown(
        """
        Toutes les 3 sont entraînées **dans le même pipeline** preprocessor → classifieur,
        avec `class_weight='balanced'` (LogReg/RF) ou `scale_pos_weight` (XGBoost) pour
        compenser le déséquilibre. Hyperparamètres optimisés par **Optuna** (15 trials par modèle,
        3-fold CV stratifiée). Seuil de décision tuné en CV via `TunedThresholdClassifierCV` —
        anti-leakage par construction.
        """
    )

    st.markdown("---")
    st.markdown("### Comparaison des modèles — résultats sur le test set (2 466 sessions)")

    if metrics_df is None:
        st.warning(
            "`results/model_metrics.csv` introuvable. Lance "
            "`python scripts/generate_plots.py`."
        )
        return

    pretty = metrics_df.copy()
    for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if c in pretty.columns:
            pretty[c] = pretty[c].map(lambda x: f"{x:.4f}")
    pretty["encoder"] = ["OneHot", "OneHot", "Ordinal"]
    pretty["threshold"] = ["0.244", "0.360", "0.305"]
    cols_order = ["model_name", "encoder", "threshold", "f1", "precision", "recall", "roc_auc", "accuracy"]
    cols_order = [c for c in cols_order if c in pretty.columns]
    st.dataframe(pretty[cols_order], use_container_width=True, hide_index=True)

    metric_choice = st.selectbox(
        "Métrique à visualiser",
        ["f1", "roc_auc", "precision", "recall", "accuracy"],
        index=0,
    )
    fig = px.bar(
        metrics_df.sort_values(metric_choice, ascending=True),
        x=metric_choice, y="model_name", orientation="h",
        text=metrics_df.sort_values(metric_choice, ascending=True)[metric_choice].map(lambda x: f"{x:.3f}"),
        color=metric_choice, color_continuous_scale="Viridis",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title=f"Comparaison des modèles — {metric_choice.upper()}",
        height=320, coloraxis_showscale=False,
    )
    st.plotly_chart(_style(fig), use_container_width=True)

    st.markdown("### Plots de comparaison (sauvegardés dans `plots/`)")
    plot_tabs = st.tabs([
        "ROC curves", "Precision-Recall", "Confusion matrix XGBoost",
        "Feature importance XGBoost",
    ])
    plot_paths = [
        ("ROC", PLOTS_DIR / "roc_curves.png", "XGBoost domine avec AUC=0.93."),
        ("PR", PLOTS_DIR / "pr_curves.png", "Précision >0.7 jusqu'à recall ~0.55."),
        ("CM", PLOTS_DIR / "confusion_matrix_xgboost.png",
         "281 acheteurs détectés, 101 ratés (FN), 172 fausses alertes (FP)."),
        ("FI", PLOTS_DIR / "feature_importance_xgb.png",
         "PageValues domine — agrège la valeur des pages vues."),
    ]
    for tab, (_, path, caption) in zip(plot_tabs, plot_paths):
        with tab:
            if path.exists():
                st.image(str(path), use_container_width=True)
                st.caption(caption)
            else:
                st.info(f"`{path.name}` non trouvé. Lance `scripts/generate_plots.py`.")

    st.success(
        "**Conclusion :** XGBoost gagne avec **F1 = 0.6731** et **ROC-AUC = 0.9292**. "
        "Recall = 0.7356 → on attrape **74 % des acheteurs réels**. "
        "Cible (F1 > 0.60) dépassée de +12 %."
    )


# ---------------------------------------------------------------------------
# PART 3 — Démo monde réel
# ---------------------------------------------------------------------------


def part3_real_world_demo(pipeline, test_pred: pd.DataFrame | None) -> None:
    st.header("Partie 3 — Démo : utiliser le modèle dans le monde réel")

    st.markdown(
        """
        ### Cas d'usage opérationnel

        > Imaginez : tu es **data scientist chez un retailer e-commerce**. Le modèle
        > XGBoost tourne en production, scoré à chaque navigation. À chaque session
        > en cours, ton outil interne reçoit la **probabilité d'achat estimée**.

        Le système Growth utilise ce score pour décider :
        - **Score élevé** (`p >= seuil`) → action marketing premium (coupon, popup, support live).
        - **Score moyen** → action légère (newsletter, retargeting standard).
        - **Score faible** → ne rien faire, économiser le budget.

        Le slider ci-dessous **simule l'arrivée d'une session** sur le site. Règle les
        paramètres comme si tu observais un visiteur en temps réel, et lis la décision
        produite par le modèle.
        """
    )

    if pipeline is None:
        st.warning(
            "Modèle XGBoost introuvable. Lance `python scripts/train.py` pour créer "
            "`models/xgboost.joblib`."
        )
        return

    st.markdown("---")
    st.markdown("### 🛒 Simulateur de session")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Comportement de navigation**")
        product_related = st.slider("Pages produit visitées", 0, 200, 30)
        product_duration = st.slider("Durée pages produit (s)", 0, 6000, 1000, step=50)
        admin_pages = st.slider("Pages admin (login/cart…)", 0, 30, 2)
    with col2:
        st.markdown("**Métriques d'engagement**")
        page_values = st.slider("PageValues (proxy intention)", 0.0, 200.0, 5.0, step=1.0)
        bounce_rate = st.slider("Bounce rate", 0.0, 0.2, 0.02, step=0.005, format="%.3f")
        exit_rate = st.slider("Exit rate", 0.0, 0.2, 0.04, step=0.005, format="%.3f")
    with col3:
        st.markdown("**Contexte**")
        visitor_type = st.selectbox(
            "Type de visiteur",
            ["Returning_Visitor", "New_Visitor", "Other"],
        )
        month = st.selectbox(
            "Mois",
            ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            index=8,
        )
        weekend = st.checkbox("Session weekend", value=False)

    threshold = st.slider(
        "Seuil de décision (0.305 = optimum tuné par CV)",
        0.05, 0.95, 0.305, step=0.01,
    )

    # Build the row using add_engineered_features — same code path as training.
    base = pd.DataFrame([{
        "Administrative": admin_pages,
        "Administrative_Duration": admin_pages * 30.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": product_related,
        "ProductRelated_Duration": float(product_duration),
        "BounceRates": bounce_rate,
        "ExitRates": exit_rate,
        "PageValues": float(page_values),
        "SpecialDay": 0.0,
        "Month": month,
        "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2,
        "VisitorType": visitor_type, "Weekend": weekend,
    }])
    row = add_engineered_features(base)

    try:
        proba = float(pipeline.predict_proba(row)[0, 1])
    except Exception as e:
        st.error(f"Erreur de prédiction : {e}")
        return

    decision = "🟢 CIBLER — déclencher action marketing" if proba >= threshold else "⚪ NE PAS CIBLER — économiser le budget"

    st.markdown("---")
    st.markdown("### Décision en temps réel")

    cols = st.columns([2, 1])
    with cols[0]:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": " %", "font": {"size": 48, "color": PRIMARY}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": GREY},
                "bar": {"color": PRIMARY, "thickness": 0.7},
                "bordercolor": "#E2E8F0",
                "borderwidth": 1,
                "threshold": {
                    "line": {"color": ACCENT, "width": 4},
                    "value": threshold * 100,
                },
                "steps": [
                    {"range": [0, 30], "color": "#F1F5F9"},
                    {"range": [30, 60], "color": "#E0E7FF"},
                    {"range": [60, 100], "color": "#C7D2FE"},
                ],
            },
            title={"text": "<b>Probabilité d'achat estimée par XGBoost</b>"},
        ))
        fig.update_layout(height=320, margin=dict(t=50, b=20, l=20, r=20),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with cols[1]:
        st.metric("Probabilité", f"{proba:.1%}")
        st.metric("Seuil business", f"{threshold:.3f}")
        st.metric("Confiance", f"{abs(proba - 0.5) * 2:.0%}", help="Ecart au 50/50")
        st.markdown(f"### {decision}")

    st.markdown("---")
    st.markdown("### Simuler le ROI sur le test set")

    if test_pred is not None and "proba_xgboost" in test_pred.columns:
        st.caption(
            "On applique ton seuil au test set complet (2 466 sessions) pour voir "
            "combien de sessions seraient ciblées et combien d'acheteurs réels seraient "
            "captés."
        )
        y_true = test_pred["y_true"].astype(int).values
        y_score = test_pred["proba_xgboost"].values
        y_pred = (y_score >= threshold).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        total_real_buyers = tp + fn
        total_targeted = tp + fp

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sessions ciblées", f"{total_targeted}/{len(y_true)}",
                  f"{total_targeted/len(y_true):.0%} du trafic")
        c2.metric("Acheteurs captés (TP)", f"{tp}/{total_real_buyers}",
                  f"recall {tp/total_real_buyers:.0%}")
        c3.metric("Fausses alertes (FP)", f"{fp}",
                  f"precision {tp/max(tp+fp,1):.0%}")
        c4.metric("Acheteurs ratés (FN)", f"{fn}",
                  f"{fn/total_real_buyers:.0%} ratés")

        fig = go.Figure(go.Bar(
            x=["Acheteurs captés", "Fausses alertes", "Acheteurs ratés", "Bien ignorés"],
            y=[tp, fp, fn, tn],
            marker_color=[SUCCESS, ACCENT, "#F59E0B", "#94A3B8"],
            text=[tp, fp, fn, tn], textposition="outside",
        ))
        fig.update_layout(
            title=f"Décomposition de la décision au seuil {threshold:.2f}",
            height=320, showlegend=False, yaxis_title="Sessions",
        )
        st.plotly_chart(_style(fig), use_container_width=True)

        st.info(
            "**Lecture business :**  \n"
            f"- Si on cible toutes ces sessions, on touche **{total_targeted} visiteurs** "
            f"(soit {total_targeted/len(y_true):.0%} du trafic).  \n"
            f"- Parmi eux, **{tp}** sont vraiment des acheteurs (gain).  \n"
            f"- **{fp}** ne le sont pas (coût de l'action perdue).  \n"
            f"- On rate **{fn}** acheteurs qu'on aurait pu cibler.  \n\n"
            "**Comment ajuster :** baisse le seuil si une conversion vaut beaucoup "
            "plus qu'une action marketing perdue (recall ↑, precision ↓)."
        )
    else:
        st.info("`results/test_predictions.csv` introuvable — la simulation ROI nécessite ce fichier.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_app() -> None:
    st.set_page_config(
        page_title="Online Shoppers — Conversion Prediction",
        layout="wide",
        page_icon="🛒",
    )

    with st.sidebar:
        st.title("🛒 ML PoC")
        st.markdown(
            """
            **Auteur :** Manech Carriou
            **École :** Albert School
            **Repo :** [github.com/manechcarriou-lab/ml-poc-project](https://github.com/manechcarriou-lab/ml-poc-project)
            """
        )
        st.divider()
        st.markdown(
            """
            **Pipeline :**
            EDA → preprocessing (no leakage) → 3 modèles → Optuna + MLflow → threshold tuning → XGBoost.

            **Pour relancer le training :**
            ```
            python scripts/train.py
            ```
            """
        )

    st.title("Prédire la conversion d'un visiteur e-commerce")
    st.caption("Online Shoppers Purchasing Intention — UCI Machine Learning Repository")

    df = _load_dataset()
    metrics_df = _load_metrics()
    test_pred = _load_test_predictions()
    pipeline = _load_xgb_pipeline()

    tab1, tab2, tab3 = st.tabs([
        "1️⃣  Le problème & EDA",
        "2️⃣  Modèles & métriques",
        "3️⃣  Démo monde réel",
    ])

    with tab1:
        part1_problem_and_eda(df)
    with tab2:
        part2_models_and_metrics(metrics_df)
    with tab3:
        part3_real_world_demo(pipeline, test_pred)


if __name__ == "__main__":
    build_app()
