"""Streamlit entry point — Online Shoppers Conversion Prediction.

Sections
--------
1. Business context
2. Dataset & EDA highlights
3. Model comparison (from results/model_metrics.csv)
4. Interactive prediction demo (XGBoost)
5. Threshold tuning playground

Launch with:  python scripts/main.py   (or)   streamlit run src/app.py
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

DATASET_PATH = DATA_DIR / "online_shoppers_intention.csv"
TEST_PRED_PATH = RESULTS_DIR / "test_predictions.csv"


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
# Sections
# ---------------------------------------------------------------------------


def _section_business() -> None:
    st.header("1. Le problème business")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
            **Question :** Peut-on prédire dès le début d'une session qu'un visiteur va acheter ?

            **Pourquoi c'est utile**

            - Les sites e-commerce convertissent en moyenne **1 à 3 %** de leurs visiteurs.
            - Identifier les sessions à fort potentiel permet de **prioriser les budgets de retargeting**, coupons et pop-ups.
            - On évite de spammer les visiteurs déjà engagés et on réduit le coût d'acquisition (CAC).

            **Décision opérationnelle :** déclencher (ou non) une action marketing pendant la session.

            **Type de problème ML :** classification binaire supervisée (`Revenue ∈ {True, False}`).
            """
        )
    with col2:
        st.metric("Sessions analysées", "12 330")
        st.metric("Taux de conversion global", "15.5 %")
        st.metric("Modèle retenu", "XGBoost")
        st.metric("F1 (test)", "0.654", "+9 % vs cible 0.60")


def _section_dataset_eda(df: pd.DataFrame | None) -> None:
    st.header("2. Le dataset & l'EDA")

    st.markdown(
        """
        **Source :** UCI Machine Learning Repository — *Online Shoppers Purchasing Intention Dataset*
        (CC BY 4.0). 12 330 sessions × 18 colonnes.
        """
    )

    if df is None:
        st.warning("Dataset non trouvé. Lance `scripts/generate_plots.py` après le téléchargement.")
        return

    tab_balance, tab_segment, tab_corr = st.tabs(
        ["Class imbalance", "Conversion par segment", "Top corrélations"]
    )

    with tab_balance:
        counts = df["Revenue"].value_counts().rename({False: "No purchase", True: "Purchase"})
        fig = px.bar(
            x=counts.index, y=counts.values,
            labels={"x": "Outcome", "y": "Sessions"},
            color=counts.index, color_discrete_sequence=["#4c72b0", "#dd8452"],
            text=[f"{v / counts.sum():.1%}" for v in counts.values],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, title="Distribution de la cible — fort déséquilibre 85/15")
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "Conséquence : l'**accuracy** seule serait trompeuse (~85 % en prédisant toujours `No purchase`). "
            "On privilégie le **F1 sur la classe positive** + `class_weight='balanced'`."
        )

    with tab_segment:
        col_select = st.selectbox(
            "Segmenter par",
            ["Month", "VisitorType", "Weekend", "TrafficType", "Region"],
            index=0,
        )
        order = None
        if col_select == "Month":
            order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        agg = (
            df.groupby(col_select)["Revenue"]
              .agg(["mean", "count"])
              .rename(columns={"mean": "conversion_rate", "count": "n_sessions"})
              .reset_index()
        )
        if order:
            agg[col_select] = pd.Categorical(agg[col_select], categories=order, ordered=True)
            agg = agg.sort_values(col_select)
        else:
            agg = agg.sort_values("conversion_rate", ascending=False)

        fig = px.bar(
            agg, x=col_select, y="conversion_rate",
            text=agg["conversion_rate"].map(lambda x: f"{x:.1%}"),
            hover_data=["n_sessions"],
            color="conversion_rate", color_continuous_scale="Viridis",
        )
        fig.add_hline(y=df["Revenue"].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Moyenne {df['Revenue'].mean():.1%}",
                      annotation_position="bottom right")
        fig.update_traces(textposition="outside")
        fig.update_layout(title=f"Taux de conversion par {col_select}")
        st.plotly_chart(fig, use_container_width=True)

    with tab_corr:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        corrs = df[num_cols].corrwith(df["Revenue"].astype(int)).sort_values(key=lambda s: s.abs(), ascending=False)
        corr_df = corrs.drop("Revenue", errors="ignore").reset_index()
        corr_df.columns = ["feature", "corr_with_revenue"]
        fig = px.bar(corr_df.head(10), x="corr_with_revenue", y="feature", orientation="h",
                     color="corr_with_revenue", color_continuous_scale="RdBu", range_color=[-0.5, 0.5])
        fig.update_layout(title="Top 10 corrélations linéaires avec la cible")
        st.plotly_chart(fig, use_container_width=True)


def _section_models(metrics_df: pd.DataFrame | None) -> None:
    st.header("3. Comparaison des modèles")

    if metrics_df is None:
        st.warning(
            "`results/model_metrics.csv` introuvable. Lance "
            "`python scripts/generate_plots.py` ou `python scripts/main.py`."
        )
        return

    pretty = metrics_df.copy()
    for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if c in pretty.columns:
            pretty[c] = pretty[c].map(lambda x: f"{x:.4f}")
    st.dataframe(pretty, use_container_width=True, hide_index=True)

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
    fig.update_layout(title=f"Comparaison sur {metric_choice}")
    st.plotly_chart(fig, use_container_width=True)

    plot_paths = {
        "ROC curves": PLOTS_DIR / "roc_curves.png",
        "Precision-Recall curves": PLOTS_DIR / "pr_curves.png",
        "Confusion matrix — XGBoost": PLOTS_DIR / "confusion_matrix_xgboost.png",
        "Feature importance — XGBoost": PLOTS_DIR / "feature_importance_xgb.png",
    }
    available = {label: path for label, path in plot_paths.items() if path.exists()}
    if available:
        with st.expander("Plots additionnels (générés par `scripts/generate_plots.py`)", expanded=False):
            cols = st.columns(2)
            for i, (label, path) in enumerate(available.items()):
                with cols[i % 2]:
                    st.image(str(path), caption=label, use_container_width=True)


def _section_demo(pipeline) -> None:
    st.header("4. Démo interactive — XGBoost")

    if pipeline is None:
        st.warning(
            "Modèle XGBoost introuvable. Lance `python scripts/train.py` pour créer "
            "`models/xgboost.joblib`."
        )
        return

    st.markdown(
        "Règle quelques features clés et le modèle te donne la **probabilité d'achat** "
        "estimée pour cette session synthétique."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        product_related = st.slider("Pages produit visitées", 0, 200, 30, step=1)
        product_duration = st.slider("Durée pages produit (s)", 0, 6000, 1000, step=50)
        page_values = st.slider("PageValues (proxy d'intention d'achat)", 0.0, 200.0, 5.0, step=1.0)
    with col2:
        bounce_rate = st.slider("Bounce rate", 0.0, 0.2, 0.02, step=0.005, format="%.3f")
        exit_rate = st.slider("Exit rate", 0.0, 0.2, 0.04, step=0.005, format="%.3f")
        admin_pages = st.slider("Pages admin (login/cart…)", 0, 30, 2, step=1)
    with col3:
        visitor_type = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
        month = st.selectbox(
            "Mois",
            ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            index=8,
        )
        weekend = st.checkbox("Weekend", value=False)

    threshold = st.slider(
        "Seuil de décision (par défaut 0.5 — descendre pour capter plus d'acheteurs)",
        0.05, 0.95, 0.50, step=0.01,
    )

    row = pd.DataFrame([{
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
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
        "VisitorType": visitor_type,
        "Weekend": weekend,
        # Engineered features (also added inside data.add_engineered_features)
        "TotalPages": admin_pages + product_related,
        "TotalDuration": admin_pages * 30.0 + product_duration,
        "AvgTimePerPage": (admin_pages * 30.0 + product_duration) / max(admin_pages + product_related, 1),
        "ProductRelatedRatio": product_related / max(admin_pages + product_related, 1),
        "HighPageValue": int(page_values > 0),
        "IsHighBounce": int(bounce_rate > 0.1),
        "IsSpecialDay": 0,
    }])

    proba = float(pipeline.predict_proba(row)[0, 1])
    decision = "🟢 Cibler — action marketing recommandée" if proba >= threshold else "⚪ Ne pas cibler"

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Probabilité d'achat", f"{proba:.1%}")
    col_b.metric("Seuil", f"{threshold:.2f}")
    col_c.metric("Décision", decision)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": " %"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1f77b4"},
            "threshold": {
                "line": {"color": "red", "width": 4},
                "value": threshold * 100,
            },
            "steps": [
                {"range": [0, 30], "color": "#f5f5f5"},
                {"range": [30, 60], "color": "#e8eef5"},
                {"range": [60, 100], "color": "#cfdce7"},
            ],
        },
        title={"text": "Score XGBoost"},
    ))
    fig.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)


def _section_threshold_playground(test_pred: pd.DataFrame | None) -> None:
    st.header("5. Threshold playground (XGBoost sur le test set)")
    if test_pred is None or "proba_xgboost" not in test_pred.columns:
        st.info(
            "`results/test_predictions.csv` non trouvé. Lance "
            "`python scripts/generate_plots.py` pour générer les prédictions du test."
        )
        return

    threshold = st.slider("Seuil de décision", 0.05, 0.95, 0.50, step=0.01, key="thresh_pg")
    y_true = test_pred["y_true"].astype(int).values
    y_score = test_pred["proba_xgboost"].values
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{precision:.3f}")
    c2.metric("Recall", f"{recall:.3f}")
    c3.metric("F1", f"{f1:.3f}")
    c4.metric("Sessions ciblées", f"{tp + fp} / {len(y_true)}")

    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Real: No buy", "Real: Buy"],
        columns=["Pred: No buy", "Pred: Buy"],
    )
    st.dataframe(cm_df, use_container_width=False)

    st.caption(
        "Plus le seuil baisse, plus on cible large : recall augmente, precision diminue. "
        "Le seuil optimal dépend du coût relatif d'une action marketing vs la valeur d'une conversion."
    )


# ---------------------------------------------------------------------------
# Entry point
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
        st.caption("Pipeline : EDA → preprocessing (no leakage) → Optuna + MLflow → XGBoost.")

    st.title("Prédire la conversion d'un visiteur e-commerce")
    st.caption("Online Shoppers Purchasing Intention — UCI ML Repository")

    df = _load_dataset()
    metrics_df = _load_metrics()
    test_pred = _load_test_predictions()
    pipeline = _load_xgb_pipeline()

    _section_business()
    st.divider()
    _section_dataset_eda(df)
    st.divider()
    _section_models(metrics_df)
    st.divider()
    _section_demo(pipeline)
    st.divider()
    _section_threshold_playground(test_pred)


if __name__ == "__main__":
    build_app()
