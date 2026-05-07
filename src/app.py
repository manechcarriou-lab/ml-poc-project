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
import streamlit_shadcn_ui as ui
from streamlit_extras.add_vertical_space import add_vertical_space

from config import DATA_DIR, MODEL_METRICS_FILE, MODELS, PLOTS_DIR, RESULTS_DIR
from features import add_engineered_features

DATASET_PATH = DATA_DIR / "online_shoppers_intention.csv"
TEST_PRED_PATH = RESULTS_DIR / "test_predictions.csv"

PRIMARY = "#6366F1"        # indigo-500
PRIMARY_DARK = "#4338CA"   # indigo-700
PRIMARY_LIGHT = "#A5B4FC"  # indigo-300
ACCENT = "#EC4899"         # pink-500
ACCENT_DARK = "#BE185D"    # pink-700
SUCCESS = "#10B981"        # emerald-500
WARNING = "#F59E0B"        # amber-500
DANGER = "#EF4444"         # red-500
GREY = "#64748B"           # slate-500
LIGHT = "#F1F5F9"          # slate-100
LIGHTER = "#F8FAFC"        # slate-50
DARK = "#0F172A"           # slate-900
WHITE = "#FFFFFF"


# ---------------------------------------------------------------------------
# Custom CSS — same identity as src/presentation.py
# ---------------------------------------------------------------------------


def _inject_css() -> None:
    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        .stApp {{
            background:
                radial-gradient(circle at 0% 0%, rgba(99, 102, 241, 0.08), transparent 40%),
                radial-gradient(circle at 100% 0%, rgba(236, 72, 153, 0.06), transparent 40%),
                {LIGHTER};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        .stApp, .stApp * {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        .section-eyebrow {{
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.22rem;
            background: linear-gradient(90deg, {PRIMARY} 0%, {ACCENT} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }}
        .section-title {{
            font-size: 2rem;
            font-weight: 800;
            color: {DARK};
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }}
        .section-lead {{
            font-size: 1.05rem;
            color: {GREY};
            line-height: 1.6;
            margin-bottom: 1.6rem;
            max-width: 820px;
        }}
        .insight-card {{
            background: {WHITE};
            border: 1px solid #E2E8F0;
            border-left: 4px solid {ACCENT};
            padding: 1rem 1.2rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 4px 12px rgba(15, 23, 42, 0.04);
            margin: 0.6rem 0;
        }}
        .insight-card.success {{ border-left-color: {SUCCESS}; }}
        .insight-card.warning {{ border-left-color: {WARNING}; }}
        .insight-card.danger {{ border-left-color: {DANGER}; }}
        .insight-card.info {{ border-left-color: {PRIMARY}; }}
        .insight-title {{
            font-size: 0.7rem;
            font-weight: 700;
            color: {ACCENT};
            letter-spacing: 0.16rem;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }}
        .insight-card.success .insight-title {{ color: {SUCCESS}; }}
        .insight-card.warning .insight-title {{ color: {WARNING}; }}
        .insight-card.danger .insight-title {{ color: {DANGER}; }}
        .insight-card.info .insight-title {{ color: {PRIMARY}; }}
        .insight-body {{
            font-size: 0.95rem;
            color: {DARK};
            line-height: 1.55;
        }}
        .insight-body strong {{ color: {PRIMARY_DARK}; }}
        .conclusion-card {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.06) 0%, rgba(99, 102, 241, 0.06) 100%);
            border: 1px solid rgba(16, 185, 129, 0.25);
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            margin: 1.4rem 0;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);
        }}
        .conclusion-card .label {{
            font-size: 0.7rem;
            font-weight: 800;
            color: {SUCCESS};
            letter-spacing: 0.18rem;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }}
        .conclusion-card .body {{
            font-size: 1.05rem;
            color: {DARK};
            line-height: 1.55;
        }}
        .hero-card {{
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
            border-radius: 18px;
            padding: 2rem 2.2rem;
            color: white;
            margin-bottom: 1.6rem;
            box-shadow: 0 12px 40px rgba(99, 102, 241, 0.25);
        }}
        .hero-card .kicker {{
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.28rem;
            opacity: 0.85;
            text-transform: uppercase;
        }}
        .hero-card .title {{
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -0.02em;
            margin-top: 0.6rem;
        }}
        .hero-card .subtitle {{
            font-size: 1rem;
            opacity: 0.92;
            margin-top: 0.7rem;
            max-width: 760px;
        }}
        h1, h2, h3, h4 {{
            color: {DARK};
            letter-spacing: -0.015em;
        }}
        h2 {{
            font-weight: 800 !important;
        }}
        code {{
            background: rgba(99, 102, 241, 0.08) !important;
            color: {PRIMARY_DARK} !important;
            padding: 0.15rem 0.4rem !important;
            border-radius: 4px !important;
            font-size: 0.88em !important;
        }}
        pre code {{
            background: {DARK} !important;
            color: #E0E7FF !important;
        }}
        hr {{
            border-color: rgba(99, 102, 241, 0.15) !important;
            margin: 1.5rem 0 !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
            border-bottom-color: rgba(99, 102, 241, 0.15);
        }}
        .stTabs [data-baseweb="tab-list"] button {{
            font-weight: 600;
            color: {GREY};
            padding: 0.6rem 1rem;
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            color: {PRIMARY} !important;
            border-bottom-color: {PRIMARY} !important;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {DARK} 0%, #1E1B4B 100%);
        }}
        section[data-testid="stSidebar"] * {{
            color: {WHITE} !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a {{
            color: {PRIMARY_LIGHT} !important;
        }}
        footer, header {{visibility: hidden;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable styled components
# ---------------------------------------------------------------------------


_KPI_COUNTER = {"i": 0}
_BADGE_COUNTER = {"i": 0}


def _section_header(eyebrow: str, title: str, lead: str = "") -> None:
    lead_html = f'<div class="section-lead">{lead}</div>' if lead else ""
    html = (
        f'<div class="section-eyebrow">{eyebrow}</div>'
        f'<div class="section-title">{title}</div>'
        f'{lead_html}'
    )
    st.markdown(html, unsafe_allow_html=True)
    add_vertical_space(1)


def _kpi(label: str, value: str, delta: str = "") -> None:
    """Polished metric card — shadcn UI."""
    _KPI_COUNTER["i"] += 1
    ui.metric_card(
        title=label,
        content=value,
        description=delta,
        key=f"kpi_{_KPI_COUNTER['i']}",
    )


def _insight(title: str, body_html: str, kind: str = "default") -> None:
    """Card with accent border. kind: default | info | success | warning | danger."""
    cls = "insight-card" + ("" if kind == "default" else f" {kind}")
    html = (
        f'<div class="{cls}">'
        f'<div class="insight-title">{title}</div>'
        f'<div class="insight-body">{body_html}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _conclusion(label: str, body_html: str) -> None:
    """Highlighted conclusion block with subtle gradient."""
    html = (
        f'<div class="conclusion-card">'
        f'<div class="label">{label}</div>'
        f'<div class="body">{body_html}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _hero(kicker: str, title: str, subtitle: str = "") -> None:
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    html = (
        f'<div class="hero-card">'
        f'<div class="kicker">{kicker}</div>'
        f'<div class="title">{title}</div>'
        f'{sub}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


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
    _section_header(
        "PARTIE 01 · LE PROBLÈME & EDA",
        "Comprendre la donnée avant de modéliser",
        "Contexte business, dataset, et les graphiques EDA qui orientent les décisions techniques.",
    )

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
    with cols[0]:
        _kpi("Sessions analysées", "12 330")
    with cols[1]:
        _kpi("Features", "17")
    with cols[2]:
        _kpi("Class imbalance", "85 / 15", "84.5 % False · 15.5 % True")
    with cols[3]:
        _kpi("Source", "UCI ML Repo", "Licence CC BY 4.0")

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
            _insight(
                "Pourquoi c'est important",
                "Avec 85 % de la majorité, un modèle qui prédit toujours <code>False</code> aurait "
                "<strong>85 % d'accuracy mais 0 de recall</strong> sur la classe d'intérêt. "
                "On choisit donc le <strong>F1 sur la classe positive</strong> comme métrique principale, pas l'accuracy.",
                kind="info",
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
        _insight(
            "Insights clés",
            "Nov/Sep/Oct sont les mois forts (saison Black Friday). "
            "<strong>New_Visitor convertit 2× plus</strong> que Returning_Visitor — "
            "contre-intuitif mais signal très discriminant pour le modèle.",
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
        _insight(
            "Lecture",
            "<code>PageValues</code> domine très largement (corr ≈ 0.49). "
            "C'est la feature qui agrège la valeur des pages vues — directement liée à l'intention d'achat.",
        )


# ---------------------------------------------------------------------------
# PART 2 — Modèles & métriques
# ---------------------------------------------------------------------------


def _validation_calibration(test_pred: pd.DataFrame) -> None:
    """Reliability diagram — checks if predicted probabilities are well-calibrated."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    y_true = test_pred["y_true"].astype(int).values
    y_score = test_pred["proba_xgboost"].values

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true, y_score)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color=GREY, width=2),
        name="Calibration parfaite",
    ))
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true, mode="lines+markers",
        line=dict(color=PRIMARY, width=3),
        marker=dict(size=11, color=ACCENT, line=dict(color="white", width=2)),
        name="XGBoost (10 bins quantile)",
    ))
    fig.update_layout(
        title="Reliability diagram — XGBoost sur le test set",
        xaxis_title="Probabilité prédite (moyenne par bin)",
        yaxis_title="Fréquence d'achat réelle (par bin)",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(_style(fig), use_container_width=True)

    verdict_text = (
        "légèrement sous-confiant dans les hauts scores"
        if (prob_true - prob_pred).max() > 0.05
        else "globalement bien calibré"
    )
    _insight(
        "Comment lire",
        "Chaque point représente un bin de probabilité prédite. "
        "Si la courbe colle à la diagonale, le modèle est bien calibré — quand il dit "
        "« 70 % de chances d'acheter », ça arrive vraiment 70 % du temps.",
        kind="info",
    )
    cols = st.columns(2)
    with cols[0]:
        _kpi("Brier score", f"{brier:.4f}", "plus bas = mieux (max 0.25)")
    with cols[1]:
        _kpi("Calibration", verdict_text.split()[0].capitalize(), verdict_text)
    _insight(
        "Verdict",
        f"XGBoost est <strong>{verdict_text}</strong>. "
        f"C'est important parce que ça valide que le seuil 0.305 du threshold tuning correspond à une "
        f"vraie probabilité de 30.5 %, pas à un nombre arbitraire.",
        kind="success",
    )


def _validation_error_analysis(test_pred: pd.DataFrame) -> None:
    """Where does the model systematically fail?"""
    from sklearn.metrics import f1_score, precision_score, recall_score

    df = test_pred.copy()
    threshold = 0.305
    df["y_pred"] = (df["proba_xgboost"] >= threshold).astype(int)

    seg = st.selectbox(
        "Segmenter par",
        ["VisitorType", "Month", "Weekend", "TrafficType"],
        help="Choisis une dimension pour voir comment les performances du modèle varient selon les modalités.",
    )

    rows = []
    for modality, group in df.groupby(seg):
        n = len(group)
        if n < 20:
            continue  # skip too-tiny groups for stable metrics
        y_t = group["y_true"].astype(int).values
        y_p = group["y_pred"].values
        n_pos = int(y_t.sum())
        if n_pos == 0:
            f1 = prec = rec = 0.0
        else:
            f1 = f1_score(y_t, y_p, zero_division=0)
            prec = precision_score(y_t, y_p, zero_division=0)
            rec = recall_score(y_t, y_p, zero_division=0)
        rows.append({
            "modalité": str(modality),
            "n_sessions": n,
            "n_acheteurs": n_pos,
            "F1": f1,
            "precision": prec,
            "recall": rec,
        })

    seg_df = pd.DataFrame(rows).sort_values("n_sessions", ascending=False)
    overall_f1 = f1_score(
        df["y_true"].astype(int), df["y_pred"], zero_division=0
    )

    fig = go.Figure()
    for metric, color, name in [
        ("precision", PRIMARY, "Precision"),
        ("recall", ACCENT, "Recall"),
        ("F1", SUCCESS, "F1"),
    ]:
        fig.add_trace(go.Bar(
            x=seg_df["modalité"], y=seg_df[metric],
            name=name, marker=dict(color=color),
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(
        y=overall_f1, line_dash="dot", line_color="black",
        annotation_text=f"F1 global {overall_f1:.3f}",
        annotation_position="top right",
        annotation_font=dict(color="black", size=11),
    )
    fig.update_layout(
        title=f"Performance du modèle par modalité de {seg} (modalités <20 sessions exclues)",
        barmode="group",
        height=440,
        yaxis_title="Métrique",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(_style(fig), use_container_width=True)

    # Highlight problematic segments — biggest gap to overall F1
    seg_df["gap_to_overall"] = seg_df["F1"] - overall_f1
    worst = seg_df.nsmallest(3, "gap_to_overall")
    best = seg_df.nlargest(3, "F1")

    col_w, col_b = st.columns(2)
    with col_w:
        bullets = "".join(
            f"<li><strong>{r['modalité']}</strong> "
            f"(n={r['n_sessions']}, {r['n_acheteurs']} acheteurs) → "
            f"F1 = {r['F1']:.3f} <em>({r['gap_to_overall']:+.3f} vs global)</em></li>"
            for _, r in worst.iterrows()
        )
        _insight(
            "🔴 Modalités où le modèle fait moins bien",
            f"<ul style='margin: 0; padding-left: 1.2rem;'>{bullets}</ul>",
            kind="danger",
        )
    with col_b:
        bullets = "".join(
            f"<li><strong>{r['modalité']}</strong> "
            f"(n={r['n_sessions']}, {r['n_acheteurs']} acheteurs) → F1 = {r['F1']:.3f}</li>"
            for _, r in best.iterrows()
        )
        _insight(
            "🟢 Modalités où le modèle excelle",
            f"<ul style='margin: 0; padding-left: 1.2rem;'>{bullets}</ul>",
            kind="success",
        )

    with st.expander("Tableau détaillé par modalité"):
        pretty = seg_df.copy()
        for c in ["F1", "precision", "recall"]:
            pretty[c] = pretty[c].map(lambda x: f"{x:.4f}")
        pretty["gap_to_overall"] = pretty["gap_to_overall"].map(lambda x: f"{x:+.4f}")
        st.dataframe(pretty, use_container_width=True, hide_index=True)


def _validation_naive_baseline(test_pred: pd.DataFrame) -> None:
    """Compare the model to dumb rules to put the ML gain in perspective."""
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    y_true = test_pred["y_true"].astype(int).values

    # Rule 1: predict everything as non-buyer (the "easy" baseline that fools accuracy)
    rule_always_no = np.zeros_like(y_true)

    # Rule 2: PageValues > 0 (the most-correlated single feature)
    rule_page_values = (test_pred["PageValues"] > 0).astype(int).values

    # Rule 3: PageValues > 0 AND New_Visitor or Returning in Nov/Sep/Oct (heuristique métier)
    high_season = test_pred["Month"].isin(["Nov", "Sep", "Oct"]).values
    rule_business = ((test_pred["PageValues"] > 0).values & high_season).astype(int)

    # XGBoost at tuned threshold
    rule_xgb = (test_pred["proba_xgboost"] >= 0.305).astype(int).values

    rules = [
        ("Toujours « non-acheteur » (baseline dégénéré)", rule_always_no),
        ("Règle simple : PageValues > 0", rule_page_values),
        ("Règle métier : PageValues > 0 ET mois Nov/Sep/Oct", rule_business),
        ("XGBoost (tuned, threshold = 0.305)", rule_xgb),
    ]

    rows = []
    for name, preds in rules:
        rows.append({
            "Stratégie": name,
            "Accuracy": accuracy_score(y_true, preds),
            "Precision": precision_score(y_true, preds, zero_division=0),
            "Recall": recall_score(y_true, preds, zero_division=0),
            "F1": f1_score(y_true, preds, zero_division=0),
            "Sessions ciblées": int(preds.sum()),
        })
    bench = pd.DataFrame(rows)

    pretty = bench.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1"]:
        pretty[c] = pretty[c].map(lambda x: f"{x:.4f}")
    pretty["Sessions ciblées"] = pretty["Sessions ciblées"].map(lambda x: f"{x:,}")
    st.dataframe(pretty, use_container_width=True, hide_index=True)

    # Bar chart on F1
    fig = px.bar(
        bench.sort_values("F1"), x="F1", y="Stratégie", orientation="h",
        text=bench.sort_values("F1")["F1"].map(lambda x: f"{x:.4f}"),
        color="F1", color_continuous_scale="Viridis",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title="F1 par stratégie de décision",
        height=300, coloraxis_showscale=False,
    )
    st.plotly_chart(_style(fig), use_container_width=True)

    naive_f1 = bench.iloc[1]["F1"]
    business_f1 = bench.iloc[2]["F1"]
    xgb_f1 = bench.iloc[3]["F1"]
    gain_vs_simple = (xgb_f1 - naive_f1) / max(naive_f1, 1e-9) * 100
    gain_vs_business = (xgb_f1 - business_f1) / max(business_f1, 1e-9) * 100

    _insight(
        "Lecture honnête du gain ML",
        f"<ul style='margin: 0 0 0.6rem 1.2rem; padding: 0;'>"
        f"<li>Vs <em>toujours « non-acheteur »</em> : F1 passe de <strong>0.0000 à {xgb_f1:.4f}</strong>. "
        f"L'accuracy est trompeuse (le baseline a déjà 84.5 %), mais le F1 montre qu'il est inutilisable.</li>"
        f"<li>Vs <em>règle simple <code>PageValues &gt; 0</code></em> : ML apporte "
        f"<strong>+{gain_vs_simple:.0f} %</strong> de F1 ({naive_f1:.4f} → {xgb_f1:.4f}). "
        f"Cette règle est étonnamment forte parce que <code>PageValues</code> est la feature la plus corrélée.</li>"
        f"<li>Vs <em>règle métier (PageValues + saison Nov/Sep/Oct)</em> : ML apporte "
        f"<strong>+{gain_vs_business:.0f} %</strong> ({business_f1:.4f} → {xgb_f1:.4f}).</li>"
        f"</ul>"
        f"Conclusion : le ML apporte un vrai gain mesurable même contre des heuristiques métier non triviales — "
        f"ce qui justifie le coût d'entraînement et de maintenance d'un modèle plutôt qu'une simple règle.",
        kind="info",
    )


def part2_models_and_metrics(metrics_df: pd.DataFrame | None,
                              test_pred: pd.DataFrame | None = None) -> None:
    _section_header(
        "PARTIE 02 · MODÈLES & MÉTRIQUES",
        "Choisir, comparer, valider rigoureusement",
        "Pourquoi le F1, pourquoi 3 familles complémentaires, et comment on valide qu'on n'a pas triché.",
    )

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
                _insight(
                    "Plot manquant",
                    f"<code>{path.name}</code> non trouvé. Lance <code>python scripts/generate_plots.py</code>.",
                    kind="warning",
                )

    _conclusion(
        "VERDICT",
        "<strong>XGBoost gagne avec F1 = 0.6731</strong> et ROC-AUC = 0.9292. "
        "Recall = 0.7356 → on attrape <strong>74 % des acheteurs réels</strong>. "
        "Cible initiale (F1 > 0.60) dépassée de <strong>+12 %</strong>.",
    )

    # ------------------------------------------------------------------
    # Rigorous validation — calibration, error analysis, naive baseline
    # ------------------------------------------------------------------
    if test_pred is None or "proba_xgboost" not in test_pred.columns:
        return

    st.markdown("---")
    st.markdown("### 🔬 Validation rigoureuse")
    st.caption(
        "Au-delà des métriques agrégées : calibration des probabilités, "
        "analyse d'erreur par segment, et comparaison à des baselines non-ML."
    )

    rig_tabs = st.tabs([
        "📐 Calibration",
        "🎯 Erreurs par segment",
        "⚖️ vs baselines non-ML",
    ])
    with rig_tabs[0]:
        _validation_calibration(test_pred)
    with rig_tabs[1]:
        _validation_error_analysis(test_pred)
    with rig_tabs[2]:
        _validation_naive_baseline(test_pred)


# ---------------------------------------------------------------------------
# PART 3 — Démo monde réel
# ---------------------------------------------------------------------------


def part3_real_world_demo(pipeline, test_pred: pd.DataFrame | None) -> None:
    _section_header(
        "PARTIE 03 · DÉMO MONDE RÉEL",
        "Utiliser le modèle comme en production",
        "Simulateur de session + simulation de ROI sur le test set complet.",
    )

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

        Le simulateur ci-dessous **simule l'arrivée d'une session** sur le site. Règle les
        paramètres comme si tu observais un visiteur en temps réel, et lis la décision
        produite par le modèle.
        """
    )

    if pipeline is None:
        _insight(
            "Modèle introuvable",
            "Le fichier <code>models/xgboost.joblib</code> n'existe pas. "
            "Lance <code>python scripts/train.py</code> pour générer les modèles.",
            kind="warning",
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
        _insight("Erreur de prédiction", str(e), kind="danger")
        return

    decision_emoji = "🟢" if proba >= threshold else "⚪"
    decision_text = "Cibler — action marketing recommandée" if proba >= threshold else "Ne pas cibler"

    st.markdown("---")
    st.markdown("### Décision en temps réel")

    cols = st.columns([2, 1])
    with cols[0]:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": " %", "font": {"size": 52, "color": PRIMARY, "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": GREY, "tickfont": {"color": GREY}},
                "bar": {"color": PRIMARY, "thickness": 0.7},
                "bordercolor": "#E2E8F0",
                "borderwidth": 1,
                "threshold": {
                    "line": {"color": ACCENT_DARK, "width": 4},
                    "value": threshold * 100,
                },
                "steps": [
                    {"range": [0, 30], "color": "#F1F5F9"},
                    {"range": [30, 60], "color": "#E0E7FF"},
                    {"range": [60, 100], "color": "#C7D2FE"},
                ],
            },
            title={"text": "<b>Probabilité d'achat (XGBoost)</b>",
                   "font": {"size": 16, "color": DARK, "family": "Inter"}},
        ))
        fig.update_layout(height=340, margin=dict(t=50, b=20, l=20, r=20),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with cols[1]:
        _kpi("Probabilité", f"{proba:.1%}")
        _kpi("Seuil business", f"{threshold:.3f}")
        _kpi("Confiance", f"{abs(proba - 0.5) * 2:.0%}", "écart au 50/50")
        st.markdown(f"### {decision_emoji} {decision_text}")

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
        with c1:
            _kpi("Sessions ciblées", f"{total_targeted}/{len(y_true)}",
                 f"{total_targeted/len(y_true):.0%} du trafic")
        with c2:
            _kpi("Acheteurs captés (TP)", f"{tp}/{total_real_buyers}",
                 f"recall {tp/total_real_buyers:.0%}")
        with c3:
            _kpi("Fausses alertes (FP)", f"{fp}",
                 f"precision {tp/max(tp+fp,1):.0%}")
        with c4:
            _kpi("Acheteurs ratés (FN)", f"{fn}",
                 f"{fn/total_real_buyers:.0%} ratés")

        fig = go.Figure(go.Bar(
            x=["Acheteurs captés", "Fausses alertes", "Acheteurs ratés", "Bien ignorés"],
            y=[tp, fp, fn, tn],
            marker=dict(
                color=[SUCCESS, ACCENT, WARNING, "#94A3B8"],
                line=dict(color="white", width=2),
            ),
            text=[tp, fp, fn, tn], textposition="outside",
        ))
        fig.update_layout(
            title=f"Décomposition de la décision au seuil {threshold:.2f}",
            height=340, showlegend=False, yaxis_title="Sessions",
        )
        st.plotly_chart(_style(fig), use_container_width=True)

        _insight(
            "Lecture business",
            f"<ul style='margin: 0 0 0.6rem 1.2rem; padding: 0;'>"
            f"<li>Si on cible toutes ces sessions, on touche <strong>{total_targeted} visiteurs</strong> "
            f"({total_targeted/len(y_true):.0%} du trafic).</li>"
            f"<li>Parmi eux, <strong>{tp}</strong> sont vraiment des acheteurs (gain).</li>"
            f"<li><strong>{fp}</strong> ne le sont pas (coût de l'action perdue).</li>"
            f"<li>On rate <strong>{fn}</strong> acheteurs qu'on aurait pu cibler.</li>"
            f"</ul>"
            f"<strong>Comment ajuster :</strong> baisse le seuil si une conversion vaut beaucoup "
            f"plus qu'une action marketing perdue (recall ↑, precision ↓).",
            kind="info",
        )
    else:
        _insight(
            "Données manquantes",
            "<code>results/test_predictions.csv</code> introuvable — la simulation ROI nécessite ce fichier.",
            kind="warning",
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_app() -> None:
    st.set_page_config(
        page_title="Online Shoppers — Conversion Prediction",
        layout="wide",
        page_icon="🛒",
    )
    _inject_css()
    _KPI_COUNTER["i"] = 0
    _BADGE_COUNTER["i"] = 0

    with st.sidebar:
        sidebar_html = (
            '<div style="text-align: center; padding: 1.4rem 0 1rem 0;">'
            '<div style="font-size: 2rem;">🛒</div>'
            '<div style="font-size: 1.1rem; font-weight: 800; color: white; margin-top: 0.3rem;">ML PoC</div>'
            '<div style="font-size: 0.72rem; color: rgba(255,255,255,0.6); letter-spacing: 0.18rem; '
            'text-transform: uppercase; margin-top: 0.2rem;">Dashboard 3 parties</div>'
            '</div>'
        )
        st.markdown(sidebar_html, unsafe_allow_html=True)
        st.markdown(
            '<div style="margin: 0.4rem 0.6rem; padding: 1rem; '
            'background: rgba(255,255,255,0.04); border-radius: 10px; '
            'font-size: 0.82rem; color: rgba(255,255,255,0.85); line-height: 1.6;">'
            '<div style="font-weight: 600; color: white; margin-bottom: 0.3rem;">Manech Carriou</div>'
            'Albert School · ML PoC<br>'
            '<a href="https://github.com/manechcarriou-lab/ml-poc-project" '
            'style="color: #A5B4FC !important; text-decoration: none;">→ Voir le repo</a><br>'
            '<a href="https://github.com/manechcarriou-lab/ml-poc-project/blob/main/deliverables/RAPPORT_COMPLET.md" '
            'style="color: #A5B4FC !important; text-decoration: none;">→ Lire le rapport</a>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="margin: 0.8rem 0.6rem 0; padding: 0.8rem; '
            'background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3); '
            'border-radius: 10px; font-size: 0.78rem; color: rgba(255,255,255,0.85); line-height: 1.5;">'
            '<div style="font-weight: 700; color: white; letter-spacing: 0.08rem; '
            'text-transform: uppercase; font-size: 0.68rem; margin-bottom: 0.4rem;">Pipeline</div>'
            'EDA → preprocessing<br>(anti-leakage) → 3 modèles<br>→ Optuna + MLflow<br>→ threshold tuning'
            '</div>',
            unsafe_allow_html=True,
        )

    _hero(
        "ML POC · ALBERT SCHOOL",
        "Prédire la conversion d'un visiteur e-commerce",
        "Online Shoppers Purchasing Intention — UCI Machine Learning Repository · 12 330 sessions, classification binaire déséquilibrée.",
    )

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
        part2_models_and_metrics(metrics_df, test_pred)
    with tab3:
        part3_real_world_demo(pipeline, test_pred)


if __name__ == "__main__":
    build_app()
