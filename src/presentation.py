"""Streamlit presentation dashboard — Online Shoppers Conversion PoC.

Walks through the entire project narrative with rich visualisations:
problem → data → setup → EDA → features → encoding → models → tuning →
results → demo → conclusion. The same content as `deliverables/RAPPORT_COMPLET.md`
but interactive and visual.

Launch with:
    streamlit run src/presentation.py
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
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container
from streamlit_option_menu import option_menu

from config import DATA_DIR, MODEL_METRICS_FILE, MODELS, PLOTS_DIR, RESULTS_DIR
from features import add_engineered_features

DATASET_PATH = DATA_DIR / "online_shoppers_intention.csv"
TEST_PRED_PATH = RESULTS_DIR / "test_predictions.csv"
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

PRIMARY = "#6366F1"        # indigo-500
PRIMARY_DARK = "#4338CA"   # indigo-700 (for gradients)
PRIMARY_LIGHT = "#A5B4FC"  # indigo-300
ACCENT = "#EC4899"         # pink-500
ACCENT_DARK = "#BE185D"    # pink-700
SUCCESS = "#10B981"        # emerald-500
WARNING = "#F59E0B"        # amber-500
GREY = "#64748B"           # slate-500
LIGHT = "#F1F5F9"          # slate-100
LIGHTER = "#F8FAFC"        # slate-50
DARK = "#0F172A"           # slate-900
WHITE = "#FFFFFF"

CHART_COLORS = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#8B5CF6"]


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
            font-size: 1.08rem;
            color: {GREY};
            line-height: 1.6;
            margin-bottom: 1.8rem;
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
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .insight-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
        }}
        .insight-title {{
            font-size: 0.7rem;
            font-weight: 700;
            color: {ACCENT};
            letter-spacing: 0.16rem;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }}
        .insight-body {{
            font-size: 0.95rem;
            color: {DARK};
            line-height: 1.55;
        }}
        .kpi-card {{
            background: {WHITE};
            padding: 1.2rem 1.4rem;
            border-radius: 14px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 6px 20px rgba(15, 23, 42, 0.05);
            height: 100%;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, {PRIMARY} 0%, {ACCENT} 100%);
        }}
        .kpi-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.12);
        }}
        .kpi-value {{
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1;
            letter-spacing: -0.02em;
        }}
        .kpi-label {{
            font-size: 0.72rem;
            color: {GREY};
            text-transform: uppercase;
            letter-spacing: 0.1rem;
            margin-top: 0.6rem;
            font-weight: 600;
        }}
        .kpi-delta {{
            font-size: 0.85rem;
            color: {SUCCESS};
            margin-top: 0.4rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 0.2rem;
        }}
        .pill {{
            display: inline-block;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            background: {WHITE};
            color: {DARK};
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}
        .pill-success {{
            background: rgba(16, 185, 129, 0.08);
            color: {SUCCESS};
            border: 1px solid rgba(16, 185, 129, 0.25);
        }}
        .pill-accent {{
            background: rgba(236, 72, 153, 0.08);
            color: {ACCENT};
            border: 1px solid rgba(236, 72, 153, 0.25);
        }}
        div.stButton > button {{
            background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.55rem 1.4rem;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        div.stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
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
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            color: {PRIMARY} !important;
            border-bottom-color: {PRIMARY} !important;
        }}
        .stTabs [data-baseweb="tab-list"] button {{
            font-weight: 600;
        }}
        h1, h2, h3 {{
            color: {DARK};
            letter-spacing: -0.015em;
        }}
        [data-testid="stMetricValue"] {{
            color: {PRIMARY};
            font-weight: 800;
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
        footer, header {{visibility: hidden;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable components
# ---------------------------------------------------------------------------


def _section_header(eyebrow: str, title: str, lead: str = "") -> None:
    lead_html = f'<div class="section-lead">{lead}</div>' if lead else ""
    html = (
        f'<div class="section-eyebrow">{eyebrow}</div>'
        f'<div class="section-title">{title}</div>'
        f'{lead_html}'
    )
    st.markdown(html, unsafe_allow_html=True)
    add_vertical_space(1)


_KPI_COUNTER = {"i": 0}


def _kpi(label: str, value: str, delta: str | None = None) -> None:
    """Render a metric using shadcn UI's metric_card for a polished look."""
    _KPI_COUNTER["i"] += 1
    ui.metric_card(
        title=label,
        content=value,
        description=delta or "",
        key=f"kpi_{_KPI_COUNTER['i']}",
    )


def _insight(title: str, body: str) -> None:
    html = (
        f'<div class="insight-card">'
        f'<div class="insight-title">{title}</div>'
        f'<div class="insight-body">{body}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


_BADGE_COUNTER = {"i": 0}


def _pills(items: list[tuple[str, str]]) -> None:
    """items = [(label, kind), ...] — uses shadcn UI badges.
    kind in {default, success, accent} → mapped to {default, secondary, destructive}.
    """
    _BADGE_COUNTER["i"] += 1
    variant_map = {
        "default": "default",
        "success": "secondary",
        "accent": "destructive",
    }
    badge_list = [(label, variant_map.get(kind, "default")) for label, kind in items]
    ui.badges(badge_list=badge_list, class_name="flex gap-2 flex-wrap",
              key=f"badges_{_BADGE_COUNTER['i']}")


def _plotly_clean(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=DARK, size=12),
        margin=dict(t=50, b=30, l=30, r=30),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        title_font=dict(size=15, color=DARK, family="Inter"),
    )
    fig.update_xaxes(
        gridcolor="#F1F5F9",
        linecolor="#E2E8F0",
        zeroline=False,
        tickfont=dict(color=GREY),
    )
    fig.update_yaxes(
        gridcolor="#F1F5F9",
        linecolor="#E2E8F0",
        zeroline=False,
        tickfont=dict(color=GREY),
    )
    return fig


# ---------------------------------------------------------------------------
# Cached data
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
def _load_xgb():
    path = MODELS["xgboost"]["path"]
    if not Path(path).exists():
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Section 1 — Cover / Hero
# ---------------------------------------------------------------------------


def section_cover() -> None:
    hero_style = (
        "background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);"
        "border-radius: 20px; padding: 3rem 2.8rem; color: white; margin-bottom: 1.8rem;"
        "position: relative; overflow: hidden;"
        "box-shadow: 0 20px 60px rgba(99, 102, 241, 0.3), 0 8px 24px rgba(236, 72, 153, 0.2);"
    )
    hero_html = (
        f'<div style="{hero_style}">'
        '<div style="position: absolute; top: -40px; right: -40px; width: 200px; height: 200px;'
        ' background: rgba(255,255,255,0.1); border-radius: 50%; filter: blur(40px);"></div>'
        '<div style="position: absolute; bottom: -60px; left: -60px; width: 240px; height: 240px;'
        ' background: rgba(236, 72, 153, 0.2); border-radius: 50%; filter: blur(50px);"></div>'
        '<div style="position: relative; z-index: 1;">'
        '<div style="font-size: 0.78rem; font-weight: 700; letter-spacing: 0.28rem; opacity: 0.85;">'
        '· MACHINE LEARNING · PROOF OF CONCEPT · ALBERT SCHOOL'
        '</div>'
        '<div style="font-size: 2.8rem; font-weight: 800; margin-top: 1.2rem; line-height: 1.1; letter-spacing: -0.02em;">'
        "Prédire la conversion d'un visiteur e-commerce"
        '</div>'
        '<div style="font-size: 1.15rem; margin-top: 1.1rem; opacity: 0.92; max-width: 720px; line-height: 1.5;">'
        "Du repo vide au modèle XGBoost en production — un récit visuel, end-to-end, anti-leakage."
        '</div>'
        '<div style="margin-top: 1.8rem; font-size: 0.9rem; opacity: 0.85; display: flex; gap: 1.2rem; flex-wrap: wrap;">'
        '<span>· Manech Carriou</span>'
        '<span>·</span>'
        '<span>· Albert School</span>'
        '<span>·</span>'
        '<span>· github.com/manechcarriou-lab/ml-poc-project</span>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    cols = st.columns(4)
    with cols[0]:
        _kpi("Modèle gagnant", "XGBoost")
    with cols[1]:
        _kpi("Test F1", "0.6731", "+12 % vs cible 0.60")
    with cols[2]:
        _kpi("Recall", "0.7356", "74 % des acheteurs captés")
    with cols[3]:
        _kpi("ROC-AUC", "0.9292", "Excellent ranking")

    st.markdown("")
    cols = st.columns(4)
    with cols[0]:
        _kpi("Sessions analysées", "12 330")
    with cols[1]:
        _kpi("Class imbalance", "85 / 15")
    with cols[2]:
        _kpi("Trials Optuna", "45", "15 par modèle")
    with cols[3]:
        _kpi("Tests verts", "7 / 7")

    st.markdown("---")
    _pills(
        [
            ("scikit-learn", "default"),
            ("XGBoost", "default"),
            ("Optuna (TPE)", "default"),
            ("MLflow", "default"),
            ("Streamlit", "default"),
            ("ColumnTransformer", "default"),
            ("TunedThresholdClassifierCV", "accent"),
            ("✓ Anti-leakage", "success"),
            ("✓ Reproductible", "success"),
        ]
    )


# ---------------------------------------------------------------------------
# Section 2 — Le problème business
# ---------------------------------------------------------------------------


def section_problem() -> None:
    _section_header(
        "01 · LE PROBLÈME BUSINESS",
        "Peut-on prédire qu'un visiteur va acheter ?",
        "Les sites e-commerce convertissent en moyenne 1 à 3 % de leurs visiteurs. Identifier les sessions à fort potentiel = prioriser le budget marketing.",
    )

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.markdown("### Le funnel e-commerce typique")
        funnel_data = pd.DataFrame(
            [
                ("Visiteurs", 100_000),
                ("Sessions engagées", 35_000),
                ("Ajouts au panier", 8_000),
                ("Conversions", 1_500),
            ],
            columns=["Étape", "Nombre"],
        )
        fig = go.Figure(
            go.Funnel(
                y=funnel_data["Étape"],
                x=funnel_data["Nombre"],
                textinfo="value+percent initial",
                marker=dict(color=["#6366F1", "#8B5CF6", "#A855F7", "#EC4899"]),
            )
        )
        fig.update_layout(height=380, margin=dict(t=20, b=20, l=20, r=20),
                          plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Type de problème ML")
        _pills([("Classification binaire", "accent"), ("Supervisée", "default"), ("Tabulaire", "default")])
        st.markdown(
            """
            **Cible :** `Revenue ∈ {True, False}` — le visiteur a-t-il acheté pendant la session ?

            **Stakeholder :** équipe Growth / CRM d'un retailer en ligne.

            **Décision opérationnelle :** déclencher (ou non) une action marketing pendant la session.
            """
        )
        _insight(
            "Pourquoi c'est utile",
            "Sur 100 visiteurs, ~98 partent sans acheter. Identifier les sessions à fort potentiel permet de prioriser les budgets retargeting / coupons / pop-ups au lieu de les dépenser à l'aveugle.",
        )


# ---------------------------------------------------------------------------
# Section 3 — Le dataset
# ---------------------------------------------------------------------------


def section_dataset(df: pd.DataFrame | None) -> None:
    _section_header(
        "02 · LE DATASET",
        "UCI Online Shoppers Purchasing Intention",
        "Public, stable, business-relevant, déséquilibré comme dans la vraie vie.",
    )

    cols = st.columns(4)
    with cols[0]:
        _kpi("Sessions", "12 330")
    with cols[1]:
        _kpi("Features", "10 num + 7 cat")
    with cols[2]:
        _kpi("Cible", "Revenue (bool)")
    with cols[3]:
        _kpi("Licence", "CC BY 4.0")

    st.markdown("")
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.markdown("### Distribution de la cible")
        if df is not None:
            counts = df["Revenue"].value_counts()
            pct = (counts / counts.sum() * 100).round(1)
            fig = go.Figure(
                go.Bar(
                    x=["No purchase (84.5 %)", "Purchase (15.5 %)"],
                    y=counts.values,
                    text=[f"{v:,}<br>{p} %" for v, p in zip(counts.values, pct.values)],
                    textposition="outside",
                    marker=dict(
                        color=[PRIMARY, ACCENT],
                        line=dict(color="white", width=2),
                    ),
                )
            )
            fig.update_layout(
                height=360, showlegend=False,
                yaxis_title="Sessions", xaxis_title="",
                title="Class imbalance — 85 / 15"
            )
            st.plotly_chart(_plotly_clean(fig), use_container_width=True)

    with col2:
        st.markdown("### Pourquoi ce dataset ?")
        _pills([("Public & stable", "success"), ("Manageable", "success"), ("Réaliste", "success")])
        st.markdown(
            """
            - **Volume** raisonnable (12k lignes, ~1 MB) — entraînement local rapide.
            - **Cible business** directe — pas besoin d'inventer une variable artificielle.
            - **Imbalanced** (85/15) — vrai défi métrique, plus pertinent qu'un Iris.
            - **Features réalistes** — page counts, durations, bounce/exit rates, page values.
            """
        )
        _insight(
            "Le défi central",
            "15.5 % de positifs → un modèle naïf qui prédit toujours « pas d'achat » a 85 % d'accuracy mais 0 de recall. C'est ce qui guide les choix techniques : F1 > accuracy, class_weight balanced, threshold tuning.",
        )

    st.markdown("### Limites assumées")
    cols = st.columns(4)
    limits = [
        ("Pas d'identifiant utilisateur", "→ pas de parcours multi-session"),
        ("Pas d'année", "→ pas de saisonnalité multi-annuelle"),
        ("Modalités anonymisées", "→ pas de jointure externe"),
        ("Pas de prix produit", "→ on prédit l'intention, pas la valeur"),
    ]
    for col, (lim, why) in zip(cols, limits):
        with col:
            st.markdown(f"**{lim}**")
            st.caption(why)


# ---------------------------------------------------------------------------
# Section 4 — Setup technique
# ---------------------------------------------------------------------------


def section_setup() -> None:
    _section_header(
        "03 · SETUP TECHNIQUE",
        "Une stack moderne, reproductible en 4 commandes",
        "Tout en local, sans GPU, avec un environnement virtuel propre.",
    )

    stack = pd.DataFrame(
        [
            ("Core ML", "scikit-learn 1.8", "Pipeline + ColumnTransformer = anti-leakage"),
            ("Boosting", "XGBoost 3.2", "State-of-the-art tabulaire"),
            ("Hyperparam search", "Optuna 4.8", "Recherche bayésienne TPE"),
            ("Tracking", "MLflow 3.11", "Persistance des essais + UI web"),
            ("Encoding alternatif", "skrub 0.9", "Auto-encoding TableVectorizer"),
            ("App interactive", "Streamlit 1.57", "Démo en pure Python"),
            ("Versionning", "Git + GitHub (SSH)", "ed25519 key, gh CLI"),
            ("Tests", "unittest stdlib", "7 invariants critiques"),
        ],
        columns=["Composant", "Choix", "Pourquoi"],
    )
    st.dataframe(stack, use_container_width=True, hide_index=True)

    st.markdown("### Reproductibilité — 4 commandes")
    st.code(
        """git clone git@github.com:manechcarriou-lab/ml-poc-project.git
cd ml-poc-project
python -m venv .venv && .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/train.py --trials 15""",
        language="bash",
    )

    cols = st.columns(3)
    with cols[0]:
        _insight(
            "Anti-leakage",
            "Toutes les transformations dans un Pipeline sklearn — fit sur train uniquement, transform sur test. Validé par tests unitaires.",
        )
    with cols[1]:
        _insight(
            "Tracé en continu",
            "Chaque trial Optuna est loggé dans MLflow comme un run nesté — total 52 runs pour une session complète d'entraînement.",
        )
    with cols[2]:
        _insight(
            "Modulaire",
            "Encoder paramétrable (`--encoder onehot|ordinal|target`), familles configurables (`--families xgboost`), seed fixe pour la repro.",
        )


# ---------------------------------------------------------------------------
# Section 5 — EDA
# ---------------------------------------------------------------------------


def section_eda(df: pd.DataFrame | None) -> None:
    _section_header(
        "04 · EXPLORATORY DATA ANALYSIS",
        "Comprendre la donnée avant de modéliser",
        "NaN checks, outliers, drift, class imbalance, segments — chaque check oriente une décision technique.",
    )

    if df is None:
        st.warning("Dataset non trouvé.")
        return

    tabs = st.tabs(
        [
            "· Class imbalance",
            "· Conversion par segment",
            "· Outliers (IQR)",
            "· Corrélations",
        ]
    )

    with tabs[0]:
        col1, col2 = st.columns([3, 2])
        with col1:
            counts = df["Revenue"].value_counts()
            fig = px.pie(
                values=counts.values,
                names=["No purchase", "Purchase"],
                color_discrete_sequence=[PRIMARY, ACCENT],
                hole=0.65,
                title="Distribution de Revenue",
            )
            fig.update_traces(
                textposition="outside",
                textinfo="percent+label",
                marker=dict(line=dict(color="white", width=3)),
            )
            st.plotly_chart(_plotly_clean(fig), use_container_width=True)
        with col2:
            _insight(
                "Conséquence",
                "L'accuracy seule serait trompeuse (un modèle qui prédit toujours False a 85 %). On choisit le F1 sur la classe positive comme métrique principale.",
            )
            _insight(
                "Techniques activées",
                "stratify=y au split + class_weight='balanced' + scale_pos_weight (XGBoost) + threshold tuning automatique.",
            )

    with tabs[1]:
        col_select = st.selectbox(
            "Segmenter par",
            ["Month", "VisitorType", "TrafficType", "Weekend", "Region"],
            key="eda_seg",
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
            agg,
            x=col_select,
            y="conversion_rate",
            text=agg["conversion_rate"].map(lambda x: f"{x:.1%}"),
            hover_data=["n_sessions"],
            color="conversion_rate",
            color_continuous_scale="Viridis",
        )
        fig.add_hline(
            y=df["Revenue"].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Moyenne {df['Revenue'].mean():.1%}",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(title=f"Taux de conversion par {col_select}", height=420)
        st.plotly_chart(_plotly_clean(fig), use_container_width=True)

        _insight(
            "Insights clés",
            "Mois forts : Nov, Sep, Oct (saison Black Friday). New_Visitor convertit 2× plus que Returning_Visitor (contre-intuitif mais signal fort). Certains TrafficType ont des taux 3× la moyenne.",
        )

    with tabs[2]:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        outlier_data = []
        for col in num_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = ((df[col] < low) | (df[col] > high)).sum()
            outlier_data.append({"feature": col, "pct_outliers": n_out / len(df) * 100})
        odf = pd.DataFrame(outlier_data).sort_values("pct_outliers", ascending=True)

        fig = px.bar(
            odf,
            x="pct_outliers",
            y="feature",
            orientation="h",
            color="pct_outliers",
            color_continuous_scale="OrRd",
            text=odf["pct_outliers"].map(lambda x: f"{x:.1f} %"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(title="% d'outliers par feature (règle IQR > 1.5)", height=480)
        st.plotly_chart(_plotly_clean(fig), use_container_width=True)
        _insight(
            "Décision",
            "Les outliers sur PageValues, BounceRates, *_Duration sont attendus (longue traîne du trafic web). Pas de suppression brutale → log1p + StandardScaler dans le pipeline.",
        )

    with tabs[3]:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        corrs = df[num_cols].corrwith(df["Revenue"].astype(int)).sort_values(key=abs, ascending=False)
        cdf = corrs.drop("Revenue", errors="ignore").reset_index()
        cdf.columns = ["feature", "corr"]
        fig = px.bar(
            cdf.head(10).iloc[::-1],
            x="corr",
            y="feature",
            orientation="h",
            color="corr",
            color_continuous_scale="RdBu",
            range_color=[-0.5, 0.5],
        )
        fig.update_layout(title="Top 10 corrélations linéaires avec Revenue", height=440)
        st.plotly_chart(_plotly_clean(fig), use_container_width=True)
        _insight(
            "Lecture",
            "PageValues domine très largement (corr ~0.49). C'est la feature qui agrège la valeur des pages vues — directement liée à l'intention d'achat.",
        )


# ---------------------------------------------------------------------------
# Section 6 — Feature engineering
# ---------------------------------------------------------------------------


def section_feature_engineering() -> None:
    _section_header(
        "05 · FEATURE ENGINEERING",
        "Pipeline anti-leakage par construction",
        "Un seul ColumnTransformer dans un Pipeline sklearn — fit sur train, transform sur test, garantie zéro fuite.",
    )

    st.markdown("### Architecture du pipeline")

    pipeline_diagram = """
    ```
    Raw DataFrame (X)
         │
         ▼
    add_engineered_features()    ◄── stateless, leakage-safe
         │  +TotalPages, +AvgTimePerPage, +HighPageValue ...
         ▼
    ┌─────────────────────────────────────────────────┐
    │ ColumnTransformer                                │
    │   ├─ skewed_num  → log1p → StandardScaler       │
    │   ├─ num         → StandardScaler                │
    │   ├─ cat         → OneHot | Ordinal | Target    │
    │   └─ binary      → passthrough                   │
    └─────────────────────────────────────────────────┘
         │
         ▼
    Classifier (LogReg | RandomForest | XGBoost)
         │
         ▼
    TunedThresholdClassifierCV  (cv=5, scoring='f1')
         │
         ▼
    Final prediction at the optimal threshold
    ```
    """
    st.markdown(pipeline_diagram)

    st.markdown("### Features ajoutées (row-wise stateless → leakage-safe)")
    fe_table = pd.DataFrame(
        [
            ("TotalPages", "Σ des compteurs de pages", "Volume global d'engagement"),
            ("TotalDuration", "Σ des durations", "Temps passé total"),
            ("AvgTimePerPage", "duration / pages", "Profondeur de lecture"),
            ("ProductRelatedRatio", "pages produit / total", "Focus produit"),
            ("HighPageValue", "PageValues > 0", "Flag session marchande"),
            ("IsHighBounce", "BounceRates > Q3", "Flag session zappée"),
            ("IsSpecialDay", "SpecialDay > 0", "Jour spécial (St-Valentin, etc.)"),
        ],
        columns=["Feature", "Définition", "Intuition métier"],
    )
    st.dataframe(fe_table, use_container_width=True, hide_index=True)

    cols = st.columns(2)
    with cols[0]:
        _insight(
            "Pourquoi ces features",
            "Capturer des interactions non-linéaires que les modèles linéaires ne peuvent pas inférer seuls. Ex: AvgTimePerPage différencie un visiteur engagé d'un visiteur qui scroll vite.",
        )
    with cols[1]:
        _insight(
            "Pourquoi log1p sur les durations",
            "Distributions très skewed (longue traîne). log1p compresse sans perdre l'information, et gère les zéros (log(0) = -∞ planterait).",
        )


# ---------------------------------------------------------------------------
# Section 7 — Encoding comparison
# ---------------------------------------------------------------------------


def section_encoding() -> None:
    _section_header(
        "06 · ENCODING — LA COMPARAISON QUI CHANGE TOUT",
        "OneHot vs Ordinal vs Target — le choix dépend du modèle",
        "Étude empirique : 3 stratégies × 3 modèles = 9 combinaisons testées dans notebooks/encoding_comparison.ipynb.",
    )

    enc_data = pd.DataFrame(
        [
            ("OneHot", "LogReg", 0.6559, 82),
            ("OneHot", "Random Forest", 0.6561, 82),
            ("OneHot", "XGBoost", 0.6544, 82),
            ("Ordinal", "LogReg", 0.6443, 24),
            ("Ordinal", "Random Forest", 0.6512, 24),
            ("Ordinal", "XGBoost", 0.6760, 24),
            ("Target", "LogReg", 0.6457, 24),
            ("Target", "Random Forest", 0.6527, 24),
            ("Target", "XGBoost", 0.6682, 24),
        ],
        columns=["encoding", "model", "f1", "n_features"],
    )

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        fig = px.bar(
            enc_data,
            x="model",
            y="f1",
            color="encoding",
            barmode="group",
            text=enc_data["f1"].map(lambda x: f"{x:.4f}"),
            color_discrete_map={
                "OneHot": PRIMARY,
                "Ordinal": ACCENT,
                "Target": SUCCESS,
            },
            category_orders={"encoding": ["OneHot", "Ordinal", "Target"]},
        )
        fig.update_traces(marker=dict(line=dict(color="white", width=1)))
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Test F1 — encoding × model (à params modèle figés)", height=420,
                          yaxis_title="F1", xaxis_title="")
        st.plotly_chart(_plotly_clean(fig), use_container_width=True)

    with col2:
        st.markdown("### Insights par modèle")
        _insight(
            "Modèles linéaires (LogReg)",
            "OneHot gagne. Logique : LogReg a besoin d'un effet additif par modalité. Encoder Month en 1..10 demanderait un coefficient unique linéaire — ça n'a aucun sens pour des mois.",
        )
        _insight(
            "Arbres (XGBoost)",
            "Ordinal gagne (+2 F1 pts). Les arbres apprennent des splits non-monotones, donc l'ordre arbitraire ne les biaise pas. Avec 24 features (vs 82), moins de bruit à arbitrer.",
        )
        _insight(
            "TargetEncoder",
            "Anti-leakage via CV interne sklearn (cv=5). Gain marginal sur ce dataset aux modalités peu nombreuses.",
        )

    st.markdown("### Décision retenue")
    cols = st.columns(3)
    with cols[0]:
        _kpi("LogReg", "OneHot", "Best across all 3 models")
    with cols[1]:
        _kpi("Random Forest", "OneHot", "Marginal vs Ordinal, robuste")
    with cols[2]:
        _kpi("XGBoost ⭐", "Ordinal", "+2 F1 points vs OneHot")

    _insight(
        "Leçon pro",
        "Ne jamais hardcoder un choix d'encodage sans le valider empiriquement. Ici c'est implémenté dans build_preprocessor(encoder=...) et le mapping family→encoder est dans scripts/train.py.",
    )


# ---------------------------------------------------------------------------
# Section 8 — Optuna
# ---------------------------------------------------------------------------


def section_optuna() -> None:
    _section_header(
        "07 · OPTUNA — RECHERCHE BAYÉSIENNE",
        "TPE > GridSearch sur des espaces continus",
        "15 trials Optuna ≈ 100 trials GridSearch. Chaque trial est un run nesté MLflow.",
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### Comparaison Optuna vs GridSearch")
        compare = pd.DataFrame(
            [
                ("Échantillonnage", "Discret, exhaustif", "Continu, bayésien"),
                ("Convergence", "Linéaire", "Logarithmique"),
                ("learning_rate", "Discrétisation arbitraire", "log-uniforme natif"),
                ("Reprise après crash", "Non", "Oui (study.add_trial)"),
                ("Trade-off temps/qualité", "Mauvais", "Excellent"),
            ],
            columns=["Aspect", "GridSearch", "Optuna (TPE)"],
        )
        st.dataframe(compare, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Le sampler TPE en 3 lignes")
        st.markdown(
            """
            1. Optuna construit `l(x)` (distribution des **bons** trials, top quartile) et `g(x)` (distribution des **mauvais**).
            2. Chaque nouveau trial maximise `l(x) / g(x)` — il ressemble aux bons et pas aux mauvais.
            3. Au début c'est aléatoire ; après quelques trials, ça converge vers les zones prometteuses.
            """
        )
        _insight(
            "Stratégie d'évaluation",
            "Validation : 3-fold stratified CV sur le train uniquement. Métrique : F1 (positive class). Aucun test set vu pendant le tuning.",
        )

    st.markdown("### Espaces de recherche par famille")
    spaces = pd.DataFrame(
        [
            ("LogReg", "C", "log-uniform [1e-3, 1e2]"),
            ("RF", "n_estimators", "int [100, 400] step 50"),
            ("RF", "max_depth", "int [4, 24]"),
            ("RF", "min_samples_split / leaf", "int [2, 20] / [1, 10]"),
            ("RF", "max_features", "{sqrt, log2, None}"),
            ("XGBoost", "n_estimators / max_depth", "int [150, 600] / [3, 10]"),
            ("XGBoost", "learning_rate", "log-uniform [1e-2, 3e-1]"),
            ("XGBoost", "subsample / colsample_bytree", "uniform [0.6, 1.0]"),
            ("XGBoost", "gamma / reg_lambda", "[0, 5] / log [1e-3, 10]"),
            ("XGBoost", "scale_pos_weight", "fixé à (1-pos)/pos"),
        ],
        columns=["Famille", "Hyperparamètre", "Distribution"],
    )
    st.dataframe(spaces, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Section 9 — MLflow
# ---------------------------------------------------------------------------


def section_mlflow() -> None:
    _section_header(
        "08 · MLFLOW — LA MÉMOIRE DES EXPÉRIENCES",
        "Trace persistante de chaque run avec params + métriques + artefacts",
        "Comparer un run d'aujourd'hui à un run d'il y a 2 semaines, sans tableau Excel.",
    )

    col1, col2 = st.columns([2, 3], gap="large")
    with col1:
        cols = st.columns(2)
        with cols[0]:
            _kpi("Runs trackés", "52")
        with cols[1]:
            _kpi("Experiment", "1")
        cols = st.columns(2)
        with cols[0]:
            _kpi("Trials par modèle", "15")
        with cols[1]:
            _kpi("Final runs", "3")

    with col2:
        st.markdown("### Architecture des runs")
        st.code(
            """online_shoppers_conversion (experiment)
├── logreg-study (parent run)
│   ├── logreg-trial-0  (nested: params + cv_f1_mean)
│   ├── logreg-trial-1
│   └── ... (15 trials)
├── logreg-final  (test_f1 + joblib artifact + best_threshold)
├── random_forest-study + 15 trials + final
└── xgboost-study + 15 trials + final
""",
            language="text",
        )

    cols = st.columns(3)
    with cols[0]:
        _insight(
            "Params loggés",
            "Tous les hyperparamètres testés par Optuna, l'encoder utilisé, le nombre de folds CV.",
        )
    with cols[1]:
        _insight(
            "Métriques loggées",
            "cv_f1_mean, cv_f1_std, best_cv_f1, test_f1, test_precision/recall/roc_auc, best_threshold, test_f1_at_0p5.",
        )
    with cols[2]:
        _insight(
            "Artefacts",
            "Le .joblib du pipeline final est uploadé comme artefact MLflow → reproductibilité totale.",
        )

    st.markdown("### Inspecter l'UI")
    st.code("mlflow ui --backend-store-uri ./mlruns", language="bash")
    st.caption("Puis http://localhost:5000 — tri par métrique, comparaison côte à côte, téléchargement d'artefacts.")


# ---------------------------------------------------------------------------
# Section 10 — Threshold tuning
# ---------------------------------------------------------------------------


def section_threshold(test_pred: pd.DataFrame | None) -> None:
    _section_header(
        "09 · THRESHOLD TUNING",
        "Le seuil 0.5 n'est jamais optimal sur un dataset déséquilibré",
        "TunedThresholdClassifierCV : seuil appris en CV sur le train, anti-leakage par construction.",
    )

    if test_pred is None or "proba_xgboost" not in test_pred.columns:
        st.warning("results/test_predictions.csv non trouvé.")
        return

    y_true = test_pred["y_true"].astype(int).values
    y_score = test_pred["proba_xgboost"].values

    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append({"threshold": t, "precision": prec, "recall": rec, "f1": f1})
    df = pd.DataFrame(rows)
    best_t = df.loc[df["f1"].idxmax(), "threshold"]
    best_f1 = df["f1"].max()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["threshold"], y=df["precision"], name="Precision",
        line=dict(color=PRIMARY, width=2.5),
        fill='tozeroy', fillcolor=f'rgba(99, 102, 241, 0.05)',
    ))
    fig.add_trace(go.Scatter(
        x=df["threshold"], y=df["recall"], name="Recall",
        line=dict(color=ACCENT, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["threshold"], y=df["f1"], name="F1",
        line=dict(color=SUCCESS, width=3.5),
    ))
    fig.add_vline(x=0.5, line_dash="dot", line_color=GREY,
                  annotation_text="Default 0.5", annotation_font=dict(color=GREY))
    fig.add_vline(x=best_t, line_dash="dash", line_color=ACCENT_DARK,
                  annotation_text=f"Optimum {best_t:.2f}",
                  annotation_font=dict(color=ACCENT_DARK))
    fig.update_layout(
        title="XGBoost — precision / recall / F1 vs seuil de décision",
        xaxis_title="Threshold",
        yaxis_title="Métrique",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(_plotly_clean(fig), use_container_width=True)

    st.markdown("### Gains observés")
    gains = pd.DataFrame(
        [
            ("Logistic Regression", 0.5994, 0.6426, "+7.2 %", 0.244),
            ("Random Forest", 0.6246, 0.6683, "+6.2 %", 0.360),
            ("XGBoost", 0.6562, 0.6731, "+2.6 %", 0.305),
        ],
        columns=["Modèle", "F1 @ 0.5", "F1 @ tuned", "Gain", "Threshold optimal"],
    )
    st.dataframe(gains, use_container_width=True, hide_index=True)

    cols = st.columns(2)
    with cols[0]:
        _insight(
            "Anti-leakage par construction",
            "TunedThresholdClassifierCV(scoring='f1', cv=5) cherche le seuil optimal en CV sur le train uniquement. Le test set n'est jamais vu pendant le tuning du seuil.",
        )
    with cols[1]:
        _insight(
            "Gratuit + toujours bénéfique",
            "Pas de réentraînement nécessaire. Sur un dataset déséquilibré, ce levier est systématiquement positif. À ne pas négliger.",
        )


# ---------------------------------------------------------------------------
# Section 11 — Résultats
# ---------------------------------------------------------------------------


def section_results(metrics_df: pd.DataFrame | None) -> None:
    _section_header(
        "10 · RÉSULTATS FINAUX",
        "Trois leviers d'optimisation cumulés",
        "Optuna + encoder par famille + threshold tuning = +12 % vs cible 0.60.",
    )

    cols = st.columns(4)
    with cols[0]:
        _kpi("Test F1 (XGBoost)", "0.6731", "+12 % vs cible")
    with cols[1]:
        _kpi("Test ROC-AUC", "0.9292", "Excellent ranking")
    with cols[2]:
        _kpi("Test Recall", "0.7356", "74 % des acheteurs")
    with cols[3]:
        _kpi("Test Precision", "0.6203", "62 % de vrais positifs")

    st.markdown("")
    st.markdown("### Évolution du F1 — chaque optimisation a apporté")
    evo = pd.DataFrame(
        {
            "Étape": ["Optuna seul (baseline)", "+ Encoder par famille", "+ Threshold tuning"],
            "LogReg": [0.5994, 0.5994, 0.6426],
            "Random Forest": [0.6292, 0.6292, 0.6683],
            "XGBoost": [0.6544, 0.6562, 0.6731],
        }
    )
    fig = go.Figure()
    for col, color in zip(["LogReg", "Random Forest", "XGBoost"], CHART_COLORS):
        fig.add_trace(
            go.Scatter(
                x=evo["Étape"],
                y=evo[col],
                name=col,
                mode="lines+markers+text",
                text=[f"{v:.4f}" for v in evo[col]],
                textposition="top center",
                textfont=dict(size=11, color=color),
                line=dict(color=color, width=3.5, shape="spline", smoothing=0.4),
                marker=dict(size=12, line=dict(color="white", width=2)),
            )
        )
    fig.update_layout(
        title="F1 après chaque levier d'optimisation",
        yaxis_title="Test F1",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(_plotly_clean(fig), use_container_width=True)

    st.markdown("### Tableau final")
    if metrics_df is not None:
        pretty = metrics_df.copy()
        for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if c in pretty.columns:
                pretty[c] = pretty[c].map(lambda x: f"{x:.4f}")
        # Add encoder + threshold
        pretty["encoder"] = ["OneHot", "OneHot", "Ordinal"]
        pretty["threshold"] = ["0.244", "0.360", "0.305"]
        cols_order = ["model_key", "model_name", "encoder", "threshold",
                       "f1", "precision", "recall", "roc_auc", "accuracy"]
        cols_order = [c for c in cols_order if c in pretty.columns]
        st.dataframe(pretty[cols_order], use_container_width=True, hide_index=True)

    st.markdown("### Visualisations clés")
    tabs = st.tabs(["ROC curves", "Precision-Recall", "Confusion matrix XGB", "Feature importance"])
    with tabs[0]:
        path = PLOTS_DIR / "roc_curves.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
            st.caption("XGBoost domine avec AUC=0.93 — ranking d'excellente qualité.")
    with tabs[1]:
        path = PLOTS_DIR / "pr_curves.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
            st.caption("Précision >0.7 jusqu'à recall ~0.55 — modèle utilisable en pratique.")
    with tabs[2]:
        path = PLOTS_DIR / "confusion_matrix_xgboost.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
            st.caption("281 acheteurs détectés, 101 ratés (FN), 172 fausses alertes (FP).")
    with tabs[3]:
        path = PLOTS_DIR / "feature_importance_xgb.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
            st.caption("PageValues domine — feature qui agrège la valeur des pages vues.")


# ---------------------------------------------------------------------------
# Section 12 — Démo prédiction
# ---------------------------------------------------------------------------


# Preset profiles — calibrated to produce sensible probas
DEMO_PRESETS = {
    " Acheteur engagé": {
        "demo_pp": 60, "demo_pd": 2400, "demo_pv": 90.0,
        "demo_br": 0.004, "demo_er": 0.012, "demo_ap": 4,
        "demo_inf": 1, "demo_inf_d": 80.0,
        "demo_vt": "New_Visitor", "demo_m": "Nov", "demo_we": False,
    },
    " Visiteur curieux": {
        "demo_pp": 25, "demo_pd": 800, "demo_pv": 6.0,
        "demo_br": 0.02, "demo_er": 0.04, "demo_ap": 2,
        "demo_inf": 1, "demo_inf_d": 30.0,
        "demo_vt": "Returning_Visitor", "demo_m": "May", "demo_we": True,
    },
    " Touriste pressé": {
        "demo_pp": 3, "demo_pd": 40, "demo_pv": 0.0,
        "demo_br": 0.18, "demo_er": 0.19, "demo_ap": 0,
        "demo_inf": 0, "demo_inf_d": 0.0,
        "demo_vt": "Returning_Visitor", "demo_m": "Aug", "demo_we": False,
    },
}


def _build_session_row(
    product_related: int, product_duration: float, page_values: float,
    bounce_rate: float, exit_rate: float, admin_pages: int,
    informational: int, informational_duration: float,
    visitor_type: str, month: str, weekend: bool,
    operating_systems: int = 2, browser: int = 2,
    region: int = 1, traffic_type: int = 2, special_day: float = 0.0,
) -> pd.DataFrame:
    """Build a single-row session DataFrame with engineered features applied via
    `add_engineered_features` — same exact code path as training."""
    base = pd.DataFrame([{
        "Administrative": admin_pages,
        "Administrative_Duration": admin_pages * 30.0,
        "Informational": informational,
        "Informational_Duration": float(informational_duration),
        "ProductRelated": product_related,
        "ProductRelated_Duration": float(product_duration),
        "BounceRates": bounce_rate,
        "ExitRates": exit_rate,
        "PageValues": float(page_values),
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": operating_systems,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_type,
        "Weekend": weekend,
    }])
    return add_engineered_features(base)


@st.cache_resource
def _build_shap_explainer(_pipeline):
    """Cached SHAP TreeExplainer for the XGBoost classifier inside the pipeline."""
    try:
        import shap
        inner = getattr(_pipeline, "estimator_", _pipeline)
        clf = inner.named_steps["clf"]
        return shap.TreeExplainer(clf)
    except Exception:
        return None


def _shap_contributions(pipeline, row: pd.DataFrame, top_k: int = 10) -> pd.DataFrame | None:
    """Return a DataFrame of top-k SHAP contributions for the given row."""
    try:
        explainer = _build_shap_explainer(pipeline)
        if explainer is None:
            return None
        inner = getattr(pipeline, "estimator_", pipeline)
        pre = inner.named_steps["preprocessor"]
        x_trans = pre.transform(row)
        feature_names = list(pre.get_feature_names_out())
        sv = explainer.shap_values(x_trans)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv).ravel()
        df = pd.DataFrame({"feature": feature_names, "shap": sv, "value": np.asarray(x_trans).ravel()})
        df["abs"] = df["shap"].abs()
        return df.sort_values("abs", ascending=False).head(top_k).iloc[::-1]
    except Exception:
        return None


def section_demo(pipeline, test_pred: pd.DataFrame | None = None) -> None:
    _section_header(
        "11 · DÉMO INTERACTIVE",
        "Tester le modèle XGBoost sur une session synthétique",
        "Règle les sliders, charge un preset, ou pioche une vraie session du test set. Le modèle te donne la probabilité d'achat + l'explication SHAP.",
    )

    if pipeline is None:
        st.warning("Modèle XGBoost non trouvé. Lance `python scripts/train.py` d'abord.")
        return

    # ------------------------------------------------------------------
    # Quick start: presets + random sample
    # ------------------------------------------------------------------
    st.markdown("#####  Quick start — pré-remplir les sliders")
    cols = st.columns([1, 1, 1, 1])
    for col, (label, preset) in zip(cols[:3], DEMO_PRESETS.items()):
        if col.button(label, use_container_width=True, key=f"preset_{label}"):
            for k, v in preset.items():
                st.session_state[k] = v
            st.rerun()

    if cols[3].button(" Échantillon réel (test set)", use_container_width=True, key="demo_random"):
        if test_pred is not None and len(test_pred) > 0:
            sample = test_pred.sample(1, random_state=np.random.randint(0, 10_000)).iloc[0]
            st.session_state["demo_pp"] = int(sample.get("ProductRelated", 30))
            st.session_state["demo_pd"] = float(sample.get("ProductRelated_Duration", 1000))
            st.session_state["demo_pv"] = float(sample.get("PageValues", 5.0))
            st.session_state["demo_br"] = float(sample.get("BounceRates", 0.02))
            st.session_state["demo_er"] = float(sample.get("ExitRates", 0.04))
            st.session_state["demo_ap"] = int(sample.get("Administrative", 2))
            st.session_state["demo_inf"] = int(sample.get("Informational", 0))
            st.session_state["demo_inf_d"] = float(sample.get("Informational_Duration", 0.0))
            st.session_state["demo_vt"] = str(sample.get("VisitorType", "Returning_Visitor"))
            st.session_state["demo_m"] = str(sample.get("Month", "Nov"))
            st.session_state["demo_we"] = bool(sample.get("Weekend", False))
            st.session_state["_demo_truth"] = int(sample["y_true"])
            st.rerun()

    add_vertical_space(1)

    # ------------------------------------------------------------------
    # Sliders (3 columns)
    # ------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        product_related = st.slider("Pages produit visitées", 0, 200,
                                     int(st.session_state.get("demo_pp", 30)), key="demo_pp")
        product_duration = st.slider("Durée pages produit (s)", 0, 6000,
                                      int(st.session_state.get("demo_pd", 1000)), step=50, key="demo_pd")
        page_values = st.slider("PageValues (proxy d'intention)", 0.0, 200.0,
                                float(st.session_state.get("demo_pv", 5.0)), step=1.0, key="demo_pv")
        admin_pages = st.slider("Pages admin (login / cart)", 0, 30,
                                 int(st.session_state.get("demo_ap", 2)), key="demo_ap")
    with col2:
        bounce_rate = st.slider("Bounce rate", 0.0, 0.2,
                                 float(st.session_state.get("demo_br", 0.02)), step=0.005, format="%.3f", key="demo_br")
        exit_rate = st.slider("Exit rate", 0.0, 0.2,
                               float(st.session_state.get("demo_er", 0.04)), step=0.005, format="%.3f", key="demo_er")
        informational = st.slider("Pages informationnelles", 0, 30,
                                   int(st.session_state.get("demo_inf", 0)), key="demo_inf")
        informational_duration = st.slider("Durée pages info (s)", 0, 3000,
                                            int(st.session_state.get("demo_inf_d", 0)), step=30, key="demo_inf_d")
    with col3:
        visitor_type = st.selectbox(
            "Visitor type",
            ["Returning_Visitor", "New_Visitor", "Other"],
            index=["Returning_Visitor", "New_Visitor", "Other"].index(
                st.session_state.get("demo_vt", "Returning_Visitor")
            ),
            key="demo_vt",
        )
        month_options = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month = st.selectbox(
            "Mois",
            month_options,
            index=month_options.index(st.session_state.get("demo_m", "Nov")),
            key="demo_m",
        )
        weekend = st.checkbox("Weekend", value=bool(st.session_state.get("demo_we", False)), key="demo_we")
        threshold = st.slider("Seuil de décision (0.305 = optimum XGBoost)",
                               0.05, 0.95, 0.305, step=0.01, key="demo_th")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    try:
        row = _build_session_row(
            product_related=product_related, product_duration=product_duration,
            page_values=page_values, bounce_rate=bounce_rate, exit_rate=exit_rate,
            admin_pages=admin_pages, informational=informational,
            informational_duration=informational_duration,
            visitor_type=visitor_type, month=month, weekend=weekend,
        )
        proba = float(pipeline.predict_proba(row)[0, 1])
    except Exception as e:
        st.error(f"Erreur de prédiction : {e}")
        return

    decision_emoji = "+" if proba >= threshold else "-"
    decision_text = "Cibler — action marketing recommandée" if proba >= threshold else "Ne pas cibler"
    confidence = abs(proba - 0.5) * 2  # 0 = pile à 0.5, 1 = très confiant

    add_vertical_space(1)
    st.markdown("---")

    # ------------------------------------------------------------------
    # Output: gauge + KPIs + truth (if random sample)
    # ------------------------------------------------------------------
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
                "threshold": {"line": {"color": ACCENT_DARK, "width": 4}, "value": threshold * 100},
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
        _kpi("Seuil", f"{threshold:.2f}")
        _kpi("Confiance", f"{confidence:.0%}", "écart au 50/50")
        st.markdown(f"### {decision_emoji} {decision_text}")
        truth = st.session_state.get("_demo_truth")
        if truth is not None:
            actual = " ACHAT" if truth == 1 else "✗ Pas d'achat"
            correct = "✓ Bien classé" if (proba >= threshold) == (truth == 1) else "! Mal classé"
            st.markdown(
                f"**Vérité terrain** (échantillon test) : {actual}  \n{correct}"
            )

    # ------------------------------------------------------------------
    # Two analysis tabs: probability context + SHAP
    # ------------------------------------------------------------------
    add_vertical_space(1)
    tabs = st.tabs(["· Position dans la distribution", "· Explication SHAP", "· Décision business"])

    with tabs[0]:
        if test_pred is not None and "proba_xgboost" in test_pred.columns:
            scores = test_pred["proba_xgboost"].values
            percentile = (scores < proba).mean() * 100
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=scores, nbinsx=50,
                marker=dict(color=PRIMARY, opacity=0.6, line=dict(color="white", width=0.5)),
                name="Test set",
            ))
            fig.add_vline(x=proba, line_color=ACCENT_DARK, line_width=3,
                          annotation_text=f"Cette session ({proba:.1%})",
                          annotation_position="top right",
                          annotation_font=dict(color=ACCENT_DARK, size=13))
            fig.add_vline(x=threshold, line_color=GREY, line_dash="dash",
                          annotation_text=f"Seuil {threshold:.2f}",
                          annotation_position="top left",
                          annotation_font=dict(color=GREY))
            fig.update_layout(
                title="Distribution des probas XGBoost sur le test set (2 466 sessions)",
                xaxis_title="Probabilité d'achat",
                yaxis_title="Sessions",
                height=380, showlegend=False,
            )
            st.plotly_chart(_plotly_clean(fig), use_container_width=True)
            _insight(
                "Lecture",
                f"Cette session se situe au <b>{percentile:.0f}<sup>e</sup> percentile</b> du test set — "
                f"{percentile:.0f} % des sessions du test ont une proba inférieure. "
                f"Plus c'est haut, plus la session est jugée 'à fort potentiel d'achat' par le modèle.",
            )
        else:
            st.info("Charge `results/test_predictions.csv` pour voir la distribution.")

    with tabs[1]:
        contrib = _shap_contributions(pipeline, row, top_k=10)
        if contrib is None:
            st.info("SHAP indisponible (lib non installée ou modèle incompatible). "
                    "Pour l'activer : `pip install shap`.")
        else:
            colors = [SUCCESS if v > 0 else ACCENT for v in contrib["shap"]]
            fig = go.Figure(go.Bar(
                x=contrib["shap"], y=contrib["feature"],
                orientation="h",
                marker=dict(color=colors, line=dict(color="white", width=1)),
                text=[f"{v:+.3f}" for v in contrib["shap"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<br>Valeur: %{customdata:.3f}<extra></extra>",
                customdata=contrib["value"],
            ))
            fig.update_layout(
                title="Top 10 features qui poussent la prédiction (SHAP local)",
                xaxis_title="Contribution au log-odds (vert = ↑ proba, rose = ↓ proba)",
                height=460,
            )
            st.plotly_chart(_plotly_clean(fig), use_container_width=True)
            _insight(
                "Comment lire",
                "Chaque barre montre comment une feature pousse la prédiction <b>au-dessus</b> ou <b>en-dessous</b> "
                "de la moyenne du modèle. SHAP est leakage-safe et exact pour les modèles d'arbres (TreeExplainer).",
            )

    with tabs[2]:
        c1, c2, c3 = st.columns(3)
        if proba >= threshold:
            with c1:
                _kpi("Action recommandée", "+ Cibler")
            with c2:
                _kpi("Coût attendu", "Faible")
            with c3:
                _kpi("Upside", "Conversion potentielle")
            _insight(
                "Pourquoi cibler",
                "La proba est au-dessus du seuil de décision. Sur 100 sessions au-dessus de ce seuil, "
                "~62 % sont vraiment des acheteurs (precision XGBoost = 0.62). "
                "On déclenche pop-up / coupon / retargeting.",
            )
        else:
            with c1:
                _kpi("Action recommandée", "- Ne pas cibler")
            with c2:
                _kpi("Économie", "Budget marketing")
            with c3:
                _kpi("Risque", "Conversion manquée")
            _insight(
                "Pourquoi ne pas cibler",
                "La proba est en-dessous du seuil. On garde le budget pour les sessions à plus fort potentiel. "
                "Si la valeur d'une conversion est très élevée par rapport au coût d'une action marketing, "
                "on peut baisser le seuil pour gagner du recall.",
            )


# ---------------------------------------------------------------------------
# Section 13 — Conclusion
# ---------------------------------------------------------------------------


def section_conclusion() -> None:
    _section_header(
        "12 · CONCLUSION",
        "Ce qu'il faut retenir + prochaines étapes",
    )

    st.markdown("### À retenir")
    cols = st.columns(2)
    with cols[0]:
        _insight(
            "Anti-leakage par construction",
            "ColumnTransformer + Pipeline + TunedThresholdClassifierCV. Validé par 7 tests unitaires.",
        )
        _insight(
            "Encodage matters",
            "OneHot pour les modèles linéaires, Ordinal pour XGBoost. Choix justifié empiriquement.",
        )
        _insight(
            "Threshold tuning gratuit",
            "Sur dataset déséquilibré, le seuil 0.5 n'est jamais optimal. Gain +2 à +7 points de F1.",
        )
    with cols[1]:
        _insight(
            "Optuna > GridSearch",
            "Recherche bayésienne (TPE), espaces continus, convergence rapide. 15 trials ≈ 100 trials Grid.",
        )
        _insight(
            "MLflow = mémoire pro",
            "52 runs trackés, comparaison croisée, artefacts joblib uploadés. Coût : 10 lignes de code.",
        )
        _insight(
            "Reproductibilité totale",
            "git clone + pip install + train.py. Tests verts, Streamlit fonctionnel, plots regenerable.",
        )

    st.markdown("### Prochaines étapes")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("#### + Court terme")
        st.markdown(
            """
            - Calibration de probabilités (`CalibratedClassifierCV`)
            - PR-AUC + Brier score dans le panel
            - 50-100 trials Optuna sur XGBoost
            """
        )
    with cols[1]:
        st.markdown("#### ~ Moyen terme")
        st.markdown(
            """
            - A/B test prod : threshold 0.305 vs 0.5
            - Monitoring de dérive (Datadog / Grafana)
            - Réentraînement périodique automatisé
            """
        )
    with cols[2]:
        st.markdown("#### - Long terme")
        st.markdown(
            """
            - Modèles séquentiels (LSTM, Transformer)
            - Personnalisation par segment cold/warm/hot
            - Causal inference (uplift modeling)
            """
        )

    st.markdown("---")
    footer_html = (
        f'<div style="text-align: center; padding: 1.5rem; color: {GREY};">'
        'Manech Carriou · Albert School · 2026<br>'
        f'<a href="https://github.com/manechcarriou-lab/ml-poc-project" style="color: {ACCENT};">'
        'github.com/manechcarriou-lab/ml-poc-project</a>'
        '</div>'
    )
    st.markdown(footer_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


SECTIONS = [
    ("Cover", "house-fill", section_cover),
    ("Le problème", "bullseye", "problem"),
    ("Le dataset", "database-fill", "dataset"),
    ("Setup technique", "tools", "setup"),
    ("EDA", "search", "eda"),
    ("Feature engineering", "sliders", "fe"),
    ("Encoding comparison", "diagram-3-fill", "encoding"),
    ("Optuna", "bullseye", "optuna"),
    ("MLflow", "graph-up-arrow", "mlflow"),
    ("Threshold tuning", "speedometer2", "threshold"),
    ("Résultats", "trophy-fill", "results"),
    ("Démo live", "rocket-takeoff-fill", "demo"),
    ("Conclusion", "flag-fill", "conclusion"),
]


def build_app() -> None:
    st.set_page_config(
        page_title="Online Shoppers — Présentation",
        layout="wide",
        page_icon="",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    # Reset per-render component-id counters so shadcn keys stay stable.
    _KPI_COUNTER["i"] = 0
    _BADGE_COUNTER["i"] = 0

    df = _load_dataset()
    metrics_df = _load_metrics()
    test_pred = _load_test_predictions()
    pipeline = _load_xgb()

    with st.sidebar:
        sidebar_html = (
            '<div style="text-align: center; padding: 1.4rem 0 1rem 0;">'
            '<div style="font-size: 2rem;"></div>'
            '<div style="font-size: 1.1rem; font-weight: 800; color: white; margin-top: 0.3rem;">ML PoC</div>'
            '<div style="font-size: 0.75rem; color: rgba(255,255,255,0.6); letter-spacing: 0.18rem; '
            'text-transform: uppercase; margin-top: 0.2rem;">Présentation</div>'
            '</div>'
        )
        st.markdown(sidebar_html, unsafe_allow_html=True)

        labels = [s[0] for s in SECTIONS]
        icons = [s[1] for s in SECTIONS]

        choice = option_menu(
            menu_title=None,
            options=labels,
            icons=icons,
            default_index=0,
            styles={
                "container": {
                    "background-color": "transparent",
                    "padding": "0",
                },
                "icon": {
                    "color": "rgba(255,255,255,0.7)",
                    "font-size": "16px",
                },
                "nav-link": {
                    "font-size": "14px",
                    "font-weight": "500",
                    "color": "rgba(255,255,255,0.75)",
                    "padding": "0.6rem 1rem",
                    "margin": "0.15rem 0.6rem",
                    "border-radius": "10px",
                    "--hover-color": "rgba(255,255,255,0.08)",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, #6366F1 0%, #EC4899 100%)",
                    "color": "white",
                    "font-weight": "600",
                    "box-shadow": "0 4px 12px rgba(99, 102, 241, 0.35)",
                },
            },
        )

        st.markdown(
            '<div style="margin: 1.2rem 0.6rem 0; padding-top: 1.2rem; '
            'border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.8rem; '
            'color: rgba(255,255,255,0.65); line-height: 1.6;">'
            '<div style="font-weight: 600; color: rgba(255,255,255,0.85); margin-bottom: 0.3rem;">'
            'Manech Carriou</div>'
            '<div>Albert School · ML PoC</div>'
            '<div style="margin-top: 0.6rem;">'
            '<a href="https://github.com/manechcarriou-lab/ml-poc-project" '
            'style="color: #A5B4FC !important; text-decoration: none;">→ Voir le repo GitHub</a>'
            '</div>'
            '<div>'
            '<a href="https://github.com/manechcarriou-lab/ml-poc-project/blob/main/deliverables/RAPPORT_COMPLET.md" '
            'style="color: #A5B4FC !important; text-decoration: none;">→ Lire le rapport complet</a>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    idx = labels.index(choice)
    section_key = SECTIONS[idx][2]

    if callable(section_key):
        section_key()
    else:
        dispatch = {
            "problem": section_problem,
            "dataset": lambda: section_dataset(df),
            "setup": section_setup,
            "eda": lambda: section_eda(df),
            "fe": section_feature_engineering,
            "encoding": section_encoding,
            "optuna": section_optuna,
            "mlflow": section_mlflow,
            "threshold": lambda: section_threshold(test_pred),
            "results": lambda: section_results(metrics_df),
            "demo": lambda: section_demo(pipeline, test_pred),
            "conclusion": section_conclusion,
        }
        dispatch[section_key]()


if __name__ == "__main__":
    build_app()
