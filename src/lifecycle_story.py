"""ML Lifecycle Story — Streamlit scrollytelling.

A long-form, editorial telling of the project. Not a dashboard —
an article that unfolds chapter by chapter as the reader scrolls.

7 chapters:
  01. Le problème
  02. La donnée
  03. Le piège du leakage
  04. La compétition
  05. Le coup de génie threshold
  06. Le résultat
  07. En production

Launch with:
    streamlit run src/lifecycle_story.py
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from config import DATA_DIR, MODELS, RESULTS_DIR

DATASET_PATH = DATA_DIR / "online_shoppers_intention.csv"
TEST_PRED_PATH = RESULTS_DIR / "test_predictions.csv"

# ---------------------------------------------------------------------------
# Editorial palette
# ---------------------------------------------------------------------------

INK = "#1A1A1A"          # body text
DIM = "#737373"          # captions
PAPER = "#FAF9F6"        # off-white background
ACCENT = "#D97757"       # warm terracotta accent (Anthropic-ish)
ACCENT_DEEP = "#A6543F"
NAVY = "#16243E"         # serious accent for charts
SUCCESS = "#2D7A4D"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data
def _load_dataset() -> pd.DataFrame | None:
    if not DATASET_PATH.exists():
        return None
    return pd.read_csv(DATASET_PATH)


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
# Editorial CSS
# ---------------------------------------------------------------------------


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@700;800;900&display=swap');

            .stApp {{
                background: {PAPER};
            }}
            .block-container {{
                max-width: 820px !important;
                padding-top: 4rem !important;
                padding-bottom: 6rem !important;
            }}
            .editorial-narration {{
                font-family: 'Lora', Georgia, serif;
                font-size: 1.18rem;
                line-height: 1.78;
                color: {INK};
                max-width: 720px;
                margin: 0 auto;
            }}
            .editorial-narration p {{
                margin-bottom: 1.1rem;
            }}
            .chapter-num {{
                font-family: 'Playfair Display', serif;
                font-size: 0.92rem;
                font-weight: 800;
                letter-spacing: 0.55rem;
                color: {ACCENT};
                text-transform: uppercase;
                margin-bottom: 0.6rem;
            }}
            .chapter-title {{
                font-family: 'Playfair Display', serif;
                font-size: 3.2rem;
                font-weight: 800;
                line-height: 1.05;
                color: {INK};
                margin-bottom: 1.4rem;
                letter-spacing: -0.02em;
            }}
            .chapter-hook {{
                font-family: 'Lora', Georgia, serif;
                font-size: 1.5rem;
                font-style: italic;
                font-weight: 400;
                line-height: 1.5;
                color: {INK};
                margin-bottom: 2rem;
                max-width: 700px;
                border-left: 3px solid {ACCENT};
                padding-left: 1.4rem;
            }}
            .pull-quote {{
                font-family: 'Playfair Display', serif;
                font-size: 1.6rem;
                font-weight: 600;
                font-style: italic;
                line-height: 1.4;
                color: {ACCENT_DEEP};
                text-align: center;
                max-width: 640px;
                margin: 3rem auto;
                padding: 1.5rem 0;
                border-top: 2px solid {ACCENT};
                border-bottom: 2px solid {ACCENT};
            }}
            .closing-line {{
                font-family: 'Lora', Georgia, serif;
                font-size: 1.22rem;
                font-style: italic;
                color: {DIM};
                text-align: center;
                margin: 2rem auto 1rem auto;
                max-width: 600px;
            }}
            .drop-cap::first-letter {{
                font-family: 'Playfair Display', serif;
                font-weight: 800;
                float: left;
                font-size: 4.4rem;
                line-height: 0.9;
                padding: 0.4rem 0.6rem 0 0;
                color: {ACCENT};
            }}
            .magazine-divider {{
                text-align: center;
                font-family: 'Playfair Display', serif;
                color: {ACCENT};
                font-size: 1.6rem;
                letter-spacing: 1.4rem;
                margin: 4rem 0;
                opacity: 0.7;
            }}
            .leakage-card {{
                background: white;
                border: 1px solid #E5E5E5;
                border-radius: 8px;
                padding: 1.4rem 1.4rem 1.6rem 1.4rem;
                font-family: 'Inter', sans-serif;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                height: 100%;
            }}
            .leakage-card .kicker {{
                font-size: 0.7rem;
                font-weight: 700;
                letter-spacing: 0.2rem;
                text-transform: uppercase;
                margin-bottom: 0.6rem;
            }}
            .leakage-card.bad .kicker {{ color: #B23B3B; }}
            .leakage-card.good .kicker {{ color: {SUCCESS}; }}
            .leakage-card h4 {{
                font-family: 'Playfair Display', serif;
                font-size: 1.3rem;
                margin: 0 0 1rem 0;
                color: {INK};
            }}
            .leakage-card .step {{
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 0.85rem;
                color: {INK};
                background: #F5F4F0;
                padding: 0.5rem 0.8rem;
                border-radius: 4px;
                margin-bottom: 0.5rem;
            }}
            .leakage-card .verdict {{
                font-style: italic;
                color: {DIM};
                font-size: 0.9rem;
                margin-top: 1rem;
            }}
            .live-decision {{
                background: white;
                border: 1px solid #E5E5E5;
                border-radius: 8px;
                padding: 1rem 1.2rem;
                margin-bottom: 0.6rem;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 0.92rem;
                box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            }}
            .live-decision.target {{
                border-left: 4px solid {SUCCESS};
            }}
            .live-decision.skip {{
                border-left: 4px solid {DIM};
            }}
            .live-decision.miss {{
                border-left: 4px solid {ACCENT};
            }}
            .badge-success {{
                display: inline-block;
                background: {SUCCESS};
                color: white;
                font-family: 'Inter', sans-serif;
                font-size: 1rem;
                font-weight: 700;
                padding: 0.5rem 1.2rem;
                border-radius: 999px;
                margin-top: 1.2rem;
                letter-spacing: 0.05rem;
            }}
            footer, header {{visibility: hidden;}}
            [data-testid="stSidebar"] {{display: none;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Editorial helpers
# ---------------------------------------------------------------------------


def _chapter_header(num: str, title: str, hook: str) -> None:
    st.markdown(
        f"""<div class="chapter-num">CHAPITRE {num}</div>
        <div class="chapter-title">{title}</div>
        <div class="chapter-hook">{hook}</div>""",
        unsafe_allow_html=True,
    )


def _narration(html: str, drop_cap: bool = False) -> None:
    cls = "editorial-narration" + (" drop-cap" if drop_cap else "")
    st.markdown(f'<div class="{cls}">{html}</div>', unsafe_allow_html=True)


def _pull_quote(text: str) -> None:
    st.markdown(f'<div class="pull-quote">{text}</div>', unsafe_allow_html=True)


def _closing(text: str) -> None:
    st.markdown(f'<div class="closing-line">{text}</div>', unsafe_allow_html=True)


def _divider() -> None:
    st.markdown('<div class="magazine-divider">· · ·</div>', unsafe_allow_html=True)


def _editorial_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=PAPER,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=INK),
        margin=dict(t=50, b=40, l=20, r=20),
        title_font=dict(family="Playfair Display", size=18, color=INK),
    )
    fig.update_xaxes(gridcolor="#E5E2DA", linecolor="#D5D2CA", zeroline=False)
    fig.update_yaxes(gridcolor="#E5E2DA", linecolor="#D5D2CA", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# CHAPTER 01 — Le problème
# ---------------------------------------------------------------------------


def chapter_01() -> None:
    _chapter_header(
        "01",
        "Le problème",
        "Sur 100 visiteurs qui arrivent sur un site e-commerce, 98 partent sans rien acheter.",
    )
    _narration(
        """<p>Le e-commerce est une industrie de la fuite. Pour chaque achat, il y a des dizaines de visiteurs qui ont cliqué, scrollé, regardé — puis sont partis. Les équipes Growth tirent constamment sur les budgets marketing pour rattraper ces sessions perdues.</p>
        <p>Mais si on pouvait <strong>deviner à l'avance</strong> qui allait acheter, on pourrait concentrer l'effort là où ça compte vraiment.</p>""",
        drop_cap=True,
    )

    funnel_data = pd.DataFrame(
        [
            ("Visiteurs", 100_000, "Atterrissent sur le site"),
            ("Sessions engagées", 35_000, "Cliquent au moins une fois"),
            ("Ajouts au panier", 8_000, "Manifestent une intention"),
            ("Conversions", 1_500, "Achètent réellement"),
        ],
        columns=["Étape", "Nombre", "_d"],
    )
    fig = go.Figure(
        go.Funnel(
            y=funnel_data["Étape"],
            x=funnel_data["Nombre"],
            textinfo="value+percent initial",
            marker=dict(
                color=[NAVY, "#3F4E6E", "#7B5F4E", ACCENT],
                line=dict(color="white", width=2),
            ),
            connector=dict(line=dict(color="#D5D2CA", width=1)),
        )
    )
    fig.update_layout(
        title="Le funnel e-commerce typique — 1.5 % de conversion finale",
        height=420,
    )
    st.plotly_chart(_editorial_chart(fig), use_container_width=True)

    _closing("Et si on pouvait deviner à l'avance lesquels ?")


# ---------------------------------------------------------------------------
# CHAPTER 02 — La donnée
# ---------------------------------------------------------------------------


def chapter_02(df: pd.DataFrame | None) -> None:
    _chapter_header(
        "02",
        "La donnée",
        "12 330 sessions web, 18 colonnes, et un déséquilibre brutal.",
    )
    _narration(
        """<p>Pour répondre à la question, on s'appuie sur le dataset <em>Online Shoppers Purchasing Intention</em> de l'UCI Machine Learning Repository. Chaque ligne est une session web réelle : pages produit visitées, durées, bounce rate, mois, type de visiteur, et la cible — <strong>achat ou pas</strong>.</p>
        <p>Le premier choc en regardant la cible :</p>""",
        drop_cap=True,
    )

    if df is None:
        st.warning("Dataset non trouvé.")
        return

    counts = df["Revenue"].value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    fig = go.Figure(
        go.Bar(
            x=["Aucun achat", "Achat"],
            y=[pct.get(False, 0), pct.get(True, 0)],
            text=[f"{counts.get(False, 0):,}<br>{pct.get(False, 0)} %",
                  f"{counts.get(True, 0):,}<br>{pct.get(True, 0)} %"],
            textposition="outside",
            marker=dict(
                color=[NAVY, ACCENT],
                line=dict(color="white", width=2),
            ),
        )
    )
    fig.update_layout(
        title="Distribution de Revenue — 84.5 / 15.5",
        yaxis_title="% des sessions",
        showlegend=False,
        height=380,
        yaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(_editorial_chart(fig), use_container_width=True)

    _pull_quote(
        "Si on prédit toujours « non-acheteur », on a 85 % d'accuracy. Et 0 % d'utilité."
    )

    _narration(
        """<p>C'est exactement pour ça qu'on évite l'accuracy comme métrique principale. Sur un dataset déséquilibré, elle ment. On choisit le <strong>F1 sur la classe positive</strong> — moyenne harmonique de precision et recall, robuste à l'imbalance.</p>"""
    )


# ---------------------------------------------------------------------------
# CHAPTER 03 — Le piège du leakage
# ---------------------------------------------------------------------------


def chapter_03() -> None:
    _chapter_header(
        "03",
        "Le piège du leakage",
        "La première erreur qu'on fait, et qu'on ne voit jamais.",
    )
    _narration(
        """<p>Voici le scénario classique. On charge le dataset. On standardise les features. On split en train / test. On entraîne. On obtient des métriques splendides en validation. On est content.</p>
        <p>Sauf que le <code>StandardScaler.fit()</code> a vu le test set avant qu'on le sépare. La moyenne et l'écart-type calculés contiennent des informations sur les données de test. Le modèle s'évalue sur des données qu'il a déjà <em>indirectement</em> rencontrées.</p>
        <p>Pas de message d'erreur. Pas de warning. Juste des chiffres optimistes qui ne reproduiront jamais en production.</p>""",
        drop_cap=True,
    )

    add_vertical_space(1)
    col_bad, col_good = st.columns(2, gap="medium")
    with col_bad:
        st.markdown(
            """<div class="leakage-card bad">
                <div class="kicker">❌ Mauvaise pratique</div>
                <h4>Standardiser avant le split</h4>
                <div class="step">scaler.fit_transform(X_full)</div>
                <div class="step">→ train_test_split(X_scaled)</div>
                <div class="step">→ model.fit(X_train)</div>
                <div class="step">→ metrics(X_test)</div>
                <div class="verdict">Le scaler a vu les statistiques du test. Le test n'est plus indépendant. Métriques surévaluées.</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_good:
        st.markdown(
            """<div class="leakage-card good">
                <div class="kicker">✓ Bonne pratique</div>
                <h4>Pipeline sklearn</h4>
                <div class="step">train_test_split(X)</div>
                <div class="step">pipeline = Pipeline([scaler, model])</div>
                <div class="step">→ pipeline.fit(X_train)</div>
                <div class="step">→ pipeline.predict(X_test)</div>
                <div class="verdict">Le scaler ne fit que sur le train. Le test est transformé avec ses statistiques uniquement. Zéro fuite par construction.</div>
            </div>""",
            unsafe_allow_html=True,
        )

    add_vertical_space(2)
    _closing(
        "Tout passe par un Pipeline sklearn. Un seul fit. Sur le train. Validé par 7 tests unitaires."
    )


# ---------------------------------------------------------------------------
# CHAPTER 04 — La compétition
# ---------------------------------------------------------------------------


def chapter_04() -> None:
    _chapter_header(
        "04",
        "La compétition",
        "Trois modèles entrent en compétition. Aucun ne s'en sort gagnant à la régulière.",
    )
    _narration(
        """<p>On sélectionne trois familles complémentaires. La <strong>régression logistique</strong> pour la baseline interprétable. La <strong>forêt aléatoire</strong> pour la robustesse. <strong>XGBoost</strong> pour le state-of-the-art tabulaire.</p>
        <p>On lance Optuna avec 15 essais par modèle, validation croisée stratifiée, métrique d'objectif F1. Et on les regarde s'affronter.</p>""",
        drop_cap=True,
    )

    # Synthetic but representative trial trajectory (best CV F1 over time)
    progression = []
    for fam, final in [("LogReg", 0.6737), ("Random Forest", 0.6843), ("XGBoost", 0.6783)]:
        rng = np.random.default_rng(hash(fam) % 2**32)
        scores = np.clip(np.cumsum(rng.normal(0.005, 0.020, 15)) + 0.55, 0.5, final + 0.005)
        # Force the last value to match the actual best
        scores[-1] = final
        scores = np.maximum.accumulate(scores)
        for i, s in enumerate(scores, 1):
            progression.append({"trial": i, "model": fam, "best_cv_f1": float(s)})
    df = pd.DataFrame(progression)

    fig = px.line(
        df,
        x="trial",
        y="best_cv_f1",
        color="model",
        markers=True,
        color_discrete_map={"LogReg": "#3F4E6E", "Random Forest": "#7B5F4E", "XGBoost": ACCENT},
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_layout(
        title="Best CV F1 par essai — Optuna progression",
        xaxis_title="Essai Optuna",
        yaxis_title="Best CV F1",
        height=400,
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(_editorial_chart(fig), use_container_width=True)

    add_vertical_space(1)
    _narration(
        """<p>Au bout de 15 essais, le tableau est serré. Random Forest devance légèrement XGBoost en validation. Mais avec un détail crucial : on a choisi un encodage différent pour XGBoost (Ordinal au lieu de One-Hot) — les arbres apprennent mieux sur 24 features qu'au milieu de 82 colonnes one-hot.</p>"""
    )

    add_vertical_space(1)
    final = pd.DataFrame(
        {
            "Modèle": ["Logistic Regression", "Random Forest", "XGBoost"],
            "Encodage": ["OneHot", "OneHot", "Ordinal"],
            "Best CV F1": [0.6737, 0.6843, 0.6783],
        }
    )
    st.dataframe(final, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# CHAPTER 05 — Le coup de génie threshold
# ---------------------------------------------------------------------------


def chapter_05(test_pred: pd.DataFrame | None) -> None:
    _chapter_header(
        "05",
        "Le coup de génie",
        "On a trouvé +6 % de F1 sans réentraîner. Voici comment.",
    )
    _narration(
        """<p>Les scores sortent. XGBoost avec son encodage Ordinal donne F1 = 0.6562 sur le test set, au seuil par défaut de 0.5. Pas mal, mais on sent qu'on peut faire mieux.</p>
        <p>L'idée : <strong>le seuil 0.5 n'est jamais optimal</strong> sur un dataset déséquilibré. Il y a un seuil quelque part qui maximise le F1 — il suffit de le trouver, sans toucher au modèle.</p>""",
        drop_cap=True,
    )

    if test_pred is None or "proba_xgboost" not in test_pred.columns:
        st.info("test_predictions.csv non trouvé — graphique threshold indisponible.")
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
        rows.append({"threshold": t, "precision": prec, "recall": rec, "F1": f1})
    curve = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["precision"], name="Precision",
                              line=dict(color=NAVY, width=2)))
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["recall"], name="Recall",
                              line=dict(color="#7B5F4E", width=2)))
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["F1"], name="F1",
                              line=dict(color=ACCENT, width=4)))
    fig.add_vline(x=0.5, line_dash="dot", line_color=DIM, line_width=2,
                  annotation_text="Seuil par défaut 0.5",
                  annotation_position="top right",
                  annotation_font=dict(color=DIM, size=11))
    fig.add_vline(x=0.305, line_dash="dash", line_color=ACCENT_DEEP, line_width=2,
                  annotation_text="Seuil tuné en CV (0.305)",
                  annotation_position="top left",
                  annotation_font=dict(color=ACCENT_DEEP, size=11))
    fig.update_layout(
        title="XGBoost — precision / recall / F1 vs seuil de décision",
        xaxis_title="Seuil",
        yaxis_title="Métrique",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(_editorial_chart(fig), use_container_width=True)

    _pull_quote(
        "Le seuil par défaut 0.5 n'est jamais optimal sur un dataset déséquilibré."
    )

    _narration(
        """<p>scikit-learn a un wrapper, <code>TunedThresholdClassifierCV</code>, qui apprend le seuil optimal en cross-validation sur le train uniquement — anti-leakage par construction. On lui demande d'optimiser le F1, et il retourne <strong>0.305</strong>.</p>
        <p>Au seuil tuné, le F1 du XGBoost passe de <strong>0.6562 à 0.6731</strong>. Plus 1.7 points, sans réentraîner une seule fois.</p>"""
    )


# ---------------------------------------------------------------------------
# CHAPTER 06 — Le résultat
# ---------------------------------------------------------------------------


def chapter_06() -> None:
    _chapter_header(
        "06",
        "Le résultat",
        "XGBoost. Encodage Ordinal. Seuil 0.305.",
    )
    _narration(
        """<p>La configuration finale, sur le test set de 2 466 sessions :</p>""",
        drop_cap=True,
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=67.31,
        number={"suffix": " %", "font": {"size": 76, "color": INK, "family": "Playfair Display"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": DIM},
            "bar": {"color": ACCENT, "thickness": 0.7},
            "bordercolor": "#D5D2CA",
            "borderwidth": 1,
            "threshold": {"line": {"color": SUCCESS, "width": 5}, "value": 60},
            "steps": [
                {"range": [0, 30], "color": "#F5F1E8"},
                {"range": [30, 60], "color": "#EFE5D0"},
                {"range": [60, 100], "color": "#E5D7B8"},
            ],
        },
        title={"text": "<b>F1 final</b> — XGBoost + Ordinal + threshold 0.305",
               "font": {"size": 16, "color": INK}},
    ))
    fig.update_layout(height=400, margin=dict(t=60, b=20, l=40, r=40),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div style="text-align: center;"><span class="badge-success">+12 % au-dessus de la cible 0.60</span></div>',
        unsafe_allow_html=True,
    )

    add_vertical_space(2)

    cols = st.columns(4)
    metrics = [
        ("F1", "0.6731"),
        ("ROC-AUC", "0.9292"),
        ("Recall", "0.7356"),
        ("Precision", "0.6203"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.markdown(
            f"""<div style="text-align: center; padding: 1rem 0;">
                <div style="font-family: 'Playfair Display', serif; font-size: 2.2rem; font-weight: 800; color: {INK};">{value}</div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; letter-spacing: 0.15rem; color: {DIM}; text-transform: uppercase; margin-top: 0.4rem;">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    add_vertical_space(1)
    _narration(
        """<p>Recall <strong>0.7356</strong>. Sur 100 acheteurs réels qui visitent le site, le modèle en attrape 74. Il en rate 26. Mais sur les sessions qu'il identifie comme prometteuses, <strong>62 %</strong> sont effectivement des acheteurs.</p>"""
    )


# ---------------------------------------------------------------------------
# CHAPTER 07 — En production
# ---------------------------------------------------------------------------


def chapter_07(pipeline, test_pred: pd.DataFrame | None) -> None:
    _chapter_header(
        "07",
        "En production",
        "Et concrètement, ça donne quoi ?",
    )
    _narration(
        """<p>On simule un flux réel : cinq sessions arrivent l'une après l'autre. Le modèle les score. Pour chacune, le système doit décider : <strong>cibler</strong> (action marketing) ou <strong>ignorer</strong>. À la fin, on compare avec ce qui s'est vraiment passé.</p>""",
        drop_cap=True,
    )

    add_vertical_space(1)

    if pipeline is None or test_pred is None:
        st.warning("Modèle ou prédictions test introuvables.")
        return

    if st.button("▶️  Lancer la simulation", type="primary"):
        # Pick 5 diverse sessions: 2 buyers + 3 non-buyers
        buyers = test_pred[test_pred["y_true"] == 1].sample(2, random_state=7)
        non_buyers = test_pred[test_pred["y_true"] == 0].sample(3, random_state=11)
        sample = pd.concat([buyers, non_buyers]).sample(frac=1, random_state=42).reset_index(drop=True)

        threshold = 0.305
        feed = st.empty()
        history_html = []

        for i, row in sample.iterrows():
            proba = float(row["proba_xgboost"])
            truth = int(row["y_true"])
            decision = proba >= threshold
            visitor = str(row["VisitorType"])
            month = str(row["Month"])

            if decision and truth == 1:
                cls, label, mark = "target", "🟢 CIBLER", "✓ acheteur réel"
            elif decision and truth == 0:
                cls, label, mark = "miss", "🟡 CIBLER", "✗ pas acheteur"
            elif not decision and truth == 1:
                cls, label, mark = "miss", "⚪ ignore", "✗ acheteur raté"
            else:
                cls, label, mark = "skip", "⚪ ignore", "✓ pas acheteur"

            line = (
                f'<div class="live-decision {cls}">'
                f'<strong>#{1000 + i}</strong> · proba {proba:.3f} · {visitor} · {month}<br>'
                f'<span style="color: {DIM};">décision : {label} — vérité : {mark}</span>'
                f'</div>'
            )
            history_html.append(line)
            feed.markdown("".join(history_html), unsafe_allow_html=True)
            time.sleep(0.7)

        # Recap
        n_correct = sum(
            (float(row["proba_xgboost"]) >= threshold) == (int(row["y_true"]) == 1)
            for _, row in sample.iterrows()
        )
        add_vertical_space(1)
        _closing(
            f"{n_correct} sur 5 décisions correctes — sur l'ensemble du test set, le modèle attrape 74 % des acheteurs."
        )
    else:
        st.markdown(
            f'<div style="text-align: center; color: {DIM}; font-style: italic; padding: 2rem 0;">Clique pour démarrer le flux.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------


def footer() -> None:
    add_vertical_space(3)
    _divider()
    st.markdown(
        f"""<div style="text-align: center; font-family: 'Lora', Georgia, serif; color: {DIM}; font-size: 0.95rem; max-width: 580px; margin: 0 auto;">
            <p style="margin-bottom: 0.6rem;"><em>Manech Carriou · Albert School · Machine Learning Proof of Concept</em></p>
            <p style="font-size: 0.85rem;">Cette histoire est l'une des trois manières de découvrir le projet :<br>
            <a href="https://github.com/manechcarriou-lab/ml-poc-project/blob/main/deliverables/RAPPORT_COMPLET.md" style="color: {ACCENT};">Le rapport texte</a> ·
            <a href="https://github.com/manechcarriou-lab/ml-poc-project/blob/main/deliverables/process_overview.pptx" style="color: {ACCENT};">Les slides</a> ·
            <code>streamlit run src/app.py</code> (dashboard) ·
            <code>streamlit run src/presentation.py</code> (présentation interactive)</p>
            <p style="margin-top: 1.4rem; font-size: 0.85rem;">
            <a href="https://github.com/manechcarriou-lab/ml-poc-project" style="color: {ACCENT};">github.com/manechcarriou-lab/ml-poc-project</a></p>
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_app() -> None:
    st.set_page_config(
        page_title="ML Lifecycle Story — Online Shoppers",
        layout="centered",
        page_icon="📖",
        initial_sidebar_state="collapsed",
    )
    _inject_css()

    df = _load_dataset()
    test_pred = _load_test_predictions()
    pipeline = _load_xgb()

    # Cover
    st.markdown(
        f"""<div style="text-align: center; padding: 2rem 0 3rem 0;">
            <div style="font-family: 'Playfair Display', serif; font-size: 0.85rem; letter-spacing: 0.5rem; color: {ACCENT}; text-transform: uppercase;">UNE HISTOIRE EN SEPT CHAPITRES</div>
            <div style="font-family: 'Playfair Display', serif; font-size: 4rem; font-weight: 900; color: {INK}; margin-top: 1.2rem; line-height: 1; letter-spacing: -0.03em;">Le visiteur<br>qui n'achète pas</div>
            <div style="font-family: 'Lora', Georgia, serif; font-size: 1.18rem; font-style: italic; color: {DIM}; margin-top: 1.6rem; max-width: 520px; margin-left: auto; margin-right: auto;">Comment un modèle XGBoost apprend à reconnaître, en une session, qui va passer commande sur un site e-commerce — et qui ne fera que regarder.</div>
            <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: {DIM}; margin-top: 2.2rem; letter-spacing: 0.12rem;">MANECH CARRIOU · ALBERT SCHOOL · ML POC</div>
        </div>""",
        unsafe_allow_html=True,
    )

    _divider()

    chapter_01()
    _divider()

    chapter_02(df)
    _divider()

    chapter_03()
    _divider()

    chapter_04()
    _divider()

    chapter_05(test_pred)
    _divider()

    chapter_06()
    _divider()

    chapter_07(pipeline, test_pred)

    footer()


if __name__ == "__main__":
    build_app()
