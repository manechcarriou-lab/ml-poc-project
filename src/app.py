"""ShopSignal — Conversion intelligence dashboard.

Editorial-meets-terminal Streamlit app. Three-part dashboard for the
ML PoC of e-commerce conversion prediction:

  PART 01 · Problem & EDA — context, dataset, the imbalance.
  PART 02 · Models & metrics — three-family benchmark + rigorous validation.
  PART 03 · Live demo — score a session, see the decision, simulate ROI.

Design system: ShopSignal (Fraunces × Inter × JetBrains Mono, paper #F4F1EA,
ink #16140F, Signal Orange #FF5B1F). No emoji in chrome, sharp corners,
1px hairlines. See `design system handoff` for tokens.

Launch:
    streamlit run src/app.py
or: python scripts/main.py
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


# ---------------------------------------------------------------------------
# ShopSignal palette (mirrors colors_and_type.css from the design system)
# ---------------------------------------------------------------------------

PAPER = "#F4F1EA"
PAPER_2 = "#ECE8DF"
PAPER_3 = "#E2DDD0"
INK = "#16140F"
INK_2 = "#2B2820"
INK_3 = "#5C5849"
INK_4 = "#908A78"
SIGNAL = "#FF5B1F"
SIGNAL_DEEP = "#C8410E"
SIGNAL_SOFT = "#FFE3D4"
SIGNAL_TINT = "#FFF1E7"
DATA_INK = INK
DATA_SIGNAL = SIGNAL
DATA_MOSS = "#2C5F4D"
DATA_SAND = "#B89968"
DATA_SLATE = "#6B7280"
POSITIVE = "#2C5F4D"
NEGATIVE = "#A82E14"
SURFACE = "#FBFAF6"
SURFACE_INK = "#1A1813"
RULE_HAIR = "rgba(22, 20, 15, 0.12)"


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
# CSS — full ShopSignal design system
# ---------------------------------------------------------------------------


def _inject_css() -> None:
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --paper: #F4F1EA;
            --paper-2: #ECE8DF;
            --paper-3: #E2DDD0;
            --ink: #16140F;
            --ink-2: #2B2820;
            --ink-3: #5C5849;
            --ink-4: #908A78;
            --signal: #FF5B1F;
            --signal-deep: #C8410E;
            --signal-soft: #FFE3D4;
            --signal-tint: #FFF1E7;
            --positive: #2C5F4D;
            --negative: #A82E14;
            --surface: #FBFAF6;
            --surface-ink: #1A1813;
            --rule-hair: rgba(22, 20, 15, 0.12);
            --rule-soft: #C9C3B3;
            --serif: 'Fraunces', 'Times New Roman', Georgia, serif;
            --sans: 'Inter', -apple-system, system-ui, sans-serif;
            --mono: 'JetBrains Mono', 'SF Mono', Menlo, monospace;
        }

        .stApp {
            background: var(--paper);
            font-family: var(--sans);
            color: var(--ink);
        }
        .stApp, .stApp * {
            font-family: var(--sans);
        }

        /* Sticky brand header */
        .ss-header {
            position: sticky; top: 0; z-index: 100;
            background: rgba(244, 241, 234, 0.92);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--rule-hair);
            margin: -1rem -1rem 0 -1rem;
            padding: 14px 32px;
            display: flex; align-items: center; gap: 24px;
        }
        .ss-header .lock {
            display: flex; align-items: center; gap: 12px;
            text-decoration: none;
        }
        .ss-header .word {
            font-family: var(--serif); font-weight: 500;
            font-size: 22px; letter-spacing: -0.02em; color: var(--ink);
        }
        .ss-header .tag {
            font-family: var(--mono); font-size: 10px;
            letter-spacing: 0.12em; text-transform: uppercase;
            color: var(--ink-3);
            border-left: 1px solid var(--rule-hair);
            padding-left: 12px;
        }
        .ss-header .pill {
            margin-left: auto;
            font-family: var(--mono); font-size: 10px; font-weight: 600;
            letter-spacing: 0.12em; text-transform: uppercase;
            padding: 5px 12px;
            border: 1px solid var(--ink); color: var(--ink);
            display: inline-flex; align-items: center; gap: 8px;
        }
        .ss-header .pill .dot {
            width: 6px; height: 6px; background: var(--signal);
            border-radius: 999px;
            animation: pulse 2.4s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* Editorial hero */
        .ss-hero {
            padding: 56px 0 32px 0;
            border-bottom: 1px solid var(--ink);
            margin-bottom: 32px;
        }
        .ss-hero .eyebrow-row {
            display: flex; align-items: center; gap: 12px;
            font-family: var(--mono); font-size: 11px; font-weight: 500;
            letter-spacing: 0.18em; text-transform: uppercase;
            color: var(--ink-3); margin-bottom: 24px;
        }
        .ss-hero .eyebrow-row .num { color: var(--signal); font-weight: 700; }
        .ss-hero .eyebrow-row .dash { width: 32px; height: 1px; background: var(--ink-3); }
        .ss-hero h1 {
            font-family: var(--serif); font-weight: 500;
            font-size: clamp(48px, 6vw, 84px);
            line-height: 0.98;
            letter-spacing: -0.035em;
            margin: 0 0 20px;
            font-variation-settings: "opsz" 144;
            color: var(--ink);
        }
        .ss-hero h1 em {
            font-style: italic; color: var(--signal);
            font-weight: 500;
        }
        .ss-hero .lead {
            font-family: var(--serif); font-size: 20px;
            line-height: 1.5; color: var(--ink-2);
            max-width: 760px;
            font-variation-settings: "opsz" 24;
        }
        .ss-hero .lead strong { color: var(--ink); font-weight: 600; }

        /* KPI strip — 4 cells in ink border */
        .ss-kpi-strip {
            display: grid; grid-template-columns: repeat(4, 1fr);
            border: 1px solid var(--ink); background: var(--surface);
            margin: 32px 0 0 0;
        }
        .ss-kpi {
            padding: 20px 24px;
            border-right: 1px solid var(--rule-hair);
        }
        .ss-kpi:last-child { border-right: 0; }
        .ss-kpi .eb {
            font-family: var(--mono); font-size: 10px; font-weight: 500;
            letter-spacing: 0.18em; text-transform: uppercase;
            color: var(--ink-3); display: flex; align-items: center; gap: 8px;
            margin-bottom: 14px;
        }
        .ss-kpi .eb::before {
            content: ""; width: 5px; height: 5px; background: var(--signal);
            display: inline-block;
        }
        .ss-kpi .num {
            font-family: var(--mono); font-variant-numeric: tabular-nums;
            font-size: 36px; font-weight: 500; line-height: 1;
            letter-spacing: -0.01em; color: var(--ink);
        }
        .ss-kpi .num .u {
            color: var(--ink-3); font-size: 22px; margin-left: 2px;
        }
        .ss-kpi .delta {
            font-family: var(--sans); font-size: 12px;
            color: var(--ink-3); margin-top: 10px;
        }
        .ss-kpi .delta .pos { color: var(--positive); font-weight: 700; }

        /* Metabar — horizontal data strip below hero */
        .ss-metabar {
            display: flex; flex-wrap: wrap; gap: 28px; padding: 14px 0;
            border-top: 1px solid var(--ink); border-bottom: 1px solid var(--ink);
            margin: 24px 0 8px;
        }
        .ss-metabar .item { display: flex; flex-direction: column; }
        .ss-metabar .lab {
            font-family: var(--mono); font-size: 10px;
            letter-spacing: 0.12em; text-transform: uppercase;
            color: var(--ink-3);
        }
        .ss-metabar .val {
            font-family: var(--mono); font-variant-numeric: tabular-nums;
            font-size: 14px; font-weight: 600; color: var(--ink); margin-top: 2px;
        }

        /* Numbered Block (section header) */
        .ss-block-head {
            display: grid; grid-template-columns: 64px 1fr;
            gap: 24px; align-items: baseline;
            border-top: 1px solid var(--ink); padding-top: 28px;
            margin-top: 56px; margin-bottom: 24px;
        }
        .ss-block-head .num {
            font-family: var(--mono); font-variant-numeric: tabular-nums;
            font-size: 26px; font-weight: 400; color: var(--ink-3);
            letter-spacing: -0.01em;
        }
        .ss-block-head .eb {
            font-family: var(--mono); font-size: 11px; font-weight: 500;
            letter-spacing: 0.18em; text-transform: uppercase;
            color: var(--ink-3); margin-bottom: 10px;
        }
        .ss-block-head h2 {
            font-family: var(--serif); font-weight: 500;
            font-size: clamp(30px, 3.5vw, 42px);
            line-height: 1.1; letter-spacing: -0.018em;
            margin: 0; max-width: 900px;
            font-variation-settings: "opsz" 96;
            color: var(--ink);
        }
        .ss-block-head h2 em {
            font-style: italic; color: var(--signal); font-weight: 500;
        }
        .ss-block-body { padding-left: 0; }

        /* Card — 1px ink border, optional letterpress */
        .ss-card {
            background: var(--surface); border: 1px solid var(--ink);
            padding: 22px 24px; margin: 12px 0;
        }
        .ss-card.print { box-shadow: 2px 2px 0 var(--ink); }
        .ss-card.dark {
            background: var(--surface-ink); color: var(--paper);
            border-color: var(--surface-ink);
        }
        .ss-card.signal-tint {
            background: var(--signal-tint);
            border-left: 4px solid var(--signal);
        }
        .ss-card .eb {
            font-family: var(--mono); font-size: 10px; font-weight: 500;
            letter-spacing: 0.18em; text-transform: uppercase;
            color: var(--signal); display: flex; align-items: center; gap: 8px;
            margin-bottom: 10px;
        }
        .ss-card .eb::before { content: "◆"; }
        .ss-card.dark .eb { color: var(--signal); }
        .ss-card.positive .eb { color: var(--positive); }
        .ss-card.negative .eb { color: var(--negative); }
        .ss-card .ti {
            font-family: var(--serif); font-weight: 600; font-size: 22px;
            letter-spacing: -0.01em; color: var(--ink); margin-bottom: 8px;
            line-height: 1.25;
        }
        .ss-card.dark .ti { color: var(--paper); }
        .ss-card .bo {
            font-family: var(--sans); font-size: 14px;
            color: var(--ink-2); line-height: 1.6;
        }
        .ss-card.dark .bo { color: rgba(244, 241, 234, 0.85); }
        .ss-card .bo strong { color: var(--ink); font-weight: 600; }
        .ss-card.dark .bo strong { color: var(--paper); }
        .ss-card .bo code {
            font-family: var(--mono); font-size: 12px;
            background: var(--paper-2); padding: 1px 5px;
            color: var(--ink);
        }
        .ss-card.dark .bo code {
            background: rgba(255, 255, 255, 0.08); color: var(--paper);
        }

        /* Streamlit content overrides */
        h1 {
            font-family: var(--serif) !important;
            font-weight: 500 !important;
            color: var(--ink);
            letter-spacing: -0.025em;
            font-variation-settings: "opsz" 144;
        }
        h2, h3 {
            font-family: var(--serif) !important;
            font-weight: 500 !important;
            color: var(--ink);
            letter-spacing: -0.018em;
        }
        h4, h5, h6 {
            font-family: var(--sans) !important;
            font-weight: 600 !important;
            color: var(--ink);
        }
        p, li, label, span, div {
            color: var(--ink-2);
        }
        strong { color: var(--ink); font-weight: 600; }
        code, kbd, pre {
            font-family: var(--mono) !important;
            background: var(--paper-2) !important;
            color: var(--ink) !important;
            padding: 1px 6px !important;
            border-radius: 0 !important;
            font-size: 0.85em !important;
        }
        hr {
            border: 0;
            border-top: 1px solid var(--rule-hair) !important;
            margin: 24px 0 !important;
        }

        /* Tabs — sharp + signal accent */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            border-bottom: 1px solid var(--ink) !important;
            background: var(--paper);
        }
        .stTabs [data-baseweb="tab-list"] button {
            font-family: var(--sans); font-weight: 500; font-size: 14px;
            color: var(--ink-3) !important;
            padding: 14px 22px 16px !important;
            border-radius: 0 !important;
            border: 1px solid transparent !important;
            border-bottom: none !important;
            background: transparent !important;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: var(--ink) !important;
            background: var(--surface) !important;
            border-color: var(--ink) !important;
            position: relative; top: 1px;
        }
        .stTabs [data-baseweb="tab-list"] button:hover:not([aria-selected="true"]) {
            color: var(--ink) !important;
        }

        /* Buttons */
        div.stButton > button, .stDownloadButton > button {
            font-family: var(--sans); font-weight: 600; font-size: 13px;
            padding: 10px 18px; border-radius: 0;
            border: 1px solid var(--ink); background: var(--ink);
            color: var(--paper); letter-spacing: 0.01em;
            transition: background 0.12s ease, transform 0.12s ease;
            box-shadow: none;
        }
        div.stButton > button:hover {
            background: var(--signal); border-color: var(--signal);
            color: white;
        }
        div.stButton > button:active { transform: translateY(1px); }

        /* Sliders — sharp 16px square thumb */
        div[data-testid="stSlider"] {
            padding: 8px 0 !important;
        }
        div[data-testid="stSlider"] [role="slider"] {
            background: var(--ink) !important;
            border: 2px solid var(--paper) !important;
            border-radius: 0 !important;
            width: 16px !important; height: 16px !important;
            box-shadow: 0 0 0 1px var(--ink) !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
            background: var(--ink) !important;
            height: 2px !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] > div {
            background: var(--rule-hair) !important;
            height: 2px !important;
        }

        /* Selectbox + text input */
        div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
            border: 1px solid var(--ink) !important;
            border-radius: 0 !important;
            background: var(--surface) !important;
            font-family: var(--sans) !important;
            color: var(--ink) !important;
            box-shadow: none !important;
        }
        div[data-baseweb="select"] > div:focus-within,
        .stTextInput input:focus,
        .stNumberInput input:focus {
            border-color: var(--signal) !important;
            box-shadow: inset 0 0 0 1px var(--signal) !important;
        }

        /* Checkbox */
        .stCheckbox label {
            font-family: var(--sans);
            color: var(--ink);
        }

        /* Sidebar — surface-ink dark */
        section[data-testid="stSidebar"] {
            background: var(--surface-ink) !important;
            border-right: 1px solid var(--ink);
        }
        section[data-testid="stSidebar"] * {
            color: var(--paper) !important;
            font-family: var(--sans) !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-family: var(--serif) !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a {
            color: var(--signal) !important;
        }

        /* Data tables */
        .stDataFrame {
            border: 1px solid var(--ink);
        }
        .stDataFrame [data-testid="StyledDataFrameRowHeaderCell"],
        .stDataFrame [data-testid="StyledDataFrameDataCell"] {
            font-family: var(--mono) !important;
            font-variant-numeric: tabular-nums;
        }

        /* Custom data table — used for the model comparison */
        table.ss-dt {
            width: 100%; border-collapse: collapse;
            background: var(--surface); border: 1px solid var(--ink);
            margin: 16px 0;
        }
        table.ss-dt th, table.ss-dt td {
            font-family: var(--mono); font-variant-numeric: tabular-nums;
            font-size: 13px; padding: 12px 18px;
            border-bottom: 1px solid var(--rule-hair); text-align: right;
        }
        table.ss-dt th {
            font-weight: 600; color: var(--ink-3); font-size: 10px;
            letter-spacing: 0.15em; text-transform: uppercase;
            border-bottom: 1px solid var(--ink);
            background: var(--ink); color: var(--paper);
        }
        table.ss-dt th:first-child, table.ss-dt td:first-child { text-align: left; }
        table.ss-dt td:first-child {
            font-family: var(--sans); font-weight: 500; color: var(--ink);
            font-size: 14px;
        }
        table.ss-dt tr.win td {
            background: var(--signal-tint);
        }
        table.ss-dt tr.win td:first-child {
            color: var(--signal-deep); font-weight: 700;
        }
        table.ss-dt tr.win td.f1-cell {
            color: var(--signal); font-weight: 700;
        }
        table.ss-dt .win-tag {
            font-family: var(--mono); font-size: 9px; font-weight: 700;
            background: var(--signal); color: var(--paper);
            padding: 3px 7px; letter-spacing: 0.1em; margin-right: 8px;
            display: inline-block;
        }

        /* Chips */
        .ss-chip {
            font-family: var(--mono); font-size: 10px; font-weight: 600;
            letter-spacing: 0.08em; text-transform: uppercase;
            padding: 4px 10px; border: 1px solid var(--ink); color: var(--ink);
            background: transparent; display: inline-flex; align-items: center; gap: 6px;
        }
        .ss-chip.signal { border-color: var(--signal); color: var(--signal); }
        .ss-chip.fill { background: var(--ink); color: var(--paper); }
        .ss-chip.signal.fill { background: var(--signal); color: white; border-color: var(--signal); }

        /* Footer */
        .ss-footer {
            background: var(--surface-ink); color: var(--paper);
            padding: 56px 32px 40px;
            margin: 96px -1rem -1rem -1rem;
        }
        .ss-footer .top {
            display: grid; grid-template-columns: 2fr 1fr 1fr 1fr;
            gap: 32px; margin-bottom: 48px;
            max-width: 1280px; margin-left: auto; margin-right: auto;
        }
        .ss-footer .word {
            font-family: var(--serif); font-weight: 500; font-size: 28px;
            letter-spacing: -0.02em; color: var(--paper); margin-bottom: 8px;
        }
        .ss-footer .desc {
            font-family: var(--serif); font-size: 16px;
            color: rgba(244, 241, 234, 0.7); max-width: 380px; line-height: 1.5;
        }
        .ss-footer h5 {
            font-family: var(--mono) !important; font-size: 10px !important;
            letter-spacing: 0.18em; text-transform: uppercase;
            color: rgba(244, 241, 234, 0.6) !important;
            margin: 0 0 14px !important; font-weight: 500 !important;
        }
        .ss-footer ul { list-style: none; padding: 0; margin: 0; }
        .ss-footer li {
            font-family: var(--sans); font-size: 13px;
            color: rgba(244, 241, 234, 0.85); margin-bottom: 8px;
        }
        .ss-footer .bottom {
            max-width: 1280px; margin: 0 auto;
            display: flex; justify-content: space-between; align-items: center;
            padding-top: 24px; border-top: 1px solid rgba(244, 241, 234, 0.12);
            font-family: var(--mono); font-size: 11px;
            color: rgba(244, 241, 234, 0.55); letter-spacing: 0.05em;
        }

        /* Hide Streamlit chrome */
        #MainMenu, footer.css-1lsmgbg, header[data-testid="stHeader"] {
            visibility: hidden;
            height: 0;
        }
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0 !important;
            max-width: 1280px !important;
        }

        /* Selection */
        ::selection {
            background: var(--signal); color: var(--paper);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Building blocks (custom HTML, no shadcn — strict ShopSignal)
# ---------------------------------------------------------------------------


def _shopsignal_mark_svg(size: int = 28) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 64 64" aria-hidden="true">'
        f'<path d="M14 18 L50 18 L32 46 Z" fill="none" stroke="#16140F" stroke-width="3" stroke-linejoin="miter"/>'
        f'<line x1="14" y1="50" x2="50" y2="50" stroke="#16140F" stroke-width="3"/>'
        f'<circle cx="32" cy="28" r="2.5" fill="#FF5B1F"/>'
        f'</svg>'
    )


def _header() -> None:
    html = (
        '<div class="ss-header">'
        '<div class="lock">'
        f'{_shopsignal_mark_svg(28)}'
        '<span class="word">ShopSignal</span>'
        '<span class="tag">Conversion intelligence</span>'
        '</div>'
        '<span class="pill"><span class="dot"></span>Live · v1.0</span>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _hero(part_label: str, headline_html: str, lead_html: str,
          kpis: list[tuple[str, str, str]],
          metabar_items: list[tuple[str, str]]) -> None:
    """Editorial hero with eyebrow row, h1, lead, KPI strip, metabar."""
    eyebrow_row = (
        '<div class="eyebrow-row">'
        f'<span class="num">{part_label}</span>'
        '<span class="dash"></span>'
        '<span>The pitch · 30 seconds</span>'
        '</div>'
    )
    kpi_html = '<div class="ss-kpi-strip">'
    for eb, num, delta in kpis:
        kpi_html += (
            '<div class="ss-kpi">'
            f'<div class="eb">{eb}</div>'
            f'<div class="num">{num}</div>'
            f'<div class="delta">{delta}</div>'
            '</div>'
        )
    kpi_html += '</div>'
    metabar_html = '<div class="ss-metabar">'
    for lab, val in metabar_items:
        metabar_html += (
            '<div class="item">'
            f'<span class="lab">{lab}</span>'
            f'<span class="val">{val}</span>'
            '</div>'
        )
    metabar_html += '</div>'
    html = (
        '<section class="ss-hero">'
        f'{eyebrow_row}'
        f'<h1>{headline_html}</h1>'
        f'<p class="lead">{lead_html}</p>'
        f'{kpi_html}'
        f'{metabar_html}'
        '</section>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _block_header(num: str, eyebrow: str, title_html: str) -> None:
    html = (
        '<div class="ss-block-head">'
        f'<div class="num">{num}</div>'
        '<div>'
        f'<div class="eb">{eyebrow}</div>'
        f'<h2>{title_html}</h2>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _card(eyebrow: str, title: str, body_html: str,
          kind: str = "default") -> None:
    """Card with optional dark / signal-tint / positive / negative variants."""
    cls = "ss-card"
    if kind == "dark":
        cls += " dark"
    elif kind == "tint":
        cls += " signal-tint"
    elif kind == "print":
        cls += " print"
    elif kind == "positive":
        cls += " positive"
    elif kind == "negative":
        cls += " negative"
    html = (
        f'<div class="{cls}">'
        f'<div class="eb">{eyebrow}</div>'
        f'<div class="ti">{title}</div>'
        f'<div class="bo">{body_html}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _kpi_strip(items: list[tuple[str, str, str]]) -> None:
    """Render an n-column KPI strip with ink border."""
    n = len(items)
    grid = f'grid-template-columns: repeat({n}, 1fr);'
    html = f'<div class="ss-kpi-strip" style="{grid}">'
    for eb, num, delta in items:
        html += (
            '<div class="ss-kpi">'
            f'<div class="eb">{eb}</div>'
            f'<div class="num">{num}</div>'
            f'<div class="delta">{delta}</div>'
            '</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def _data_table(headers: list[str], rows: list[list[str]],
                winner_idx: int | None = None,
                f1_col_idx: int | None = None) -> None:
    """Custom HTML table styled with .ss-dt class."""
    html = '<table class="ss-dt"><thead><tr>'
    for h in headers:
        html += f'<th>{h}</th>'
    html += '</tr></thead><tbody>'
    for i, row in enumerate(rows):
        is_win = (i == winner_idx)
        tr_cls = ' class="win"' if is_win else ''
        html += f'<tr{tr_cls}>'
        for j, cell in enumerate(row):
            td_cls = ''
            if is_win and j == f1_col_idx:
                td_cls = ' class="f1-cell"'
            label = cell
            if is_win and j == 0:
                label = f'<span class="win-tag">WIN</span>{cell}'
            html += f'<td{td_cls}>{label}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)


def _footer() -> None:
    html = (
        '<div class="ss-footer">'
        '<div class="top">'
        '<div>'
        '<div class="word">ShopSignal</div>'
        '<p class="desc">Conversion intelligence for e-commerce growth teams. Score every session, target where it matters, stop spending budget on visitors who already converted.</p>'
        '</div>'
        '<div>'
        '<h5>Product</h5>'
        '<ul>'
        '<li>Dashboard</li><li>API access</li><li>SDK</li><li>Integrations</li>'
        '</ul>'
        '</div>'
        '<div>'
        '<h5>Resources</h5>'
        '<ul>'
        '<li><a href="https://github.com/manechcarriou-lab/ml-poc-project/blob/main/deliverables/RAPPORT_COMPLET.md" style="color:rgba(244,241,234,0.85);text-decoration:none;">Methodology</a></li>'
        '<li><a href="https://github.com/manechcarriou-lab/ml-poc-project" style="color:rgba(244,241,234,0.85);text-decoration:none;">Repository</a></li>'
        '<li>Model card</li><li>Status</li>'
        '</ul>'
        '</div>'
        '<div>'
        '<h5>Methodology</h5>'
        '<ul>'
        '<li>Manech Carriou</li><li>Albert School · 2026</li>'
        '<li>UCI Online Shoppers</li><li>CC BY 4.0</li>'
        '</ul>'
        '</div>'
        '</div>'
        '<div class="bottom">'
        '<span>© 2026 SHOPSIGNAL · ML POC · MANECH CARRIOU</span>'
        '<span>F1 = 0.6731 · ROC-AUC = 0.9292 · n = 2,466</span>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly theming — ShopSignal data palette
# ---------------------------------------------------------------------------


def _style_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=PAPER,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=INK_2, size=12),
        margin=dict(t=44, b=40, l=44, r=24),
        legend=dict(
            bgcolor="rgba(244, 241, 234, 0.95)",
            bordercolor=INK,
            borderwidth=1,
            font=dict(family="JetBrains Mono, monospace", size=11, color=INK),
        ),
        title_font=dict(family="Fraunces, serif", size=16, color=INK),
    )
    fig.update_xaxes(
        gridcolor=RULE_HAIR, linecolor=INK, zeroline=False,
        tickfont=dict(family="JetBrains Mono, monospace", color=INK_3, size=10),
    )
    fig.update_yaxes(
        gridcolor=RULE_HAIR, linecolor=INK, zeroline=False,
        tickfont=dict(family="JetBrains Mono, monospace", color=INK_3, size=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Validation rigoureuse — internal helpers (stay the same logic, restyled)
# ---------------------------------------------------------------------------


def _validation_calibration(test_pred: pd.DataFrame) -> None:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    y_true = test_pred["y_true"].astype(int).values
    y_score = test_pred["proba_xgboost"].values
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true, y_score)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color=INK_3, width=1.5),
        name="Perfect calibration",
    ))
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true, mode="lines+markers",
        line=dict(color=SIGNAL, width=2.5),
        marker=dict(size=10, color=SIGNAL, line=dict(color=PAPER, width=2)),
        name="XGBoost (10 quantile bins)",
    ))
    fig.update_layout(
        title="Reliability diagram — XGBoost on the held-out test set",
        xaxis_title="Predicted probability (mean per bin)",
        yaxis_title="Empirical purchase rate",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(_style_chart(fig), use_container_width=True)

    verdict = (
        "slightly under-confident at the high end"
        if (prob_true - prob_pred).max() > 0.05
        else "broadly well calibrated"
    )

    _kpi_strip([
        ("Brier score", f"{brier:.4f}", "lower is better · max 0.25"),
        ("Verdict", verdict.split()[0].capitalize(), verdict),
    ])
    _card(
        "Reading",
        "What this validates",
        f"Each point is one bin. The closer to the diagonal, the more honest the probabilities. "
        f"<strong>{verdict.capitalize()}.</strong> "
        f"This matters because it justifies the threshold of 0.305 — it corresponds to a real "
        f"~30.5% probability of conversion, not an arbitrary number.",
        kind="default",
    )


def _validation_error_analysis(test_pred: pd.DataFrame) -> None:
    from sklearn.metrics import f1_score, precision_score, recall_score

    df = test_pred.copy()
    threshold = 0.305
    df["y_pred"] = (df["proba_xgboost"] >= threshold).astype(int)

    seg = st.selectbox(
        "Slice by",
        ["VisitorType", "Month", "Weekend", "TrafficType"],
    )

    rows = []
    for modality, group in df.groupby(seg):
        n = len(group)
        if n < 20:
            continue
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
            "modality": str(modality),
            "n": n,
            "n_pos": n_pos,
            "F1": f1,
            "precision": prec,
            "recall": rec,
        })

    seg_df = pd.DataFrame(rows).sort_values("n", ascending=False)
    overall_f1 = f1_score(
        df["y_true"].astype(int), df["y_pred"], zero_division=0
    )

    fig = go.Figure()
    palette = [INK, SIGNAL, DATA_MOSS]
    names = ["Precision", "Recall", "F1"]
    cols = ["precision", "recall", "F1"]
    for col, color, name in zip(cols, palette, names):
        fig.add_trace(go.Bar(
            x=seg_df["modality"], y=seg_df[col],
            name=name,
            marker=dict(color=color, line=dict(color=PAPER, width=1)),
        ))
    fig.add_hline(
        y=overall_f1, line_dash="dash", line_color=INK, line_width=1,
        annotation_text=f"global F1 = {overall_f1:.3f}",
        annotation_position="top right",
        annotation_font=dict(color=INK, family="JetBrains Mono", size=10),
    )
    fig.update_layout(
        title=f"Per-modality performance · {seg}",
        barmode="group",
        height=420,
        yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(_style_chart(fig), use_container_width=True)

    seg_df["gap"] = seg_df["F1"] - overall_f1
    worst = seg_df.nsmallest(3, "gap")
    best = seg_df.nlargest(3, "F1")

    col_w, col_b = st.columns(2)
    with col_w:
        items = "".join(
            f"<li><strong>{r['modality']}</strong> "
            f"<span style='color:var(--ink-3)'>· n={r['n']} · {r['n_pos']} buyers</span><br/>"
            f"<span class='numeric' style='font-family:var(--mono);font-size:14px;color:var(--negative);'>F1 = {r['F1']:.3f}</span> "
            f"<span style='color:var(--ink-3);font-size:12px;'>({r['gap']:+.3f} vs global)</span></li>"
            for _, r in worst.iterrows()
        )
        _card(
            "Underperforming segments",
            "Where the model misses",
            f"<ul style='margin: 0; padding-left: 1.2rem;'>{items}</ul>",
            kind="negative",
        )
    with col_b:
        items = "".join(
            f"<li><strong>{r['modality']}</strong> "
            f"<span style='color:var(--ink-3)'>· n={r['n']} · {r['n_pos']} buyers</span><br/>"
            f"<span class='numeric' style='font-family:var(--mono);font-size:14px;color:var(--positive);'>F1 = {r['F1']:.3f}</span></li>"
            for _, r in best.iterrows()
        )
        _card(
            "Strong segments",
            "Where the model excels",
            f"<ul style='margin: 0; padding-left: 1.2rem;'>{items}</ul>",
            kind="positive",
        )

    with st.expander("Detailed per-modality table"):
        pretty = seg_df.copy()
        for c in ["F1", "precision", "recall"]:
            pretty[c] = pretty[c].map(lambda x: f"{x:.4f}")
        pretty["gap"] = pretty["gap"].map(lambda x: f"{x:+.4f}")
        st.dataframe(pretty, use_container_width=True, hide_index=True)


def _validation_naive_baseline(test_pred: pd.DataFrame) -> None:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
    )

    y_true = test_pred["y_true"].astype(int).values
    rule_always_no = np.zeros_like(y_true)
    rule_pv = (test_pred["PageValues"] > 0).astype(int).values
    high_season = test_pred["Month"].isin(["Nov", "Sep", "Oct"]).values
    rule_business = ((test_pred["PageValues"] > 0).values & high_season).astype(int)
    rule_xgb = (test_pred["proba_xgboost"] >= 0.305).astype(int).values

    rules = [
        ("Always no-buy",                    rule_always_no),
        ("Heuristic · PageValues > 0",       rule_pv),
        ("Heuristic · PageValues + season",  rule_business),
        ("XGBoost · tuned",                  rule_xgb),
    ]
    rows = []
    for name, preds in rules:
        rows.append({
            "Strategy": name,
            "Accuracy":  accuracy_score(y_true, preds),
            "Precision": precision_score(y_true, preds, zero_division=0),
            "Recall":    recall_score(y_true, preds, zero_division=0),
            "F1":        f1_score(y_true, preds, zero_division=0),
            "Targeted":  int(preds.sum()),
        })
    bench = pd.DataFrame(rows)

    headers = ["Strategy", "F1", "Precision", "Recall", "Accuracy", "Targeted"]
    table_rows = []
    f1_max_idx = int(bench["F1"].idxmax())
    for i, r in bench.iterrows():
        table_rows.append([
            r["Strategy"],
            f"{r['F1']:.4f}",
            f"{r['Precision']:.4f}",
            f"{r['Recall']:.4f}",
            f"{r['Accuracy']:.4f}",
            f"{int(r['Targeted']):,}",
        ])
    _data_table(headers, table_rows, winner_idx=f1_max_idx, f1_col_idx=1)

    fig = go.Figure(go.Bar(
        x=bench["F1"], y=bench["Strategy"],
        orientation="h",
        marker=dict(
            color=[INK_4, INK, INK_3, SIGNAL],
            line=dict(color=PAPER, width=1),
        ),
        text=[f"{v:.4f}" for v in bench["F1"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color=INK),
    ))
    fig.update_layout(
        title="F1 by decision strategy",
        height=300, showlegend=False,
        xaxis=dict(range=[0, max(bench["F1"]) * 1.15]),
    )
    st.plotly_chart(_style_chart(fig), use_container_width=True)

    naive_f1 = bench.iloc[1]["F1"]
    business_f1 = bench.iloc[2]["F1"]
    xgb_f1 = bench.iloc[3]["F1"]
    gain_simple = (xgb_f1 - naive_f1) / max(naive_f1, 1e-9) * 100
    gain_business = (xgb_f1 - business_f1) / max(business_f1, 1e-9) * 100

    _card(
        "Reading",
        "What ML actually adds",
        f"<ul style='margin:0 0 0.6rem 1.2rem; padding:0;'>"
        f"<li>vs <em>always-no</em>: F1 goes from <strong>0.0000 → {xgb_f1:.4f}</strong>. Accuracy is misleading (84.5% baseline).</li>"
        f"<li>vs <em>PageValues &gt; 0</em>: ML adds <strong>+{gain_simple:.0f}%</strong> ({naive_f1:.4f} → {xgb_f1:.4f}). PageValues alone is unexpectedly strong.</li>"
        f"<li>vs <em>seasonal heuristic</em>: ML adds <strong>+{gain_business:.0f}%</strong> ({business_f1:.4f} → {xgb_f1:.4f}).</li>"
        f"</ul>"
        f"ML earns its complexity even against non-trivial business heuristics.",
        kind="default",
    )


# ---------------------------------------------------------------------------
# PART 1 · Problem & EDA
# ---------------------------------------------------------------------------


def part1_problem_and_eda(df: pd.DataFrame | None) -> None:
    _block_header(
        "01",
        "The brief",
        "Predict purchase intent — <em>before</em> the visitor leaves the session.",
    )
    col_a, col_b = st.columns([1.4, 1], gap="large")
    with col_a:
        st.markdown(
            '<p class="lead" style="font-family:var(--serif);font-size:20px;line-height:1.5;color:var(--ink-2);max-width:680px;">'
            'E-commerce sites convert 1 to 3% of visitors. The job: identify the high-intent sessions '
            "<em>in real time</em>, so growth and CRM teams spend their retargeting budget where it actually moves the needle — "
            'and stop interrupting the rest. <strong style="color:var(--ink);">Binary classification, supervised, severely imbalanced.</strong>'
            '</p>',
            unsafe_allow_html=True,
        )
    with col_b:
        _card(
            "Stakeholder",
            "Growth · CRM",
            "They own retargeting, pop-ups, coupons, live-chat triggers. The score lands in their tooling and decides which sessions get the premium experience.",
            kind="default",
        )

    if df is None:
        _card("Data missing", "Dataset not found",
              "Run <code>scripts/generate_plots.py</code> after downloading the UCI dataset.",
              kind="negative")
        return

    _block_header(
        "02",
        "The dataset",
        "12,330 sessions · 17 features · <em>15.5% positive class</em>.",
    )

    _kpi_strip([
        ("Sessions",      "12,330",  "rows · UCI ML Repo"),
        ("Features",      "17",      "10 num · 7 categorical"),
        ("Positive rate", "15.5%",   "the class that matters"),
        ("Test holdout",  "2,466",   "20% stratified split"),
    ])

    st.markdown("")  # space

    eda_tab1, eda_tab2, eda_tab3 = st.tabs([
        "Class imbalance",
        "Conversion by segment",
        "Top correlations",
    ])

    with eda_tab1:
        col1, col2 = st.columns([1.2, 1], gap="large")
        with col1:
            counts = df["Revenue"].value_counts()
            pct = (counts / counts.sum() * 100).round(1)
            fig = go.Figure(go.Bar(
                x=["No purchase · 84.5%", "Purchase · 15.5%"],
                y=[counts.get(False, 0), counts.get(True, 0)],
                text=[f"{counts.get(False, 0):,}", f"{counts.get(True, 0):,}"],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=12, color=INK),
                marker=dict(
                    color=[DATA_SLATE, SIGNAL],
                    line=dict(color=PAPER, width=1),
                ),
            ))
            fig.update_layout(
                title="Distribution of the target — heavy class imbalance",
                yaxis_title="Sessions",
                showlegend=False, height=400,
            )
            st.plotly_chart(_style_chart(fig), use_container_width=True)
        with col2:
            _card(
                "Why it matters",
                "Accuracy lies on imbalanced data",
                "A constant <code>False</code> classifier would score <strong>84.5% accuracy</strong> with zero recall on the positive class. "
                "F1 on the positive class is the headline metric here — not accuracy. "
                "Stratified splits everywhere; <strong>class_weight=balanced</strong> on every model.",
                kind="print",
            )

    with eda_tab2:
        seg = st.selectbox("Slice by", ["Month", "VisitorType", "TrafficType", "Weekend"])
        order = None
        if seg == "Month":
            order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        agg = (
            df.groupby(seg)["Revenue"]
              .agg(["mean", "count"])
              .rename(columns={"mean": "rate", "count": "n"})
              .reset_index()
        )
        if order:
            agg[seg] = pd.Categorical(agg[seg], categories=order, ordered=True)
            agg = agg.sort_values(seg)
        else:
            agg = agg.sort_values("rate", ascending=False)
        peak_mask = agg["rate"] >= agg["rate"].quantile(0.7)
        colors = [SIGNAL if p else INK for p in peak_mask]

        fig = go.Figure(go.Bar(
            x=agg[seg].astype(str),
            y=agg["rate"],
            text=[f"{r:.1%}" for r in agg["rate"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=10),
            marker=dict(color=colors, line=dict(color=PAPER, width=1)),
            hovertemplate="<b>%{x}</b><br>rate %{y:.1%}<br>n = %{customdata:,}<extra></extra>",
            customdata=agg["n"],
        ))
        fig.add_hline(
            y=df["Revenue"].mean(),
            line_dash="dash", line_color=INK, line_width=1,
            annotation_text=f"average · {df['Revenue'].mean():.1%}",
            annotation_font=dict(family="JetBrains Mono", size=10, color=INK),
        )
        fig.update_layout(
            title=f"Conversion rate · by {seg}",
            yaxis_title="Conversion rate",
            yaxis=dict(tickformat=".0%"),
            height=440, showlegend=False,
        )
        st.plotly_chart(_style_chart(fig), use_container_width=True)

        _card(
            "Reading",
            "Insights that survive the model",
            "<strong>Nov, Sep, Oct</strong> dominate (Black Friday + back-to-school). "
            "<strong>New_Visitor</strong> converts ~2× more than Returning_Visitor — counter-intuitive but the strongest single categorical signal. "
            "Some <code>TrafficType</code> codes show 3× the average rate.",
        )

    with eda_tab3:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        corrs = df[num_cols].corrwith(df["Revenue"].astype(int)).sort_values(key=abs, ascending=False)
        cdf = corrs.drop("Revenue", errors="ignore").reset_index()
        cdf.columns = ["feature", "corr"]
        top = cdf.head(10).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top["corr"], y=top["feature"], orientation="h",
            marker=dict(
                color=[SIGNAL if v > 0 else DATA_MOSS for v in top["corr"]],
                line=dict(color=PAPER, width=1),
            ),
            text=[f"{v:+.3f}" for v in top["corr"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=10, color=INK),
        ))
        fig.update_layout(
            title="Top 10 linear correlations with Revenue",
            height=440, showlegend=False,
            xaxis=dict(range=[min(top["corr"]) - 0.05, max(top["corr"]) + 0.1]),
        )
        st.plotly_chart(_style_chart(fig), use_container_width=True)

        _card(
            "Reading",
            "PageValues dominates",
            "<code>PageValues</code> (Google Analytics' own conversion-propensity score) carries a correlation of "
            "<strong>~0.49</strong> with the target — it's the single most predictive raw feature, "
            "and we'll see it dominate the XGBoost feature importance in part 02.",
            kind="print",
        )


# ---------------------------------------------------------------------------
# PART 2 · Models & metrics
# ---------------------------------------------------------------------------


def part2_models_and_metrics(metrics_df: pd.DataFrame | None,
                              test_pred: pd.DataFrame | None = None) -> None:
    _block_header(
        "03",
        "The benchmark",
        "Three model families. <em>One winner.</em>",
    )

    st.markdown(
        '<p class="lead" style="font-family:var(--serif);font-size:20px;line-height:1.5;color:var(--ink-2);max-width:760px;">'
        'Logistic regression as the interpretable baseline; random forest as the tree benchmark; '
        'XGBoost with Optuna search. Same anti-leakage <code>Pipeline</code>, same stratified CV, same F1 objective. '
        '<strong style="color:var(--ink);">F1 on the positive class is the headline.</strong>'
        '</p>',
        unsafe_allow_html=True,
    )

    if metrics_df is None:
        _card("Data missing", "Run the training script first",
              "Execute <code>python scripts/generate_plots.py</code> to materialize "
              "<code>results/model_metrics.csv</code>.",
              kind="negative")
        return

    metrics_sorted = metrics_df.sort_values("f1", ascending=False).reset_index(drop=True)
    encoders = {"logreg": "OneHot", "random_forest": "OneHot", "xgboost": "Ordinal"}
    thresholds = {"logreg": "0.244", "random_forest": "0.360", "xgboost": "0.305"}
    rows = []
    for _, r in metrics_sorted.iterrows():
        rows.append([
            r["model_name"],
            encoders.get(r["model_key"], "—"),
            thresholds.get(r["model_key"], "—"),
            f"{r['f1']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['roc_auc']:.4f}",
        ])
    headers = ["Model", "Encoder", "Threshold", "F1", "Precision", "Recall", "ROC AUC"]
    _data_table(headers, rows, winner_idx=0, f1_col_idx=3)

    st.markdown("")
    st.markdown(
        '<div style="border-top:1px solid var(--rule-hair);padding-top:24px;"></div>',
        unsafe_allow_html=True,
    )

    metric_choice = st.selectbox(
        "Sort the chart by metric",
        ["f1", "roc_auc", "precision", "recall", "accuracy"],
        index=0,
    )
    sorted_df = metrics_df.sort_values(metric_choice, ascending=True)
    fig = go.Figure(go.Bar(
        x=sorted_df[metric_choice],
        y=sorted_df["model_name"],
        orientation="h",
        marker=dict(
            color=[INK_4, INK, SIGNAL][:len(sorted_df)],
            line=dict(color=PAPER, width=1),
        ),
        text=[f"{v:.4f}" for v in sorted_df[metric_choice]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color=INK),
    ))
    fig.update_layout(
        title=f"Model comparison · {metric_choice.upper()}",
        height=320, showlegend=False,
        xaxis=dict(range=[0, max(sorted_df[metric_choice]) * 1.15]),
    )
    st.plotly_chart(_style_chart(fig), use_container_width=True)

    _block_header(
        "04",
        "Visual evidence",
        "ROC, PR, confusion matrix, feature importance — the canonical four.",
    )
    plot_tabs = st.tabs([
        "ROC curves", "Precision–Recall", "Confusion matrix · XGBoost", "Feature importance",
    ])
    plot_paths = [
        (PLOTS_DIR / "roc_curves.png", "XGBoost dominates with AUC = 0.93 — but the three families are within 2 points of each other. The lift comes from threshold tuning, not picking a fancier model."),
        (PLOTS_DIR / "pr_curves.png", "Precision stays above 0.7 until recall ~0.55 — the model is usable in production without flooding marketing with false alerts."),
        (PLOTS_DIR / "confusion_matrix_xgboost.png", "281 buyers detected · 101 missed (FN) · 172 false alerts (FP) at threshold 0.305 on the held-out test set."),
        (PLOTS_DIR / "feature_importance_xgb.png", "PageValues rules. ExitRates and BounceRates push negatively — visitors who flee don't buy."),
    ]
    for tab, (path, caption) in zip(plot_tabs, plot_paths):
        with tab:
            if path.exists():
                st.image(str(path), use_container_width=True)
                st.markdown(
                    f'<p style="font-family:var(--mono);font-size:11px;color:var(--ink-3);letter-spacing:.04em;margin-top:8px;">'
                    f'{caption}</p>',
                    unsafe_allow_html=True,
                )
            else:
                _card("File missing", f"{path.name} not found",
                      "Run <code>python scripts/generate_plots.py</code>.",
                      kind="negative")

    _card(
        "The verdict",
        "XGBoost wins. Threshold tuning seals it.",
        "<strong>F1 = 0.6731 · ROC-AUC = 0.9292.</strong> "
        "Recall = 0.7356 — we catch 74% of real buyers. "
        "The PoC target of F1 &gt; 0.60 is exceeded by <strong>+12%</strong>. "
        "Threshold tuning lifts F1 from 0.61 → 0.6731 — that's where the +12% comes from, not a fancier model.",
        kind="dark",
    )

    if test_pred is None or "proba_xgboost" not in test_pred.columns:
        return

    _block_header(
        "05",
        "Rigorous validation",
        "Calibration · per-segment errors · vs <em>non-ML</em> baselines.",
    )
    rig_tabs = st.tabs([
        "Calibration",
        "Errors by segment",
        "vs non-ML baselines",
    ])
    with rig_tabs[0]:
        _validation_calibration(test_pred)
    with rig_tabs[1]:
        _validation_error_analysis(test_pred)
    with rig_tabs[2]:
        _validation_naive_baseline(test_pred)


# ---------------------------------------------------------------------------
# PART 3 · Live demo
# ---------------------------------------------------------------------------


def part3_real_world_demo(pipeline, test_pred: pd.DataFrame | None) -> None:
    _block_header(
        "06",
        "The live scorer",
        "Score a synthetic session — adjust the inputs, watch the verdict.",
    )

    st.markdown(
        '<p class="lead" style="font-family:var(--serif);font-size:20px;line-height:1.5;color:var(--ink-2);max-width:780px;">'
        'The XGBoost pipeline running below is the production artifact — '
        '<code>pipeline.predict_proba(session)</code> returns a probability, your threshold turns it into '
        'a target / skip decision. Everything orange is the model. Everything ink is the framework around it.'
        '</p>',
        unsafe_allow_html=True,
    )

    if pipeline is None:
        _card("Model missing", "Train first",
              "Run <code>python scripts/train.py</code> to generate <code>models/xgboost.joblib</code>.",
              kind="negative")
        return

    st.markdown("")
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown(
            '<div style="font-family:var(--mono);font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--ink-3);margin-bottom:8px;">▪ Navigation</div>',
            unsafe_allow_html=True,
        )
        product_related = st.slider("Product pages viewed", 0, 200, 30)
        product_duration = st.slider("Product duration (s)", 0, 6000, 1000, step=50)
        admin_pages = st.slider("Admin pages (login / cart)", 0, 30, 2)
    with col2:
        st.markdown(
            '<div style="font-family:var(--mono);font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--ink-3);margin-bottom:8px;">▪ Engagement</div>',
            unsafe_allow_html=True,
        )
        page_values = st.slider("PageValues", 0.0, 200.0, 5.0, step=1.0)
        bounce_rate = st.slider("Bounce rate", 0.0, 0.2, 0.02, step=0.005, format="%.3f")
        exit_rate = st.slider("Exit rate", 0.0, 0.2, 0.04, step=0.005, format="%.3f")
    with col3:
        st.markdown(
            '<div style="font-family:var(--mono);font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--ink-3);margin-bottom:8px;">▪ Context</div>',
            unsafe_allow_html=True,
        )
        visitor_type = st.selectbox(
            "Visitor type",
            ["Returning_Visitor", "New_Visitor", "Other"],
        )
        month = st.selectbox(
            "Month",
            ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            index=8,
        )
        weekend = st.checkbox("Weekend session", value=False)

    threshold = st.slider(
        "Decision threshold · 0.305 = CV-tuned optimum",
        0.05, 0.95, 0.305, step=0.005,
    )

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
        _card("Prediction error", "Pipeline raised an exception", str(e),
              kind="negative")
        return

    will_target = proba >= threshold
    confidence = abs(proba - 0.5) * 2

    st.markdown("")
    cols = st.columns([1.4, 1], gap="large")
    with cols[0]:
        # Custom HTML gauge — matches design system spec
        bar_pct = proba * 100
        thresh_pct = threshold * 100
        verdict_label = "▲  Target" if will_target else "—  Skip"
        verdict_color = "var(--signal)" if will_target else "var(--ink-3)"

        gauge_html = f"""
        <div style="background: var(--surface); border: 1px solid var(--ink); padding: 28px;">
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 12px;">
                <div style="font-family: var(--mono); font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink-3);">
                    Probability of purchase
                </div>
                <div style="font-family: var(--mono); font-size: 10px; color: var(--ink-3);">
                    XGBoost · v1.0
                </div>
            </div>
            <div style="font-family: var(--serif); font-weight: 400; font-size: 84px; line-height: 1; letter-spacing: -0.04em; color: var(--ink); font-variation-settings: 'opsz' 144;">
                <span style="font-family: var(--mono); font-variant-numeric: tabular-nums; font-weight: 500;">{bar_pct:.1f}</span><span style="font-size: 32px; color: var(--ink-3); margin-left: 4px;">%</span>
            </div>
            <div style="height: 6px; background: var(--paper-3); margin-top: 18px; position: relative;">
                <div style="height: 100%; background: var(--signal); width: {bar_pct}%; transition: width .6s cubic-bezier(0.2, 0, 0, 1);"></div>
                <div style="position: absolute; top: -4px; bottom: -4px; left: {thresh_pct}%; width: 2px; background: var(--ink);"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-family: var(--mono); font-size: 10px; color: var(--ink-3); margin-top: 8px;">
                <span>0%</span><span>50%</span><span>100%</span>
            </div>
            <div style="margin-top: 22px; padding-top: 18px; border-top: 1px solid var(--rule-hair);">
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span style="font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3);">Decision</span>
                    <span style="font-family: var(--serif); font-weight: 600; font-size: 22px; color: {verdict_color};">{verdict_label}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top: 8px;">
                    <span style="font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3);">Confidence</span>
                    <span style="font-family: var(--mono); font-variant-numeric: tabular-nums; font-weight: 600; font-size: 18px; color: var(--ink);">{confidence:.0%}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top: 8px;">
                    <span style="font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3);">Threshold</span>
                    <span style="font-family: var(--mono); font-variant-numeric: tabular-nums; font-weight: 600; font-size: 18px; color: var(--ink);">{threshold:.3f}</span>
                </div>
            </div>
        </div>
        """
        st.markdown(gauge_html, unsafe_allow_html=True)

    with cols[1]:
        _card(
            "How to read the gauge",
            "Threshold is policy",
            "The orange tick on the bar is your threshold. Anything to the right gets a marketing action; "
            "anything to the left gets ignored. Drop the threshold to spend more budget and catch more buyers; "
            "raise it to stay precise and economic. The science is the model. <strong>The decision is yours.</strong>",
            kind="dark",
        )

    _block_header(
        "07",
        "ROI on the test set",
        "Apply your threshold to all 2,466 unseen sessions.",
    )

    if test_pred is not None and "proba_xgboost" in test_pred.columns:
        y_true = test_pred["y_true"].astype(int).values
        y_score = test_pred["proba_xgboost"].values
        y_pred = (y_score >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        total_real_buyers = tp + fn
        total_targeted = tp + fp

        _kpi_strip([
            ("Sessions targeted", f"{total_targeted:,}",
             f"{total_targeted/len(y_true):.0%} of traffic"),
            ("Buyers caught", f"{tp:,}",
             f"recall {tp/max(total_real_buyers,1):.0%}"),
            ("False alerts", f"{fp:,}",
             f"precision {tp/max(tp+fp,1):.0%}"),
            ("Buyers missed", f"{fn:,}",
             f"{fn/max(total_real_buyers,1):.0%} missed"),
        ])

        fig = go.Figure(go.Bar(
            x=["Buyers caught", "False alerts", "Buyers missed", "Correctly ignored"],
            y=[tp, fp, fn, tn],
            marker=dict(
                color=[SIGNAL, DATA_SAND, NEGATIVE, DATA_SLATE],
                line=dict(color=PAPER, width=1),
            ),
            text=[f"{v:,}" for v in [tp, fp, fn, tn]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color=INK),
        ))
        fig.update_layout(
            title=f"Decision decomposition at threshold = {threshold:.3f}",
            height=340, showlegend=False, yaxis_title="Sessions",
        )
        st.plotly_chart(_style_chart(fig), use_container_width=True)

        _card(
            "Reading the trade-off",
            "What this threshold buys you",
            f"<ul style='margin: 0 0 0.6rem 1.2rem; padding: 0;'>"
            f"<li>You target <strong>{total_targeted:,} sessions</strong> — {total_targeted/len(y_true):.0%} of all incoming traffic.</li>"
            f"<li>Of those, <strong>{tp:,} are real buyers</strong> — your gain.</li>"
            f"<li><strong>{fp:,} are not</strong> — the cost of the action wasted.</li>"
            f"<li>You miss <strong>{fn:,} buyers</strong> who would have bought anyway.</li>"
            f"</ul>"
            f"Lower the threshold if a conversion is worth far more than a wasted marketing action; "
            f"raise it if precision is the constraint.",
            kind="default",
        )
    else:
        _card("Data missing", "Test predictions not found",
              "Run <code>python scripts/generate_plots.py</code> to materialize "
              "<code>results/test_predictions.csv</code>.",
              kind="negative")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _sidebar() -> None:
    sidebar_html = (
        '<div style="text-align: center; padding: 1.4rem 0 1rem 0;">'
        f'{_shopsignal_mark_svg(36)}'
        '<div style="font-family: var(--serif); font-size: 1.4rem; font-weight: 500; color: var(--paper); margin-top: 0.5rem; letter-spacing: -0.02em;">ShopSignal</div>'
        '<div style="font-family: var(--mono); font-size: 0.7rem; color: rgba(244,241,234,0.55); letter-spacing: 0.18rem; '
        'text-transform: uppercase; margin-top: 0.3rem;">Conversion intelligence</div>'
        '</div>'
    )
    st.markdown(sidebar_html, unsafe_allow_html=True)
    st.markdown(
        '<div style="margin: 0 0.6rem; padding: 1rem 1.2rem; '
        'background: rgba(255, 91, 31, 0.08); border-left: 2px solid var(--signal); '
        'font-family: var(--sans); font-size: 0.82rem; color: rgba(244, 241, 234, 0.85); line-height: 1.6;">'
        '<div style="font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--signal); margin-bottom: 0.5rem;">▪ Author</div>'
        '<div style="font-weight: 600; color: var(--paper); font-size: 0.95rem; font-family: var(--serif);">Manech Carriou</div>'
        '<div style="color: rgba(244,241,234,0.65); font-size: 0.78rem; margin-top: 0.2rem;">Albert School · ML PoC</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="margin: 1rem 0.6rem; padding: 1rem 1.2rem; '
        'background: rgba(255,255,255,0.04); border: 1px solid rgba(244,241,234,0.1); '
        'font-size: 0.82rem; color: rgba(244,241,234,0.85); line-height: 1.6;">'
        '<div style="font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase; color: rgba(244,241,234,0.6); margin-bottom: 0.5rem;">Pipeline</div>'
        'EDA → preprocessing<br>(anti-leakage) → 3 models<br>→ Optuna + MLflow<br>→ threshold tuning'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="margin: 0 0.6rem; font-family: var(--mono); font-size: 0.72rem; line-height: 1.8; color: rgba(244,241,234,0.6);">'
        '<a href="https://github.com/manechcarriou-lab/ml-poc-project" '
        'style="color: var(--signal) !important; text-decoration: none;">→ GitHub</a><br>'
        '<a href="https://github.com/manechcarriou-lab/ml-poc-project/blob/main/deliverables/RAPPORT_COMPLET.md" '
        'style="color: var(--signal) !important; text-decoration: none;">→ Methodology</a>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_app() -> None:
    st.set_page_config(
        page_title="ShopSignal — Conversion intelligence",
        layout="wide",
        page_icon=":material/insights:",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    with st.sidebar:
        _sidebar()

    _header()

    df = _load_dataset()
    metrics_df = _load_metrics()
    test_pred = _load_test_predictions()
    pipeline = _load_xgb_pipeline()

    # Hero
    _hero(
        part_label="PART 00",
        headline_html='Predict <em>who buys</em> — before<br/>they leave the session.',
        lead_html=(
            "85% of e-commerce visitors leave without buying. "
            "<strong>The 15% who don't are worth knowing about.</strong> "
            "ShopSignal scores every session live, so retargeting, popups and live chat "
            "hit only where they move revenue."
        ),
        kpis=[
            ("F1 · winner",  "0.6731",       "+12% vs PoC target 0.60"),
            ("Recall",        "73.6<span class='u'>%</span>", "281 of 382 buyers caught"),
            ("ROC-AUC",       "0.9292",       "held-out · n = 2,466"),
            ("Threshold",     "0.305",        "CV-tuned · anti-leakage"),
        ],
        metabar_items=[
            ("Dataset",   "UCI Online Shoppers"),
            ("Sessions",  "12,330"),
            ("Features",  "17 · 10 num · 7 cat"),
            ("Imbalance", "85 / 15"),
            ("Pipeline",  "3 models · Optuna · MLflow"),
            ("Source",    "github.com/manechcarriou-lab"),
        ],
    )

    tab1, tab2, tab3 = st.tabs([
        "01 · Problem & EDA",
        "02 · Models & metrics",
        "03 · Live demo",
    ])

    with tab1:
        part1_problem_and_eda(df)
    with tab2:
        part2_models_and_metrics(metrics_df, test_pred)
    with tab3:
        part3_real_world_demo(pipeline, test_pred)

    _footer()


if __name__ == "__main__":
    build_app()
