"""Generate the slide deck `deliverables/process_overview.pptx`.

Run with:  python scripts/build_slides.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "deliverables" / "process_overview.pptx"

PRIMARY = RGBColor(0x1F, 0x3A, 0x5F)  # deep blue
ACCENT = RGBColor(0xE6, 0x7E, 0x22)   # warm orange
LIGHT = RGBColor(0xF4, 0xF6, 0xF8)
DARK = RGBColor(0x2C, 0x3E, 0x50)
GREY = RGBColor(0x7F, 0x8C, 0x8D)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SUCCESS = RGBColor(0x27, 0xAE, 0x60)


def _add_text(box, text, size=18, bold=False, color=DARK, align=PP_ALIGN.LEFT, name="Calibri"):
    tf = box.text_frame
    tf.word_wrap = True
    tf.text = ""
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name
    return tf


def _add_bullets(box, items, size=16, color=DARK, name="Calibri"):
    tf = box.text_frame
    tf.word_wrap = True
    tf.text = ""
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = f"• {item}"
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = name
        p.space_after = Pt(6)


def _add_band(slide, top_in, height_in, color):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(top_in), Inches(13.333), Inches(height_in)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def _add_textbox(slide, left, top, width, height):
    return slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))


def _slide_with_title(prs, title, subtitle=None, page_num=None, total=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    # Top band
    band = _add_band(slide, 0, 0.9, PRIMARY)
    box = _add_textbox(slide, 0.5, 0.18, 12.3, 0.6)
    _add_text(box, title, size=26, bold=True, color=WHITE)
    if subtitle:
        sub = _add_textbox(slide, 0.5, 0.95, 12.3, 0.4)
        _add_text(sub, subtitle, size=14, color=GREY)
    if page_num is not None and total is not None:
        page = _add_textbox(slide, 11.7, 0.25, 1.3, 0.5)
        _add_text(page, f"{page_num} / {total}", size=12, color=WHITE, align=PP_ALIGN.RIGHT)
    return slide


def _make_table(slide, left_in, top_in, width_in, height_in, headers, rows):
    rows_n = len(rows) + 1
    cols_n = len(headers)
    table_shape = slide.shapes.add_table(
        rows_n, cols_n,
        Inches(left_in), Inches(top_in), Inches(width_in), Inches(height_in),
    )
    table = table_shape.table
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = h
        run.font.bold = True
        run.font.size = Pt(13)
        run.font.color.rgb = WHITE
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r % 2 else LIGHT
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(12)
            run.font.color.rgb = DARK
            if c == 0:
                run.font.bold = True
    return table


# ---------------------------------------------------------------------------


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    TOTAL = 14

    # --- Slide 1: cover ---
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_band(slide, 0, 7.5, PRIMARY)
    title_box = _add_textbox(slide, 0.8, 2.1, 11.7, 1.5)
    _add_text(
        title_box,
        "Prédire la conversion d'un visiteur e-commerce",
        size=44, bold=True, color=WHITE,
    )
    sub_box = _add_textbox(slide, 0.8, 3.5, 11.7, 0.8)
    _add_text(
        sub_box,
        "Du repo vide au modèle XGBoost — choix techniques et justifications",
        size=20, color=RGBColor(0xCF, 0xDC, 0xE7),
    )
    info_box = _add_textbox(slide, 0.8, 5.5, 11.7, 1.2)
    _add_bullets(
        info_box,
        [
            "Manech Carriou — Albert School",
            "Projet ML PoC — fork du template `thom1100/ml-poc-project`",
            "Repo : https://github.com/manechcarriou-lab/ml-poc-project",
        ],
        size=14, color=RGBColor(0xCF, 0xDC, 0xE7),
    )

    # --- Slide 2: agenda ---
    s = _slide_with_title(prs, "Agenda", "Comment lire ce deck", 2, TOTAL)
    box = _add_textbox(s, 0.8, 1.5, 11.7, 5.5)
    _add_bullets(
        box,
        [
            "Le problème business — quoi prédire et pourquoi",
            "Le dataset — choix et limites",
            "Setup technique — du fork au venv",
            "EDA — comprendre la donnée avant de modéliser",
            "Feature engineering — sans data leakage",
            "Modélisation — 3 familles + Optuna",
            "Tracking — MLflow",
            "Résultats — XGBoost gagne",
            "Conclusion + suite",
        ],
        size=18,
    )

    # --- Slide 3: business ---
    s = _slide_with_title(prs, "Le problème business", "Pourquoi ce sujet", 3, TOTAL)
    box = _add_textbox(s, 0.8, 1.5, 11.7, 1.2)
    _add_text(
        box,
        "Peut-on prédire dès le début d'une session qu'un visiteur va acheter ?",
        size=22, bold=True, color=PRIMARY,
    )
    box = _add_textbox(s, 0.8, 2.7, 11.7, 4)
    _add_bullets(
        box,
        [
            "Conversion moyenne e-commerce : 1 à 3 % — les sessions à fort potentiel sont rares",
            "Identifier ces sessions permet de cibler le retargeting / coupons / pop-ups",
            "Décision opérationnelle : déclencher (ou non) une action marketing pendant la session",
            "Stakeholder cible : équipe Growth / CRM d'un retailer en ligne",
            "Type de problème ML : classification binaire supervisée (Revenue ∈ {True, False})",
        ],
        size=17,
    )

    # --- Slide 4: dataset ---
    s = _slide_with_title(prs, "Le dataset", "Online Shoppers Purchasing Intention (UCI)", 4, TOTAL)
    _make_table(
        s, 0.8, 1.4, 11.7, 3.0,
        ["Champ", "Valeur"],
        [
            ["Source", "UCI Machine Learning Repository (CC BY 4.0)"],
            ["Volume", "12 330 sessions × 18 colonnes"],
            ["Features", "10 numériques + 7 catégorielles"],
            ["Cible", "Revenue (booléen → achat ou non)"],
            ["Balance des classes", "84.5 % False / 15.5 % True (déséquilibré)"],
        ],
    )
    box = _add_textbox(s, 0.8, 4.7, 11.7, 0.5)
    _add_text(box, "Limites assumées", size=17, bold=True, color=ACCENT)
    box = _add_textbox(s, 0.8, 5.2, 11.7, 2.0)
    _add_bullets(
        box,
        [
            "Pas d'identifiant utilisateur → pas de parcours multi-session",
            "Pas d'année → pas de saisonnalité multi-annuelle",
            "Modalités catégorielles anonymisées (Region 1-9, OS 1-8…)",
        ],
        size=15,
    )

    # --- Slide 5: setup ---
    s = _slide_with_title(prs, "Setup technique", "Stack et reproductibilité", 5, TOTAL)
    _make_table(
        s, 0.8, 1.4, 11.7, 3.5,
        ["Composant", "Choix", "Pourquoi"],
        [
            ["Versionning", "Git + GitHub", "Standard, fork du template du cours"],
            ["Auth GitHub", "Clé SSH ed25519", "Pas de token à gérer, sécurisé"],
            ["CLI GitHub", "gh", "Forker, push, PR depuis le terminal"],
            ["Python", "3.13 dans .venv", "Isolation des dépendances"],
            ["Tracking ML", "MLflow 3.11", "Persistance des essais + UI web"],
            ["Hyperparam search", "Optuna 4.8", "Recherche bayésienne (TPE)"],
        ],
    )
    box = _add_textbox(s, 0.8, 5.2, 11.7, 1.6)
    _add_text(box, "Reproductibilité — 4 commandes", size=16, bold=True, color=ACCENT)
    box = _add_textbox(s, 0.8, 5.6, 11.7, 1.6)
    _add_bullets(
        box,
        [
            "git clone git@github.com:manechcarriou-lab/ml-poc-project.git",
            "python -m venv .venv && .venv\\Scripts\\activate",
            "pip install -r requirements.txt",
            "python scripts/train.py --trials 15",
        ],
        size=13, name="Consolas", color=DARK,
    )

    # --- Slide 6: EDA highlights ---
    s = _slide_with_title(prs, "EDA — checks de qualité", "data_exploration.ipynb", 6, TOTAL)
    _make_table(
        s, 0.5, 1.4, 12.3, 4.5,
        ["Check", "Résultat", "Décision"],
        [
            ["NaN globaux", "0 ligne, 0 valeur", "Pas d'imputation"],
            ["Features avec >5 % NaN", "aucune", "Pas de drop"],
            ["Outliers IQR (>1.5 IQR)", "PageValues, BounceRates, *_Duration", "log1p + scaling, pas de suppression"],
            ["Drift (1ère vs 2ème moitié)", "SpecialDay seulement au début", "À surveiller"],
            ["Imbalance cible", "84.5 / 15.5", "stratify=y + class_weight=balanced"],
            ["Modalités rares", "VisitorType=Other (0.7 %)", "OneHotEncoder(handle_unknown=ignore)"],
        ],
    )
    box = _add_textbox(s, 0.8, 6.1, 11.7, 1.0)
    _add_text(
        box,
        "Insight clé : New_Visitor convertit 2× plus que Returning_Visitor — feature très discriminante.",
        size=14, color=ACCENT, bold=True,
    )

    # --- Slide 7: feature engineering pipeline ---
    s = _slide_with_title(prs, "Feature engineering", "Pipeline anti-leakage par construction", 7, TOTAL)
    box = _add_textbox(s, 0.8, 1.4, 11.7, 0.6)
    _add_text(box, "src/features.py — un seul ColumnTransformer dans un Pipeline sklearn", size=16, color=PRIMARY, bold=True)
    box = _add_textbox(s, 0.8, 2.0, 11.7, 3.0)
    _add_bullets(
        box,
        [
            "Skewed numeric (durations, PageValues) → log1p puis StandardScaler",
            "Numeric standard → StandardScaler",
            "Catégoriels → OneHotEncoder(handle_unknown='ignore')",
            "Engineered binaires → passthrough",
        ],
        size=15,
    )
    box = _add_textbox(s, 0.8, 5.1, 11.7, 0.5)
    _add_text(box, "Pourquoi un Pipeline ?", size=16, bold=True, color=ACCENT)
    box = _add_textbox(s, 0.8, 5.5, 11.7, 1.8)
    _add_bullets(
        box,
        [
            "fit() est appelé uniquement sur le train (et sur les folds train de la CV)",
            "transform() applique au test les statistiques apprises sur le train",
            "→ aucune fuite test → train possible. Garantie standard sklearn.",
        ],
        size=14,
    )

    # --- Slide 8: features added ---
    s = _slide_with_title(prs, "Features ajoutées", "Transformations row-wise stateless", 8, TOTAL)
    _make_table(
        s, 0.5, 1.4, 12.3, 5.0,
        ["Feature", "Définition", "Intuition métier"],
        [
            ["TotalPages", "Σ des compteurs de pages", "Volume global d'engagement"],
            ["TotalDuration", "Σ des durations", "Temps passé total"],
            ["AvgTimePerPage", "duration / pages", "Profondeur de lecture"],
            ["ProductRelatedRatio", "pages produit / total", "Focus produit"],
            ["HighPageValue", "PageValues > 0", "Flag session marchande"],
            ["IsHighBounce", "BounceRates > Q3", "Flag session zappée"],
            ["IsSpecialDay", "SpecialDay > 0", "Jour spécial (St-Valentin, etc.)"],
        ],
    )
    box = _add_textbox(s, 0.8, 6.7, 11.7, 0.5)
    _add_text(
        box,
        "Toutes ces transformations sont déterministes ligne par ligne — pas de fit, donc pas de leakage.",
        size=13, color=GREY,
    )

    # --- Slide 9: models ---
    s = _slide_with_title(prs, "Les 3 modèles testés", "Couvrir 3 familles différentes", 9, TOTAL)
    _make_table(
        s, 0.8, 1.4, 11.7, 3.0,
        ["Famille", "Pourquoi"],
        [
            ["Logistic Regression", "Baseline interprétable et rapide"],
            ["Random Forest", "Robuste, gère les features mixtes, peu de tuning"],
            ["XGBoost", "State-of-the-art sur du tabulaire (gradient boosting)"],
        ],
    )
    box = _add_textbox(s, 0.8, 4.7, 11.7, 0.5)
    _add_text(box, "Tous dans la même pipeline preprocessor → classifieur", size=15, bold=True, color=ACCENT)
    box = _add_textbox(s, 0.8, 5.2, 11.7, 1.8)
    _add_bullets(
        box,
        [
            "Optimisation : Optuna avec sampler TPE (recherche bayésienne)",
            "Validation : 3-fold stratified CV sur le train uniquement",
            "Objectif : maximiser le F1 (classe positive)",
            "class_weight='balanced' / scale_pos_weight pour compenser le 85/15",
        ],
        size=14,
    )

    # --- Slide 10: optuna + mlflow ---
    s = _slide_with_title(prs, "Optuna + MLflow", "scripts/train.py — la boucle d'expérimentation", 10, TOTAL)
    box = _add_textbox(s, 0.5, 1.4, 6.0, 5.0)
    _add_text(box, "Optuna — pourquoi", size=18, bold=True, color=PRIMARY)
    box = _add_textbox(s, 0.5, 1.9, 6.0, 4.5)
    _add_bullets(
        box,
        [
            "Recherche bayésienne (TPE) > Grid Search",
            "Échantillonne les bons coins, ignore les mauvais",
            "Espaces continus (learning_rate log-uniforme)",
            "15 trials Optuna ≈ 100 trials Grid",
            "Convergence rapide sur des espaces de 5-10 hyperparamètres",
        ],
        size=14,
    )
    box = _add_textbox(s, 6.8, 1.4, 6.0, 5.0)
    _add_text(box, "MLflow — pourquoi", size=18, bold=True, color=PRIMARY)
    box = _add_textbox(s, 6.8, 1.9, 6.0, 4.5)
    _add_bullets(
        box,
        [
            "Trace persistante des essais",
            "Compare 2 runs séparés de plusieurs jours",
            "Log auto : params + métriques + artefacts",
            "UI web : mlflow ui --backend-store-uri ./mlruns",
            "Artefact = le joblib du pipeline final",
        ],
        size=14,
    )

    # --- Slide 11: results ---
    s = _slide_with_title(prs, "Résultats", "Test set 2 466 sessions, 15 trials par famille", 11, TOTAL)
    _make_table(
        s, 0.5, 1.4, 12.3, 2.8,
        ["Modèle", "CV F1", "Test F1", "Precision", "Recall", "ROC-AUC"],
        [
            ["Logistic Regression", "0.6737", "0.5994", "0.7206", "0.5131", "0.9137"],
            ["Random Forest", "0.6843", "0.6292", "0.7500", "0.5419", "0.9202"],
            ["XGBoost (winner)", "0.6858", "0.6542", "0.7238", "0.5969", "0.9303"],
        ],
    )
    # success badge
    badge = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.5), Inches(4.7), Inches(6.3), Inches(1.2))
    badge.fill.solid()
    badge.fill.fore_color.rgb = SUCCESS
    badge.line.fill.background()
    tf = badge.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = "Critère atteint — F1 > 0.60 visé, 0.6542 obtenu (XGBoost) → +9 %"
    run.font.size = Pt(18)
    run.font.bold = True
    run.font.color.rgb = WHITE
    box = _add_textbox(s, 0.8, 6.1, 11.7, 1.0)
    _add_text(
        box,
        "Lecture business : sur 100 sessions classées 'achat probable', ~72 sont vraiment des acheteurs (precision 0.72).",
        size=13, color=GREY,
    )

    # --- Slide 12: business reading ---
    s = _slide_with_title(prs, "Lecture business du modèle", "XGBoost en exploitation", 12, TOTAL)
    box = _add_textbox(s, 0.8, 1.4, 11.7, 5.0)
    _add_bullets(
        box,
        [
            "Precision = 0.72 → 72 % des sessions ciblées sont vraiment des acheteurs (peu de gaspillage marketing)",
            "Recall = 0.60 → on attrape 60 % des acheteurs réels (on en rate 40 %)",
            "ROC-AUC = 0.93 → excellent ranking : le modèle distingue bien acheteurs et non-acheteurs",
            "Le seuil de 0.5 peut être abaissé pour viser plus de recall (plus d'acheteurs captés, plus de FP)",
            "Le seuil optimal dépend du coût d'une action marketing vs la valeur d'une conversion",
        ],
        size=16,
    )

    # --- Slide 13: next steps ---
    s = _slide_with_title(prs, "Prochaines étapes", "Ce qui reste à faire", 13, TOTAL)
    box = _add_textbox(s, 0.8, 1.4, 11.7, 5.5)
    _add_bullets(
        box,
        [
            "src/app.py — Streamlit : contexte business + EDA + démo interactive XGBoost",
            "Threshold tuning — choisir un seuil métier au lieu de 0.5",
            "Ajouter PR-AUC dans le panel de métriques (plus informatif sur dataset déséquilibré)",
            "Tests unitaires sur le pipeline de preprocessing (tests/)",
            "Affiner XGBoost avec un budget Optuna plus large (~50-100 trials)",
        ],
        size=17,
    )

    # --- Slide 14: takeaways ---
    s = _slide_with_title(prs, "À retenir", "Synthèse", 14, TOTAL)
    box = _add_textbox(s, 0.8, 1.4, 11.7, 5.5)
    _add_bullets(
        box,
        [
            "Choix dataset : public, business-relevant, déséquilibré (réaliste)",
            "Métrique principale : F1 sur classe positive (imbalanced + coût FP ≈ FN)",
            "Pipeline sklearn ColumnTransformer = garantie zéro fuite test → train",
            "Optuna (TPE) > Grid Search pour des espaces continus",
            "MLflow = mémoire persistante de toutes les expériences",
            "XGBoost gagne avec F1 = 0.6542 et ROC-AUC = 0.9303",
            "Tout est reproductible en 4 commandes",
        ],
        size=17,
    )
    # Final accent block
    accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.0), Inches(13.333), Inches(0.5))
    accent.fill.solid()
    accent.fill.fore_color.rgb = ACCENT
    accent.line.fill.background()
    tf = accent.text_frame
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = "Merci — Manech Carriou — github.com/manechcarriou-lab/ml-poc-project"
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = WHITE

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Wrote {OUT.relative_to(ROOT)}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
