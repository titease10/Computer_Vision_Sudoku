"""Application Streamlit : photo d'un sudoku papier -> solution écrite sur la photo.

Lancement :
    .venv/Scripts/streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import copy
import hashlib
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import charts  # noqa: E402

from sudoku_cv.pipeline import SudokuPipeline, render_solution  # noqa: E402
from sudoku_cv.solver import count_solutions, is_valid_grid, solve, solve_trace  # noqa: E402

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
LOCAL_WEIGHTS = (ROOT / "runs/grid/grid_v1/weights/best.pt", ROOT / "runs/cells/cells_v2/weights/best.pt")
LOW_CONF = 0.8  # en dessous, la case est signalée comme douteuse

LIGHT_VARS = """--ink: #1F1D1A; --muted: #6A665C; --line: #E3DFD2; --paper: #FAF8F3; --card: #FFFEFA;
  --green: #1B6E3C; --green-soft: #E4F0E8; --amber: #F6E3B4; --rose: #F3CDBB; --blue: #CFE3F6; --red: #F2B8B5;
  --shadow: rgba(31,29,26,.14); color-scheme: light;"""
DARK_VARS = """--ink: #ECE8DD; --muted: #A59F90; --line: #37352E; --paper: #16150F; --card: #1F1E18;
  --green: #62C48C; --green-soft: #1F3A2B; --amber: #5C4B1B; --rose: #5E3828; --blue: #204360; --red: #6E302E;
  --shadow: rgba(0,0,0,.5); color-scheme: dark;"""

DARK_TABLE = '[data-testid="stDataFrame"] { filter: invert(.9) hue-rotate(180deg); }'

CSS = """
<style>
:root { __LIGHT__ }
@media (prefers-color-scheme: dark) { :root { __DARK__ } }
:root { --serif: "Iowan Old Style", "Palatino Linotype", Georgia, serif; }
.stApp { background: var(--paper); color: var(--ink); }
[data-testid="stSidebar"] { background: var(--card); }
.stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp [data-testid="stMarkdownContainer"] { color: var(--ink); }
[data-testid="stCaptionContainer"] { color: var(--muted); }
/* widgets natifs : Streamlit reste clair même quand le thème est sombre */
[data-testid="stButtonGroup"] button, [data-testid^="stBaseButton-segmented_control"] { background: var(--card); color: var(--ink); border-color: var(--line); }
[data-testid="stButtonGroup"] button *, [data-testid^="stBaseButton-segmented_control"] * { color: inherit; }
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-pressed="true"] { background: var(--green-soft) !important; color: var(--green) !important; border-color: var(--green) !important; font-weight: 600; }
[data-testid="stBaseButton-segmented_controlActive"] { background: var(--green-soft); color: var(--green); border-color: var(--green); }
[data-testid="stFileUploaderDropzone"] button, .stButton > button, .stDownloadButton > button { background: var(--card); color: var(--ink); }
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span { color: var(--muted); }
[data-testid="stSpinner"], [data-testid="stSpinner"] * { background: transparent; color: var(--ink); }
[data-testid="stExpander"] summary { background: var(--card); }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--ink); }
[data-testid="stExpanderDetails"] { background: var(--card); }
[data-testid="stElementToolbarButtonContainer"] { background: var(--card); border: 1px solid var(--line); border-radius: 8px; }
[data-testid="stElementToolbarButtonContainer"] * { color: var(--ink); }
[data-testid="stSidebar"] * { color: var(--ink); }
.stButton > button:disabled { opacity: .45; }
[data-testid="stFileChip"] { background: var(--paper); border: 1px solid var(--line); }
[data-testid="stFileChip"] *, [data-testid="stFileChipName"] { color: var(--ink); }
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"] { display: none !important; }
.block-container { max-width: 1120px; padding: 2rem 1.5rem 4rem; }

/* en-tête */
.hero { padding: 0 0 1.6rem; margin-top: -2.2rem; border-bottom: 1px solid var(--line); margin-bottom: 1.6rem; }
.hero .kicker { font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; color: var(--green); font-weight: 600; }
.hero h1 { font-family: var(--serif); font-size: 2.8rem; line-height: 1.1; margin: .25rem 0 .5rem; color: var(--ink); }
.hero p { font-size: 1.05rem; color: var(--muted); margin: 0; max-width: 40rem; }
.hero .tags { margin-top: .9rem; display: flex; flex-wrap: wrap; gap: .4rem; }
.hero .tags span { font-size: .78rem; padding: .18rem .65rem; border-radius: 999px; background: var(--green-soft); color: var(--green); }

/* zone d'upload et boutons */
[data-testid="stFileUploaderDropzone"] {
  border: 2px dashed #C9C3B2; border-radius: 14px; background: var(--card); padding: 2rem 1rem;
  transition: border-color .15s, background .15s;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--green); background: var(--green-soft); }
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; border: 1px solid var(--line); font-weight: 600; transition: transform .1s, box-shadow .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(31,29,26,.12); }
.stButton > button[kind="primary"], .stButton > button[kind="primary"] * { background: var(--green); border-color: var(--green); color: var(--paper); }
[data-testid="stImage"] img { border-radius: 12px; box-shadow: 0 2px 14px var(--shadow); }
[data-testid="stExpander"] { border-radius: 12px; border-color: var(--line); background: var(--card); }

/* indicateur d'étapes */
.stepper { display: flex; gap: .5rem; margin: .4rem 0 1.1rem; flex-wrap: wrap; }
.stepper .s { display: flex; align-items: center; gap: .5rem; padding: .35rem .85rem .35rem .4rem; border-radius: 999px;
  border: 1px solid var(--line); background: var(--card); color: var(--muted); font-size: .88rem; }
.stepper .n { width: 1.5rem; height: 1.5rem; border-radius: 50%; display: grid; place-items: center; font-size: .78rem;
  background: var(--line); color: var(--muted); font-weight: 700; }
.stepper .done .n { background: var(--green); color: #fff; }
.stepper .done { color: var(--ink); }
.stepper .on { border-color: var(--green); background: var(--green-soft); color: var(--green); font-weight: 600; }
.stepper .on .n { background: var(--green); color: #fff; }
.step-text { color: var(--muted); margin: 0 0 1rem; }

/* tuiles de chiffres clés */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: .7rem; margin: 0 0 1rem; }
.tile { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: .7rem .9rem; }
.tile b { display: block; font-family: var(--serif); font-size: 1.7rem; line-height: 1.15; color: var(--ink); }
.tile > span { font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
.tile.good b { color: var(--green); }

/* la grille */
.board {
  display: grid; box-sizing: border-box; aspect-ratio: 1; width: 100%; max-width: 440px; overflow: hidden;
  grid-template-columns: repeat(9, minmax(0, 1fr)); grid-template-rows: repeat(9, minmax(0, 1fr));
  border: 2px solid var(--ink); border-radius: 10px; background: var(--card);
  box-shadow: 0 6px 22px var(--shadow);
  font-family: var(--serif); font-size: clamp(1rem, 4.2vw, 1.5rem); font-variant-numeric: tabular-nums;
}
.board div { box-sizing: border-box; display: flex; align-items: center; justify-content: center; border: 1px solid var(--line); }
.board div:nth-child(9n+3), .board div:nth-child(9n+6) { border-right: 2px solid var(--ink); }
.board div:nth-child(n+19):nth-child(-n+27), .board div:nth-child(n+46):nth-child(-n+54) { border-bottom: 2px solid var(--ink); }
.board .found { color: var(--green); font-weight: 700; }
.board .doubt { background: var(--amber); }
.board .fixed { background: var(--rose); }
.board .try { background: var(--blue); }
.board .undo { background: var(--red); }
.legend { font-size: .85rem; color: var(--muted); margin-top: .7rem; }
.legend b { font-weight: normal; padding: .1em .5em; border-radius: 4px; }
.legend .l-found { color: var(--green); font-weight: 700; }
.legend .l-doubt { background: var(--amber); }
.legend .l-fixed { background: var(--rose); }
/* panneau « sous le capot » */
.info { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1.1rem 1.3rem; margin: 1.2rem 0 1rem;
  display: grid; grid-template-columns: 3fr 2fr; gap: 1.5rem; }
.info h4 { margin: 0 0 .6rem; font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; color: var(--green); }
.info dl { margin: 0; display: grid; grid-template-columns: max-content 1fr; gap: .45rem 1rem; font-size: .92rem; }
.info dt { color: var(--muted); }
.info dd { margin: 0; color: var(--ink); }
.info code { background: var(--green-soft); color: var(--green); padding: .05em .4em; border-radius: 4px; font-size: .85em; }
.info .info-chart { grid-column: 1 / -1; border-top: 1px solid var(--line); padding-top: 1rem; }
.info .tiles { margin: 0; grid-template-columns: repeat(2, 1fr); }
.info .tile b { font-size: 1.3rem; }
@media (max-width: 760px) { .info { grid-template-columns: 1fr; } }

/* animations */
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes pop { 0% { opacity: 0; transform: scale(.3); } 70% { transform: scale(1.18); } 100% { opacity: 1; transform: scale(1); } }
@keyframes ring { 0% { box-shadow: 0 0 0 0 rgba(27,110,60,.45); } 100% { box-shadow: 0 0 0 9px rgba(27,110,60,0); } }
.hero, .stepper, .step-text, .info, .tiles .tile { animation: rise .55s cubic-bezier(.2,.7,.2,1) both; }
.tiles .tile:nth-child(2) { animation-delay: .07s; } .tiles .tile:nth-child(3) { animation-delay: .14s; }
.tiles .tile:nth-child(4) { animation-delay: .21s; }
.stepper .on .n { animation: ring 1.8s ease-out infinite; }
.board.pop .found { animation: pop .4s cubic-bezier(.3,1.4,.5,1) both; animation-delay: calc(var(--i) * 14ms); }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
@media (max-width: 640px) { .hero { margin-top: 0; } .hero h1 { font-size: 2.1rem; } .stepper .t { display: none; } }
</style>
"""


HERO = """
<div class="hero">
  <div class="kicker">Computer vision</div>
  <h1>Snap Sudoku</h1>
  <p>Snap a photo of the grid: the missing digits get written right onto the paper.</p>
  <div class="tags"><span>YOLO · detection</span><span>OpenCV · dewarping</span>
  <span>YOLO-cls · reading</span><span>Backtracking · solving</span></div>
</div>
"""


@st.cache_resource(show_spinner="Loading models…")
def load_pipeline(source: str, try_rotations: bool, repair: bool) -> SudokuPipeline:
    if source == "registry":
        return SudokuPipeline.from_registry(TRACKING_URI, try_rotations=try_rotations, repair=repair)
    return SudokuPipeline(*LOCAL_WEIGHTS, try_rotations=try_rotations, repair=repair)


def css_html(theme: str) -> str:
    """Thème Auto : suit le système. Clair / Sombre : forcé, ce qui l'emporte sur la media query."""
    css = CSS.replace("</style>", charts.CHARTS_CSS + "</style>", 1)
    css = css.replace("__LIGHT__", LIGHT_VARS).replace("__DARK__", DARK_VARS)
    forced = {"Light": LIGHT_VARS, "Dark": DARK_VARS}.get(theme)
    if forced:
        css += f"<style>:root {{ {forced} }}</style>"
    # Le tableau éditable est un canvas, que le CSS ne peut pas recolorer : en sombre on inverse ses couleurs
    if theme == "Dark":
        css += f"<style>{DARK_TABLE}</style>"
    elif theme == "Light":
        css += "<style>[data-testid=\"stDataFrame\"] { filter: none; }</style>"
    else:
        css += f"<style>@media (prefers-color-scheme: dark) {{ {DARK_TABLE} }}</style>"
    return css


@st.cache_data(ttl=30, show_spinner=False)
def model_stats() -> dict[str, tuple[str, dict[str, float]]]:
    """Dernières mesures MLflow de chaque modèle : {clé: (nom du run, métriques)}. Vide si MLflow est absent."""
    if not (ROOT / "mlflow.db").exists():
        return {}
    try:
        import mlflow

        client = mlflow.MlflowClient(tracking_uri=TRACKING_URI)
        out = {}
        for key, name in (("cells", "digit_classification"), ("grid", "grid_detection"), ("pipeline", "pipeline_eval")):
            exp = client.get_experiment_by_name(name)
            runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1) if exp else []
            if runs:
                out[key] = (runs[0].info.run_name or "", dict(runs[0].data.metrics))
        return out
    except Exception:  # base verrouillée ou illisible : l'onglet l'indique, l'app continue
        return {}


def pick(metrics: dict[str, float], *names: str):
    return next((metrics[n] for n in names if n in metrics), None)


def models_html() -> str:
    stats = model_stats()
    if not stats:
        return charts.panel("Models", '<p class="note">No MLflow database (mlflow.db) to read: model metrics '
                            "are not available here.</p>")
    cards = []
    if "cells" in stats:
        name, m = stats["cells"]
        perfect = m["real_grids_perfect"] / m["real_grids_total"] if m.get("real_grids_total") else None
        cards.append(charts.model_card_html(
            "Cell classifier", f"YOLO11n-cls · {name}",
            [("Real cells read correctly", pick(m, "real_cell_accuracy")), ("Digits read correctly", pick(m, "real_digit_accuracy")),
             ("Empty cells read correctly", pick(m, "real_empty_accuracy")), ("Grids read without error", perfect),
             ("Validation (synthetic)", pick(m, "metrics/accuracy_top1"))],
            "Measured on the test photos, never seen during training."))
    if "grid" in stats:
        name, m = stats["grid"]
        cards.append(charts.model_card_html(
            "Grid detector", f"YOLO11n · {name}",
            [("mAP50", pick(m, "test_mAP50", "metrics/mAP50B")), ("mAP50-95", pick(m, "test_mAP50_95", "metrics/mAP50-95B")),
             ("Precision", pick(m, "test_precision", "metrics/precisionB")), ("Recall", pick(m, "test_recall", "metrics/recallB"))],
            "mAP on the test set once training is done, on validation otherwise."))
    if "pipeline" in stats:
        name, m = stats["pipeline"]
        rows = [(k.replace("_", " "), v) for k, v in sorted(m.items()) if 0 <= v <= 1][:8]
        cards.append(charts.model_card_html("Full pipeline", f"end to end · {name}", rows,
                                            "Evaluation on the test photos (scripts/evaluate.py)."))
    return "".join(cards)


def show_dashboard(res, grid) -> None:
    """Chiffres et graphiques sur la lecture, les temps et les modèles."""
    st.subheader("Analysis")
    tab_conf, tab_perf, tab_models = st.tabs(["Reading confidence", "Performance", "Models"])
    with tab_conf:
        mean = float(res.confidence.mean())
        st.html(charts.panel("Cell-by-cell confidence", (
            f'<div class="cols"><div>{charts.heatmap_html(res.confidence, res.raw_grid)}</div>'
            f'<div>{charts.gauge_html(mean, "average confidence")}{charts.histogram_html(res.confidence)}</div></div>'),
            "Each cell is read by the classifier, which reports its certainty. Yellow or red cells are the ones "
            "to check; bar height is on a square-root scale."))
    with tab_perf:
        given = sum(v > 0 for row in grid for v in row)
        cases = charts.donut_html([("printed digits", given, "c0"), ("to fill", 81 - given, "c1")], "81", "cells")
        st.html(f'<div class="cols"><div>{charts.panel("Time per step", charts.stack_html(timing_steps(res)))}</div>'
                f'<div>{charts.panel("The 81 cells", cases)}</div></div>')
    with tab_models:
        st.html(models_html())


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_detection(image: np.ndarray, res, n_lines: int = 20, with_bbox: bool = True) -> np.ndarray:
    """bbox YOLO en bleu, grille suivie (100 intersections) en orange.

    `n_lines` limite le nombre de traits dessinés (10 lignes puis 10 colonnes), pour l'animation.
    """
    out = image.copy()
    thick = max(2, image.shape[1] // 400)
    if with_bbox and res.bbox is not None:
        x1, y1, x2, y2 = res.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 120, 0), thick)
    if res.points is not None:
        pts = res.points.astype(np.int32)
        for k in range(10):
            if k < n_lines:
                cv2.polylines(out, [np.ascontiguousarray(pts[k, :]).reshape(-1, 1, 2)], False, (0, 140, 255), thick)
            if 10 + k < n_lines:
                cv2.polylines(out, [np.ascontiguousarray(pts[:, k]).reshape(-1, 1, 2)], False, (0, 140, 255), thick)
    return out


def board_html(grid: list[list[int]], solution, doubtful: set, fixed: set, marks: dict | None = None,
               pop: bool = False) -> str:
    """La grille dessinée : chiffres lus en noir, chiffres trouvés en vert, cases à vérifier surlignées.

    `marks` ({(ligne, col): classe CSS}) surligne des cases pendant les animations ;
    `pop` fait apparaître les chiffres trouvés un à un.
    """
    marks = marks or {}
    cells = []
    for r in range(9):
        for c in range(9):
            value = grid[r][c] or (solution[r][c] if solution else 0) or ""
            classes = []
            if not grid[r][c] and value:
                classes.append("found")
            if (r, c) in marks:
                classes.append(marks[(r, c)])
            elif (r, c) in fixed:
                classes.append("fixed")
            elif (r, c) in doubtful:
                classes.append("doubt")
            cells.append(f'<div class="{" ".join(classes)}" style="--i:{r * 9 + c}">{value}</div>')
    return f'<div class="board{" pop" if pop else ""}">{"".join(cells)}</div>'


def legend_html(solved: bool, has_doubt: bool, has_fixed: bool) -> str:
    parts = ['<b class="l-found">green</b> digit found'] if solved else []
    if has_doubt:
        parts.append('<b class="l-doubt">yellow</b> uncertain reading')
    if has_fixed:
        parts.append('<b class="l-fixed">orange</b> fixed using the sudoku rules')
    return f'<p class="legend">{" · ".join(parts)}</p>' if parts else ""


def describe_repairs(repaired: list) -> str:
    changes = [f"row {r + 1}, column {c + 1}: {a or 'empty'} replaced by {b or 'empty'}"
               for r, c, a, b in repaired]
    return "The raw reading was impossible, so a correction was made (" + " ; ".join(changes) + ")."


STEPS = [
    ("Grid detection", "YOLO boxes the grid (blue), then OpenCV tracks the 20 lines that make it up (orange)."),
    ("Dewarping", "The perspective is corrected: the grid becomes a square seen head-on."),
    ("Reading the digits", "Each cell is cut out, then read by the classifier."),
    ("Solving", "The solver tries a digit (blue) and backtracks (red) when it hits a dead end."),
    ("Result", "The digits found are written on the original photo."),
]


# For each step: (model or technique, input -> output, details)
STEP_INFO = [
    ("YOLO11n <code>detect</code> (Ultralytics), then OpenCV (adaptive threshold + contours)",
     "full photo → grid box → 4 corners → 20 lines",
     "A single class, “sudoku”, trained on 156 annotated photos (wichtounet + rg1990). The YOLO box "
     "narrows the area where OpenCV looks for the quadrilateral; without a box, OpenCV searches the whole image."),
    ("4-point homography (<code>getPerspectiveTransform</code>), then fine dewarping by line tracking",
     "4 corners → 450×450 px square grid",
     "A homography is not enough on a curved page: we track the 20 lines (polynomials) and remap "
     "each cell between its 4 intersections (<code>cv2.remap</code>). The 4 orientations are then tested."),
    ("YOLO11n <code>cls</code> (Ultralytics), 10 classes: empty, 1 to 9",
     "81 cells of 64×64 px → 81 digits + confidences",
     "Trained on font digits (Chars74K) rendered like photographed cells, plus real labelled cells. "
     "If the grid read does not have a unique solution, up to 2 of the 8 least certain cells are replaced "
     "by the classifier’s 2nd or 3rd choice."),
    ("Backtracking with MRV heuristic (<code>solver.py</code>)",
     "grid read → completed grid",
     "We fill the most constrained empty cell and backtrack on a dead end. The grid is validated "
     "before the search, and its uniqueness is checked with <code>count_solutions</code>."),
    ("OpenCV rendering (<code>cv2.putText</code>) on the original photo",
     "solved grid + 100 intersections → annotated photo",
     "Each digit is centred on the mean of its cell’s 4 intersections, sized in proportion to the "
     "cell: the perspective and curvature of the paper are respected."),
]


def measures(step: int, res, extra: dict) -> list[tuple[str, str]]:
    """Real measurements of this photo for the step: (value, label)."""
    t = res.timings
    ms = lambda k: f"{t[k] * 1000:.0f} ms" if k in t else "—"
    if step == 0:
        return [(f"{res.bbox_conf:.0%}" if res.bbox is not None else "OpenCV", "box confidence"),
                (str(max(res.n_grids, 1)), "grid(s) detected"), (ms("detect"), "detection")]
    if step == 1:
        return [(f"{90 * res.rotation}°", "rotation kept"), (f"{res.flat.shape[1]} px", "grid side"),
                (ms("warp"), "dewarping")]
    if step == 2:
        digits = sum(v > 0 for row in res.raw_grid for v in row)
        doubt = int((res.confidence < LOW_CONF).sum())
        return [(str(digits), "digits read"), (str(doubt), "uncertain cells"),
                (str(len(res.repaired)), "cell(s) corrected"), (ms("classify"), "reading")]
    if step == 3:
        return [(str(sum(row.count(0) for row in res.grid)), "empty cells"), (str(extra.get("tries", 0)), "tries"),
                (str(extra.get("undos", 0)), "backtracks"), (ms("solve"), "solving")]
    return [(str(sum(row.count(0) for row in res.grid)), "digits added"), (ms("total"), "total time")]


def timing_steps(res) -> list[tuple[str, float]]:
    t = res.timings
    return [("Detection", t.get("detect", 0)), ("Dewarping", t.get("warp", 0)),
            ("Reading", t.get("classify", 0)), ("Solving", t.get("solve", 0))]


def step_chart_html(step: int, res, extra: dict) -> str:
    """The chart that illustrates the step, under the info panel."""
    if step == 0 and res.bbox is not None:
        return (f'<div class="cols">{charts.gauge_html(res.bbox_conf, "YOLO confidence")}'
                '<p class="note">The detector’s certainty about the box it found. '
                "Below 25%, the box is discarded and OpenCV searches alone.</p></div>")
    if step == 2:
        return (f'<div class="cols"><div>{charts.heatmap_html(res.confidence, res.raw_grid)}</div>'
                f'<div>{charts.histogram_html(res.confidence)}<p class="note">Classifier confidence over the 81 cells '
                "(bar height on a square-root scale).</p></div></div>")
    if step == 3 and extra:
        kept = extra["tries"] - extra["undos"]
        return charts.donut_html([("digits placed (final)", kept, "c1"), ("abandoned guesses", extra["undos"], "c3")],
                                 str(extra["tries"]), "tries")
    if step == 4:
        return charts.stack_html(timing_steps(res))
    return ""


def info_html(step: int, res, extra: dict | None = None) -> str:
    model, flow, detail = STEP_INFO[step]
    chart = step_chart_html(step, res, extra or {})
    chart_html = f'<div class="info-chart">{chart}</div>' if chart else ""
    return (f'<div class="info"><div><h4>Under the hood</h4><dl>'
            f'<dt>Technique</dt><dd>{model}</dd><dt>Input → output</dt><dd>{flow}</dd>'
            f'<dt>In detail</dt><dd>{detail}</dd></dl></div>'
            f'<div><h4>On this photo</h4>{tiles_html(measures(step, res, extra or {}), good_first=True)}</div>'
            f'{chart_html}</div>')


SHORT_TITLES = ["Detection", "Dewarping", "Reading", "Solving", "Result"]


def stepper_html(step: int, n_steps: int) -> str:
    pills = []
    for i in range(n_steps):
        state = "on" if i == step else "done" if i < step else ""
        mark = "✓" if i < step else i + 1
        pills.append(f'<div class="s {state}"><span class="n">{mark}</span><span class="t">{SHORT_TITLES[i]}</span></div>')
    return f'<div class="stepper">{"".join(pills)}</div>'


def tiles_html(items: list[tuple[str, str]], good_first: bool = False) -> str:
    """Row of key figures: (value, label)."""
    tiles = [f'<div class="tile{" good" if good_first and i == 0 else ""}"><b>{charts.counter(v)}</b><span>{k}</span></div>'
             for i, (v, k) in enumerate(items)]
    return f'<div class="tiles">{"".join(tiles)}</div>'


def timing_tiles(res, n_found: int) -> str:
    t = res.timings
    return tiles_html([(str(n_found), "digits added"), (f"{t['total'] * 1000:.0f} ms", "total time"),
                       (f"{res.bbox_conf:.0%}" if res.bbox is not None else "OpenCV", "box confidence"),
                       (f"{float(res.confidence.mean()):.0%}", "reading confidence")], good_first=True)


def go(delta: int) -> None:
    st.session_state.step += delta


def replay() -> None:
    st.session_state.played.discard(st.session_state.step)


def step_detection(image, res, anim, pause) -> None:
    slot = st.empty()
    if anim:
        slot.image(to_rgb(image), width="stretch")
        pause(0.5)
        slot.image(to_rgb(draw_detection(image, res, n_lines=0)), width="stretch")
        pause(0.5)
    for n in (range(1, 21) if anim else [20]):
        slot.image(to_rgb(draw_detection(image, res, n_lines=n)), width="stretch")
        pause(0.08)


def step_warp(image, res, anim, pause) -> None:
    left, right = st.columns(2)
    left.image(to_rgb(draw_detection(image, res)), width="stretch", caption="Original photo")
    slot = right.empty()
    h = res.flat.shape[0]
    for n in (range(1, 11) if anim else [10]):
        shown = np.full_like(res.flat, 255)
        shown[: h * n // 10] = res.flat[: h * n // 10]
        slot.image(to_rgb(shown), width="stretch")
        pause(0.12)
    right.caption("Dewarped grid" + (f" (rotated by {90 * res.rotation}°)." if res.rotation else "."))


def step_reading(res, doubtful, fixed, anim, pause) -> None:
    left, right = st.columns([3, 2], gap="large")
    flat_slot, board_slot = left.empty(), right.empty()
    cell = res.flat.shape[0] / 9
    if anim:
        read = [[0] * 9 for _ in range(9)]
        for r in range(9):
            for c in range(9):
                read[r][c] = res.raw_grid[r][c]
                out = res.flat.copy()
                p1 = (int(c * cell), int(r * cell))
                p2 = (int((c + 1) * cell), int((r + 1) * cell))
                cv2.rectangle(out, p1, p2, (255, 120, 0), max(2, res.flat.shape[0] // 150))
                flat_slot.image(to_rgb(out), width="stretch")
                board_slot.html(board_html(read, None, set(), set(), {(r, c): "try"}))
                pause(0.03)
    flat_slot.image(to_rgb(res.flat), width="stretch")
    grid = res.grid if res.grid is not None else res.raw_grid
    board_slot.html(board_html(grid, None, doubtful, fixed) + legend_html(False, bool(doubtful), bool(fixed)))
    if res.repaired:
        right.caption(describe_repairs(res.repaired))
    if res.status != "solved":
        right.error("This reading does not give a solvable grid. Switch to “Full solution” "
                    "to fix the misread cell by hand.")


def step_solving(res, anim, pause) -> dict:
    left, right = st.columns([3, 2], gap="large")
    events = solve_trace(res.grid)
    tries = sum(e[0] == "try" for e in events)
    undos = len(events) - tries
    board_slot, stats = left.empty(), right.empty()
    if anim:
        cur = [row[:] for row in res.grid]
        stride = max(1, len(events) // 200)  # au plus ~200 images, quelle que soit la difficulté
        n_try = n_undo = 0
        for i, (kind, r, c, v) in enumerate(events):
            cur[r][c] = v if kind == "try" else 0
            n_try += kind == "try"
            n_undo += kind == "undo"
            if i % stride == 0:
                board_slot.html(board_html(res.grid, cur, set(), set(), {(r, c): kind}))
                stats.markdown(f"**{n_try}** tries · **{n_undo}** backtracks")
                pause(0.06)
    board_slot.html(board_html(res.grid, res.solution, set(), set(), pop=anim))
    stats.markdown(f"**{tries}** tries · **{undos}** backtracks")
    right.caption("Each backtrack is an abandoned guess: the harder the grid, the more there are.")
    return {"tries": tries, "undos": undos}


def step_result(image, res, anim) -> None:
    solved_photo = render_solution(image, res)
    left, right = st.columns([3, 2], gap="large")
    left.image(to_rgb(solved_photo), width="stretch")
    right.html(timing_tiles(res, sum(row.count(0) for row in res.grid)))
    right.html(board_html(res.grid, res.solution, set(), set(), pop=anim) + legend_html(True, False, False))
    ok, jpeg = cv2.imencode(".jpg", solved_photo)
    if ok:
        right.download_button("Download the photo", jpeg.tobytes(), "sudoku_solved.jpg", "image/jpeg")


def show_steps(image: np.ndarray, res, doubtful: set, fixed: set) -> None:
    """Les étapes du pipeline une par une ; l'animation d'une étape ne se joue qu'à sa première visite."""
    speed = st.sidebar.select_slider("Animation speed", [0.5, 1, 2, 4], value=1, format_func=lambda s: f"×{s}")
    step = st.session_state.step
    n_steps = len(STEPS) if res.status == "solved" else 3
    anim = step not in st.session_state.played

    def pause(seconds: float) -> None:
        if anim:
            time.sleep(seconds / speed)

    extra: dict = {}
    title, text = STEPS[step]
    st.html(stepper_html(step, n_steps))
    st.subheader(title)
    st.html(f'<p class="step-text">{text}</p>')
    if step == 0:
        step_detection(image, res, anim, pause)
    elif step == 1:
        step_warp(image, res, anim, pause)
    elif step == 2:
        step_reading(res, doubtful, fixed, anim, pause)
    elif step == 3:
        extra = step_solving(res, anim, pause)
    else:
        step_result(image, res, anim)
    st.html(info_html(step, res, extra))
    st.session_state.played.add(step)

    prev, again, nxt, _ = st.columns([1, 1, 1.4, 3])
    prev.button("← Previous", on_click=go, args=(-1,), disabled=step == 0, width="stretch")
    again.button("↻ Replay", on_click=replay, width="stretch")
    nxt.button("Next step →", on_click=go, args=(1,), type="primary",
               disabled=step >= n_steps - 1, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Snap Sudoku", initial_sidebar_state="collapsed", layout="wide")
    _, picker = st.columns([5, 2])
    with picker:
        theme = st.segmented_control("Theme", ["Auto", "Light", "Dark"], default="Auto",
                                     label_visibility="collapsed") or "Auto"
    st.html(css_html(theme))
    st.html(HERO)

    with st.sidebar:
        st.subheader("Demo options")
        sources = ["registry", "local"]
        source = st.radio("Models", sources, index=sources.index(os.environ.get("SUDOKU_MODEL_SOURCE", "registry")),
                          format_func=lambda s: "MLflow Registry (champion)" if s == "registry" else "Local files")
        try_rotations = st.checkbox("Test the 4 orientations", value=True,
                                    help="Without this option, a grid photographed sideways is misread.")
        repair = st.checkbox("Fix using the sudoku rules", value=True,
                             help="Without this option, a single misread cell is enough to block solving.")

    try:
        pipe = load_pipeline(source, try_rotations, repair)
    except Exception as exc:  # registry vide, poids absents...
        st.error(f"Could not load the models ({source}): {exc}")
        st.stop()

    mode = st.segmented_control("Photo source", ["Upload", "Camera"], default="Upload",
                                label_visibility="collapsed") or "Upload"
    upload = st.file_uploader("Sudoku photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed") \
        if mode == "Upload" else st.camera_input("Sudoku photo", label_visibility="collapsed")
    if upload is None:
        st.caption("📷 For a good reading: the whole grid in frame, and no shadow on the digits.")
        return

    data = upload.getvalue()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("This file does not open as an image.")
        return
    key = (hashlib.md5(data).hexdigest(), source, try_rotations, repair)
    if st.session_state.get("key") != key:  # on ne relance le pipeline que si la photo change
        with st.spinner("Reading the grid…"):
            st.session_state.result = pipe.run(image)
        st.session_state.key = key
        st.session_state.step = 0  # nouvelle photo : retour à la première étape
        st.session_state.played = set()
    res = st.session_state.result

    if res.status == "no_grid":
        st.error("No grid found in this photo. Try moving closer and framing the whole grid.")
        st.image(to_rgb(image), width="stretch")
        return

    base = res.grid if res.grid is not None else res.raw_grid
    doubtful = {(r, c) for r in range(9) for c in range(9) if res.confidence[r, c] < LOW_CONF}
    fixed = {(r, c) for r, c, _, _ in res.repaired}

    view = st.segmented_control("View", ["Full solution", "Step by step"], default="Full solution",
                                label_visibility="collapsed") or "Full solution"
    if view == "Step by step":
        show_steps(image, res, doubtful, fixed)
        return

    # La correction se fait plus bas mais décide de ce qui s'affiche ici
    result_area = st.container()

    with st.expander("Fix a misread cell", expanded=res.status != "solved"):
        st.caption("Leave a cell empty if it has no digit. The solution is recomputed as soon as you change a cell.")
        table = pd.DataFrame([[str(v) if v else "" for v in row] for row in base],
                             index=range(1, 10), columns=[str(c) for c in range(1, 10)])
        columns = {str(c): st.column_config.TextColumn(str(c), validate="^[1-9]?$", width="small")
                   for c in range(1, 10)}
        edited = st.data_editor(table, column_config=columns, width="content",
                                key=f"editor_{'_'.join(map(str, key))}")
    grid = [[int(v) if v else 0 for v in row] for row in edited.values.tolist()]

    shown = copy.copy(res)
    if grid != res.grid:  # l'utilisateur a corrigé une case : on résout sa grille
        shown.grid = grid
        shown.solution = solve(grid) if is_valid_grid(grid) else None

    with result_area:
        left, right = st.columns([3, 2], gap="large")
        if shown.solution is None:
            msg = "a digit appears twice in a row, column or block" if not is_valid_grid(grid) \
                else "it has no solution"
            left.image(to_rgb(draw_detection(image, res)), width="stretch")
            right.error(f"This grid cannot be solved: {msg}. A cell is probably misread, "
                        "you can fix it in the table below.")
        else:
            solved_photo = render_solution(image, shown)
            left.image(to_rgb(solved_photo), width="stretch")
            n_found = sum(row.count(0) for row in grid)
            right.html(timing_tiles(res, n_found))
            if count_solutions(grid) > 1:
                right.warning("This grid has several solutions, so this one may not be the one in the paper. "
                              "A cell is probably misread.")
        right.html(board_html(grid, shown.solution, doubtful, fixed)
                   + legend_html(shown.solution is not None, bool(doubtful), bool(fixed)))
        if res.repaired and grid == res.grid:
            right.caption(describe_repairs(res.repaired))
        if shown.solution is not None:
            ok, jpeg = cv2.imencode(".jpg", solved_photo)
            if ok:
                right.download_button("Download the photo", jpeg.tobytes(), "sudoku_solved.jpg", "image/jpeg")

    show_dashboard(res, grid)

    with st.expander("See how the photo was read"):
        col1, col2 = st.columns(2)
        col1.image(to_rgb(draw_detection(image, res)), width="stretch",
                   caption="Blue: the box found by YOLO. Orange: the 20 lines tracked by OpenCV.")
        col2.image(to_rgb(res.flat), width="stretch",
                   caption="Dewarped grid, then cut into 81 cells"
                           + (f" (rotated by {90 * res.rotation}°)." if res.rotation else "."))
        t = res.timings
        st.caption(f"Detection {t['detect']*1000:.0f} ms · dewarping {t['warp']*1000:.0f} ms · "
                   f"reading {t['classify']*1000:.0f} ms · total {t['total']*1000:.0f} ms"
                   + (f" · box confidence {res.bbox_conf:.2f}" if res.bbox is not None else " · box found by OpenCV alone")
                   + (f" · {res.n_grids} grids in the photo, this is the first one solved" if res.n_grids > 1 else ""))


if __name__ == "__main__":
    main()
