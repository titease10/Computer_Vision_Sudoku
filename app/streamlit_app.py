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
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-pressed="true"] { background: var(--green-soft); color: var(--green); border-color: var(--green); }
[data-testid="stBaseButton-segmented_controlActive"] { background: var(--green-soft); color: var(--green); border-color: var(--green); }
[data-testid="stFileUploaderDropzone"] button, .stButton > button, .stDownloadButton > button { background: var(--card); color: var(--ink); }
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span { color: var(--muted); }
[data-testid="stSpinner"], [data-testid="stSpinner"] * { background: transparent; color: var(--ink); }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--ink); }
[data-testid="stSidebar"] * { color: var(--ink); }
.stButton > button:disabled { opacity: .45; }
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
.tile span { font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
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
  <div class="kicker">Vision par ordinateur</div>
  <h1>Sudoku papier</h1>
  <p>Prenez la grille en photo : les chiffres manquants s'écrivent dessus, directement sur le papier.</p>
  <div class="tags"><span>YOLO · détection</span><span>OpenCV · redressement</span>
  <span>YOLO-cls · lecture</span><span>Backtracking · résolution</span></div>
</div>
"""


@st.cache_resource(show_spinner="Chargement des modèles…")
def load_pipeline(source: str, try_rotations: bool, repair: bool) -> SudokuPipeline:
    if source == "registry":
        return SudokuPipeline.from_registry(TRACKING_URI, try_rotations=try_rotations, repair=repair)
    return SudokuPipeline(*LOCAL_WEIGHTS, try_rotations=try_rotations, repair=repair)


def css_html(theme: str) -> str:
    """Thème Auto : suit le système. Clair / Sombre : forcé, ce qui l'emporte sur la media query."""
    css = CSS.replace("__LIGHT__", LIGHT_VARS).replace("__DARK__", DARK_VARS)
    forced = {"Clair": LIGHT_VARS, "Sombre": DARK_VARS}.get(theme)
    return css + (f"<style>:root {{ {forced} }}</style>" if forced else "")


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
    parts = ['<b class="l-found">vert</b> chiffre trouvé'] if solved else []
    if has_doubt:
        parts.append('<b class="l-doubt">jaune</b> lecture peu sûre')
    if has_fixed:
        parts.append('<b class="l-fixed">orange</b> corrigé par les règles du sudoku')
    return f'<p class="legend">{" · ".join(parts)}</p>' if parts else ""


def describe_repairs(repaired: list) -> str:
    changes = [f"ligne {r + 1}, colonne {c + 1} : {a or 'vide'} remplacé par {b or 'vide'}"
               for r, c, a, b in repaired]
    return "La lecture brute était impossible, une correction a été faite (" + " ; ".join(changes) + ")."


STEPS = [
    ("Détection de la grille", "YOLO encadre la grille (bleu), puis OpenCV suit les 20 traits qui la composent (orange)."),
    ("Redressement", "La perspective est corrigée : la grille devient un carré vu de face."),
    ("Lecture des chiffres", "Chaque case est découpée puis lue par le classifieur."),
    ("Résolution", "Le solveur essaie un chiffre (bleu), et revient en arrière (rouge) quand il tombe sur une impasse."),
    ("Résultat", "Les chiffres trouvés sont écrits sur la photo d'origine."),
]


# Pour chaque étape : (modèle ou technique, entrée -> sortie, détails)
STEP_INFO = [
    ("YOLO11n <code>detect</code> (Ultralytics), puis OpenCV (seuil adaptatif + contours)",
     "photo entière → cadre de la grille → 4 coins → 20 traits",
     "Une seule classe, « sudoku », entraînée sur 156 photos annotées (wichtounet + rg1990). Le cadre YOLO "
     "restreint la zone où OpenCV cherche le quadrilatère ; sans cadre, OpenCV cherche sur toute l'image."),
    ("Homographie 4 points (<code>getPerspectiveTransform</code>), puis redressement fin par suivi des traits",
     "4 coins → grille carrée 450×450 px",
     "Une homographie ne suffit pas sur une page courbée : on suit les 20 traits (polynômes) et on remappe "
     "chaque case entre ses 4 intersections (<code>cv2.remap</code>). Les 4 orientations sont ensuite testées."),
    ("YOLO11n <code>cls</code> (Ultralytics), 10 classes : vide, 1 à 9",
     "81 cases de 64×64 px → 81 chiffres + confiances",
     "Entraîné sur des chiffres de polices (Chars74K) rendus comme des cases photographiées, plus de vraies cases "
     "étiquetées. Si la grille lue n'a pas une solution unique, on remplace jusqu'à 2 des 8 cases les moins sûres "
     "par le 2ᵉ ou 3ᵉ choix du classifieur."),
    ("Backtracking avec heuristique MRV (<code>solver.py</code>)",
     "grille lue → grille complète",
     "On remplit la case vide la plus contrainte, et on revient en arrière sur une impasse. La grille est validée "
     "avant la recherche, et son unicité vérifiée avec <code>count_solutions</code>."),
    ("Rendu OpenCV (<code>cv2.putText</code>) sur la photo d'origine",
     "grille résolue + 100 intersections → photo annotée",
     "Chaque chiffre est centré sur la moyenne des 4 intersections de sa case, avec une taille proportionnelle à la "
     "case : la perspective et la courbure du papier sont respectées."),
]


def measures(step: int, res, extra: dict) -> list[tuple[str, str]]:
    """Mesures réelles de cette photo pour l'étape : (valeur, légende)."""
    t = res.timings
    ms = lambda k: f"{t[k] * 1000:.0f} ms" if k in t else "—"
    if step == 0:
        return [(f"{res.bbox_conf:.0%}" if res.bbox is not None else "OpenCV", "confiance du cadre"),
                (str(max(res.n_grids, 1)), "grille(s) détectée(s)"), (ms("detect"), "détection")]
    if step == 1:
        return [(f"{90 * res.rotation}°", "rotation retenue"), (f"{res.flat.shape[1]} px", "côté de la grille"),
                (ms("warp"), "redressement")]
    if step == 2:
        digits = sum(v > 0 for row in res.raw_grid for v in row)
        doubt = int((res.confidence < LOW_CONF).sum())
        return [(str(digits), "chiffres lus"), (str(doubt), "cases peu sûres"),
                (str(len(res.repaired)), "case(s) corrigée(s)"), (ms("classify"), "lecture")]
    if step == 3:
        return [(str(sum(row.count(0) for row in res.grid)), "cases vides"), (str(extra.get("tries", 0)), "essais"),
                (str(extra.get("undos", 0)), "retours en arrière"), (ms("solve"), "résolution")]
    return [(str(sum(row.count(0) for row in res.grid)), "chiffres ajoutés"), (ms("total"), "temps total")]


def info_html(step: int, res, extra: dict | None = None) -> str:
    model, flow, detail = STEP_INFO[step]
    return (f'<div class="info"><div><h4>Sous le capot</h4><dl>'
            f'<dt>Technique</dt><dd>{model}</dd><dt>Entrée → sortie</dt><dd>{flow}</dd>'
            f'<dt>En détail</dt><dd>{detail}</dd></dl></div>'
            f'<div><h4>Sur cette photo</h4>{tiles_html(measures(step, res, extra or {}), good_first=True)}</div></div>')


SHORT_TITLES = ["Détection", "Redressement", "Lecture", "Résolution", "Résultat"]


def stepper_html(step: int, n_steps: int) -> str:
    pills = []
    for i in range(n_steps):
        state = "on" if i == step else "done" if i < step else ""
        mark = "✓" if i < step else i + 1
        pills.append(f'<div class="s {state}"><span class="n">{mark}</span><span class="t">{SHORT_TITLES[i]}</span></div>')
    return f'<div class="stepper">{"".join(pills)}</div>'


def tiles_html(items: list[tuple[str, str]], good_first: bool = False) -> str:
    """Rangée de chiffres clés : (valeur, légende)."""
    tiles = [f'<div class="tile{" good" if good_first and i == 0 else ""}"><b>{v}</b><span>{k}</span></div>'
             for i, (v, k) in enumerate(items)]
    return f'<div class="tiles">{"".join(tiles)}</div>'


def timing_tiles(res, n_found: int) -> str:
    t = res.timings
    return tiles_html([(str(n_found), "chiffres ajoutés"), (f"{t['total'] * 1000:.0f} ms", "temps total"),
                       (f"{res.bbox_conf:.0%}" if res.bbox is not None else "OpenCV", "confiance du cadre"),
                       (f"{float(res.confidence.mean()):.0%}", "confiance de lecture")], good_first=True)


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
    left.image(to_rgb(draw_detection(image, res)), width="stretch", caption="Photo d'origine")
    slot = right.empty()
    h = res.flat.shape[0]
    for n in (range(1, 11) if anim else [10]):
        shown = np.full_like(res.flat, 255)
        shown[: h * n // 10] = res.flat[: h * n // 10]
        slot.image(to_rgb(shown), width="stretch")
        pause(0.12)
    right.caption("Grille redressée" + (f" (tournée de {90 * res.rotation}°)." if res.rotation else "."))


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
        right.error("Cette lecture ne donne pas une grille résoluble. Passez en « Résolution complète » "
                    "pour corriger la case mal lue à la main.")


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
                stats.markdown(f"**{n_try}** essais · **{n_undo}** retours en arrière")
                pause(0.06)
    board_slot.html(board_html(res.grid, res.solution, set(), set(), pop=anim))
    stats.markdown(f"**{tries}** essais · **{undos}** retours en arrière")
    right.caption("Chaque retour en arrière est une hypothèse abandonnée : plus la grille est difficile, plus il y en a.")
    return {"tries": tries, "undos": undos}


def step_result(image, res, anim) -> None:
    solved_photo = render_solution(image, res)
    left, right = st.columns([3, 2], gap="large")
    left.image(to_rgb(solved_photo), width="stretch")
    right.html(timing_tiles(res, sum(row.count(0) for row in res.grid)))
    right.html(board_html(res.grid, res.solution, set(), set(), pop=anim) + legend_html(True, False, False))
    ok, jpeg = cv2.imencode(".jpg", solved_photo)
    if ok:
        right.download_button("Télécharger la photo", jpeg.tobytes(), "sudoku_resolu.jpg", "image/jpeg")


def show_steps(image: np.ndarray, res, doubtful: set, fixed: set) -> None:
    """Les étapes du pipeline une par une ; l'animation d'une étape ne se joue qu'à sa première visite."""
    speed = st.sidebar.select_slider("Vitesse de l'animation", [0.5, 1, 2, 4], value=1, format_func=lambda s: f"×{s}")
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
    prev.button("← Précédent", on_click=go, args=(-1,), disabled=step == 0, width="stretch")
    again.button("↻ Rejouer", on_click=replay, width="stretch")
    nxt.button("Étape suivante →", on_click=go, args=(1,), type="primary",
               disabled=step >= n_steps - 1, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Sudoku papier", initial_sidebar_state="collapsed", layout="wide")
    _, picker = st.columns([5, 2])
    with picker:
        theme = st.segmented_control("Thème", ["Auto", "Clair", "Sombre"], default="Auto",
                                     label_visibility="collapsed") or "Auto"
    st.html(css_html(theme))
    st.html(HERO)

    with st.sidebar:
        st.subheader("Options de démo")
        sources = ["registry", "local"]
        source = st.radio("Modèles", sources, index=sources.index(os.environ.get("SUDOKU_MODEL_SOURCE", "registry")),
                          format_func=lambda s: "MLflow Registry (champion)" if s == "registry" else "Fichiers locaux")
        try_rotations = st.checkbox("Tester les 4 orientations", value=True,
                                    help="Sans cette option, une grille photographiée de côté est mal lue.")
        repair = st.checkbox("Corriger avec les règles du sudoku", value=True,
                             help="Sans cette option, une seule case mal lue suffit à bloquer la résolution.")

    try:
        pipe = load_pipeline(source, try_rotations, repair)
    except Exception as exc:  # registry vide, poids absents...
        st.error(f"Impossible de charger les modèles ({source}) : {exc}")
        st.stop()

    mode = st.segmented_control("Source de la photo", ["Importer", "Caméra"], default="Importer",
                                label_visibility="collapsed") or "Importer"
    upload = st.file_uploader("Photo du sudoku", type=["jpg", "jpeg", "png"], label_visibility="collapsed") \
        if mode == "Importer" else st.camera_input("Photo du sudoku", label_visibility="collapsed")
    if upload is None:
        st.caption("📷 Pour une bonne lecture : la grille entière dans le cadre, et pas d'ombre sur les chiffres.")
        return

    data = upload.getvalue()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("Ce fichier ne s'ouvre pas comme une image.")
        return
    key = (hashlib.md5(data).hexdigest(), source, try_rotations, repair)
    if st.session_state.get("key") != key:  # on ne relance le pipeline que si la photo change
        with st.spinner("Lecture de la grille…"):
            st.session_state.result = pipe.run(image)
        st.session_state.key = key
        st.session_state.step = 0  # nouvelle photo : retour à la première étape
        st.session_state.played = set()
    res = st.session_state.result

    if res.status == "no_grid":
        st.error("Pas de grille trouvée sur cette photo. Essayez de vous rapprocher et de cadrer la grille entière.")
        st.image(to_rgb(image), width="stretch")
        return

    base = res.grid if res.grid is not None else res.raw_grid
    doubtful = {(r, c) for r in range(9) for c in range(9) if res.confidence[r, c] < LOW_CONF}
    fixed = {(r, c) for r, c, _, _ in res.repaired}

    view = st.segmented_control("Affichage", ["Résolution complète", "Pas à pas"], default="Résolution complète",
                                label_visibility="collapsed") or "Résolution complète"
    if view == "Pas à pas":
        show_steps(image, res, doubtful, fixed)
        return

    # La correction se fait plus bas mais décide de ce qui s'affiche ici
    result_area = st.container()

    with st.expander("Corriger une case mal lue", expanded=res.status != "solved"):
        st.caption("Laissez vide une case sans chiffre. La solution se recalcule dès que vous changez une case.")
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
            msg = "un chiffre apparaît deux fois dans une ligne, une colonne ou un bloc" if not is_valid_grid(grid) \
                else "elle n'a aucune solution"
            left.image(to_rgb(draw_detection(image, res)), width="stretch")
            right.error(f"Grille impossible à résoudre : {msg}. Une case est sans doute mal lue, "
                        "vous pouvez la corriger dans le tableau ci-dessous.")
        else:
            solved_photo = render_solution(image, shown)
            left.image(to_rgb(solved_photo), width="stretch")
            n_found = sum(row.count(0) for row in grid)
            right.html(timing_tiles(res, n_found))
            if count_solutions(grid) > 1:
                right.warning("Cette grille a plusieurs solutions, celle-ci n'est peut-être pas celle du journal. "
                              "Une case est sans doute mal lue.")
        right.html(board_html(grid, shown.solution, doubtful, fixed)
                   + legend_html(shown.solution is not None, bool(doubtful), bool(fixed)))
        if res.repaired and grid == res.grid:
            right.caption(describe_repairs(res.repaired))
        if shown.solution is not None:
            ok, jpeg = cv2.imencode(".jpg", solved_photo)
            if ok:
                right.download_button("Télécharger la photo", jpeg.tobytes(), "sudoku_resolu.jpg", "image/jpeg")

    with st.expander("Voir comment la photo a été lue"):
        col1, col2 = st.columns(2)
        col1.image(to_rgb(draw_detection(image, res)), width="stretch",
                   caption="Bleu : le cadre trouvé par YOLO. Orange : les 20 traits suivis par OpenCV.")
        col2.image(to_rgb(res.flat), width="stretch",
                   caption="Grille redressée, découpée ensuite en 81 cases"
                           + (f" (tournée de {90 * res.rotation}°)." if res.rotation else "."))
        t = res.timings
        st.caption(f"Détection {t['detect']*1000:.0f} ms · redressement {t['warp']*1000:.0f} ms · "
                   f"lecture {t['classify']*1000:.0f} ms · total {t['total']*1000:.0f} ms"
                   + (f" · confiance du cadre {res.bbox_conf:.2f}" if res.bbox is not None else " · cadre trouvé par OpenCV seul")
                   + (f" · {res.n_grids} grilles sur la photo, celle-ci est la première résolue" if res.n_grids > 1 else ""))


if __name__ == "__main__":
    main()
