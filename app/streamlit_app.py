"""Application Streamlit : photo d'un sudoku papier -> solution écrite sur la photo.

Lancement :
    .venv/Scripts/streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sudoku_cv.pipeline import SudokuPipeline, render_solution  # noqa: E402
from sudoku_cv.solver import count_solutions, is_valid_grid, solve  # noqa: E402

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
LOCAL_WEIGHTS = (ROOT / "runs/grid/grid_v1/weights/best.pt", ROOT / "runs/cells/cells_v2/weights/best.pt")
LOW_CONF = 0.8  # en dessous, la case est signalée comme douteuse

CSS = """
<style>
.block-container { max-width: 1080px; padding-top: 2.5rem; }
h1 { font-size: 2.1rem !important; padding-bottom: 0 !important; }
.board {
  display: grid; box-sizing: border-box; aspect-ratio: 1; width: 100%; max-width: 432px;
  grid-template-columns: repeat(9, minmax(0, 1fr)); grid-template-rows: repeat(9, minmax(0, 1fr));
  border: 2px solid #1F1D1A; background: #FFFEFA;
  font-size: clamp(1rem, 4.2vw, 1.4rem); font-variant-numeric: tabular-nums;
}
.board div { box-sizing: border-box; display: flex; align-items: center; justify-content: center; border: 1px solid #CFCBBF; }
.board div:nth-child(9n+3), .board div:nth-child(9n+6) { border-right: 2px solid #1F1D1A; }
.board div:nth-child(n+19):nth-child(-n+27), .board div:nth-child(n+46):nth-child(-n+54) { border-bottom: 2px solid #1F1D1A; }
.board .found { color: #1B6E3C; }
.board .doubt { background: #F6E3B4; }
.board .fixed { background: #F3CDBB; }
.board .try { background: #CFE3F6; }
.board .undo { background: #F2B8B5; }
.legend { font-size: .85rem; color: #6A665C; margin-top: .5rem; }
.legend b { font-weight: normal; padding: 0 .35em; }
.legend .l-found { color: #1B6E3C; }
.legend .l-doubt { background: #F6E3B4; }
.legend .l-fixed { background: #F3CDBB; }
</style>
"""


@st.cache_resource(show_spinner="Chargement des modèles…")
def load_pipeline(source: str, try_rotations: bool, repair: bool) -> SudokuPipeline:
    if source == "registry":
        return SudokuPipeline.from_registry(TRACKING_URI, try_rotations=try_rotations, repair=repair)
    return SudokuPipeline(*LOCAL_WEIGHTS, try_rotations=try_rotations, repair=repair)


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


def board_html(grid: list[list[int]], solution, doubtful: set, fixed: set, marks: dict | None = None) -> str:
    """La grille dessinée : chiffres lus en noir, chiffres trouvés en vert, cases à vérifier surlignées.

    `marks` ({(ligne, col): classe CSS}) surligne des cases pendant les animations.
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
            cells.append(f'<div class="{" ".join(classes)}">{value}</div>')
    return f'<div class="board">{"".join(cells)}</div>'


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


def main() -> None:
    st.set_page_config(page_title="Sudoku papier", initial_sidebar_state="collapsed", layout="wide")
    st.html(CSS)
    st.title("Sudoku papier")
    st.write("Prenez la grille en photo : les chiffres manquants s'écrivent dessus.")

    with st.sidebar:
        st.subheader("Options de démo")
        source = st.radio("Modèles", ["registry", "local"],
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

    mode = st.radio("Source de la photo", ["Importer", "Caméra"], horizontal=True, label_visibility="collapsed")
    upload = st.file_uploader("Photo du sudoku", type=["jpg", "jpeg", "png"], label_visibility="collapsed") \
        if mode == "Importer" else st.camera_input("Photo du sudoku", label_visibility="collapsed")
    if upload is None:
        st.caption("Pour une bonne lecture : la grille entière dans le cadre, et pas d'ombre sur les chiffres.")
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
    res = st.session_state.result

    if res.status == "no_grid":
        st.error("Pas de grille trouvée sur cette photo. Essayez de vous rapprocher et de cadrer la grille entière.")
        st.image(to_rgb(image), width="stretch")
        return

    # La correction se fait plus bas mais décide de ce qui s'affiche ici
    result_area = st.container()

    base = res.grid if res.grid is not None else res.raw_grid
    doubtful = {(r, c) for r in range(9) for c in range(9) if res.confidence[r, c] < LOW_CONF}
    fixed = {(r, c) for r, c, _, _ in res.repaired}

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
            right.subheader(f"{n_found} chiffres ajoutés")
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
