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


@st.cache_resource(show_spinner="Chargement des modèles…")
def load_pipeline(source: str, try_rotations: bool, repair: bool) -> SudokuPipeline:
    if source == "registry":
        return SudokuPipeline.from_registry(TRACKING_URI, try_rotations=try_rotations, repair=repair)
    return SudokuPipeline(*LOCAL_WEIGHTS, try_rotations=try_rotations, repair=repair)


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_detection(image: np.ndarray, res) -> np.ndarray:
    """bbox YOLO en bleu, grille suivie (100 intersections) en orange."""
    out = image.copy()
    thick = max(2, image.shape[1] // 400)
    if res.bbox is not None:
        x1, y1, x2, y2 = res.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 120, 0), thick)
    if res.points is not None:
        pts = res.points.astype(np.int32)
        for k in range(10):
            cv2.polylines(out, [np.ascontiguousarray(pts[k, :]).reshape(-1, 1, 2)], False, (0, 140, 255), thick)
            cv2.polylines(out, [np.ascontiguousarray(pts[:, k]).reshape(-1, 1, 2)], False, (0, 140, 255), thick)
    return out


def main() -> None:
    st.set_page_config(page_title="Sudoku CV", layout="wide")
    st.title("Solveur de sudoku par vision")
    st.caption("YOLO détecte la grille, OpenCV la redresse, YOLO-cls lit les 81 cases, "
               "puis le solveur complète la grille.")

    with st.sidebar:
        st.header("Réglages")
        source = st.radio("Modèles", ["registry", "local"],
                          format_func=lambda s: "MLflow Registry (champion)" if s == "registry" else "Fichiers locaux")
        try_rotations = st.checkbox("Essayer les 4 orientations", value=True)
        repair = st.checkbox("Réparer avec les règles du sudoku", value=True)
        mode = st.radio("Photo", ["Importer", "Caméra"])

    try:
        pipe = load_pipeline(source, try_rotations, repair)
    except Exception as exc:  # registry vide, poids absents...
        st.error(f"Impossible de charger les modèles ({source}) : {exc}")
        st.stop()

    upload = st.file_uploader("Photo du sudoku", type=["jpg", "jpeg", "png"]) if mode == "Importer" \
        else st.camera_input("Photo du sudoku")
    if upload is None:
        st.info("Importez une photo d'une grille de sudoku sur papier.")
        return

    data = upload.getvalue()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    key = (hashlib.md5(data).hexdigest(), source, try_rotations, repair)
    if st.session_state.get("key") != key:  # on ne relance le pipeline que si la photo change
        with st.spinner("Lecture de la grille…"):
            st.session_state.result = pipe.run(image)
        st.session_state.key = key
    res = st.session_state.result

    if res.status == "no_grid":
        st.error("Aucune grille trouvée sur la photo.")
        st.image(to_rgb(image), width="stretch")
        return

    t = res.timings
    st.caption(f"Temps : détection {t['detect']*1000:.0f} ms · redressement {t['warp']*1000:.0f} ms · "
               f"lecture {t['classify']*1000:.0f} ms · total {t['total']*1000:.0f} ms"
               + (f" · confiance bbox {res.bbox_conf:.2f}" if res.bbox is not None else " · sans bbox (OpenCV)"))

    col1, col2 = st.columns(2)
    col1.subheader("1. Grille trouvée")
    col1.image(to_rgb(draw_detection(image, res)), width="stretch")
    col2.subheader("2. Grille redressée")
    col2.image(to_rgb(res.flat), width="stretch",
               caption=f"rotation appliquée : {90 * res.rotation}°" if res.rotation else None)

    st.subheader("3. Grille lue (modifiable)")
    base = res.grid if res.grid is not None else res.raw_grid
    doubtful = [(r, c) for r in range(9) for c in range(9) if res.confidence[r, c] < LOW_CONF]
    notes = []
    if res.repaired:
        notes.append("réparé : " + ", ".join(f"case ({r + 1},{c + 1}) {a or 'vide'} → {b or 'vide'}"
                                             for r, c, a, b in res.repaired))
    if doubtful:
        notes.append("cases peu sûres : " + ", ".join(f"({r + 1},{c + 1})" for r, c in doubtful))
    if notes:
        st.caption(" · ".join(notes) + ". 0 = case vide.")
    edited = st.data_editor(pd.DataFrame(base, index=range(1, 10), columns=range(1, 10)),
                            width="content", key=f"editor_{key[0]}")
    grid = edited.clip(0, 9).fillna(0).astype(int).values.tolist()

    shown = copy.copy(res)
    if grid != res.grid:  # l'utilisateur a corrigé une case : on résout sa grille
        shown.grid = grid
        shown.solution = solve(grid) if is_valid_grid(grid) else None

    st.subheader("4. Solution")
    if shown.solution is None:
        msg = "doublon dans une ligne, une colonne ou un bloc" if not is_valid_grid(grid) else "aucune solution"
        st.error(f"Grille insoluble ({msg}). Corrigez les cases mal lues dans le tableau ci-dessus.")
        return
    if count_solutions(grid) > 1:
        st.warning("Cette grille a plusieurs solutions : une case est sans doute mal lue.")
    st.success("Sudoku résolu.")
    st.image(to_rgb(render_solution(image, shown)), width="stretch")


if __name__ == "__main__":
    main()
