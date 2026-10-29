"""Enchaînement des étapes : photo -> cases -> chiffres -> solution -> rendu."""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

from sudoku_cv.cells import preprocess_cell, split_cells
from sudoku_cv.solver import count_solutions, is_valid_grid, solve
from sudoku_cv.warp import dewarp_grid, find_grid_corners


@dataclass
class GridCells:
    cells: list[np.ndarray]  # 81 cases prétraitées (64x64 BGR), ligne par ligne
    flat: np.ndarray  # grille redressée 450x450
    points: np.ndarray  # 10x10x2 intersections de la grille dans la photo d'origine


def read_cells(image: np.ndarray, bbox: Optional[Sequence[float]] = None) -> Optional[GridCells]:
    """Photo -> 81 cases prêtes pour le classifieur. None si aucune grille n'est trouvée."""
    corners = find_grid_corners(image, bbox=bbox)
    if corners is None:
        return None
    flat, points = dewarp_grid(image, corners)
    cells = [preprocess_cell(c) for c in split_cells(flat)]
    return GridCells(cells=cells, flat=flat, points=points)


@dataclass
class PipelineResult:
    status: str  # "solved", "no_grid", "unsolvable"
    bbox: Optional[np.ndarray] = None  # bbox YOLO [x1, y1, x2, y2] (None : OpenCV seul)
    bbox_conf: float = 0.0
    rotation: int = 0  # nombre de quarts de tour (sens trigo) appliqués à la grille redressée
    raw_grid: Optional[list[list[int]]] = None  # lecture brute du classifieur
    grid: Optional[list[list[int]]] = None  # lecture après réparation
    solution: Optional[list[list[int]]] = None
    confidence: Optional[np.ndarray] = None  # (9, 9) confiance de la lecture brute
    repaired: list[tuple[int, int, int, int]] = field(default_factory=list)  # (ligne, col, avant, après)
    flat: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None  # 10x10x2, dans l'orientation de `grid`
    timings: dict[str, float] = field(default_factory=dict)


def _grid_score(probs: np.ndarray) -> float:
    """Vraisemblance moyenne de la lecture, calculée sur les cases lues comme chiffres.

    Sert à choisir l'orientation : une grille à l'envers donne des chiffres peu
    reconnaissables (2, 3, 4, 5, 7) donc une confiance plus faible.
    """
    best = probs.max(axis=1)
    digits = probs.argmax(axis=1) > 0
    return float(np.log(best[digits] + 1e-6).mean()) if digits.any() else -np.inf


def repair_grid(
    probs: np.ndarray, n_suspects: int = 8, max_changes: int = 2, n_alternatives: int = 2
) -> Optional[tuple[list[list[int]], list[tuple[int, int, int, int]]]]:
    """Corrige une lecture invalide en exploitant les règles du sudoku.

    Une vraie grille a exactement une solution. Si la lecture brute n'en a pas, on remplace
    les cases les moins sûres par leurs choix suivants du classifieur (au plus `max_changes`
    cases), en essayant les combinaisons de la plus probable à la moins probable.
    Retourne (grille réparée, changements) ou None si rien ne marche.
    """
    order = np.argsort(-probs, axis=1)
    grid = order[:, 0].copy()
    if count_solutions(grid.reshape(9, 9).tolist()) == 1:
        return grid.reshape(9, 9).tolist(), []

    logp = np.log(probs + 1e-9)
    suspects = np.argsort(probs.max(axis=1))[:n_suspects]
    candidates = []  # (coût en log-vraisemblance, [(case, nouvelle valeur), ...])
    for n in range(1, max_changes + 1):
        for cells in itertools.combinations(suspects, n):
            for alts in itertools.product(range(1, n_alternatives + 1), repeat=n):
                changes = [(int(c), int(order[c, a])) for c, a in zip(cells, alts)]
                cost = sum(logp[c, grid[c]] - logp[c, v] for c, v in changes)
                candidates.append((cost, changes))
    for _, changes in sorted(candidates, key=lambda x: x[0]):
        g = grid.copy()
        for c, v in changes:
            g[c] = v
        g2 = g.reshape(9, 9).tolist()
        if is_valid_grid(g2) and count_solutions(g2) == 1:
            return g2, [(c // 9, c % 9, int(grid[c]), v) for c, v in changes]
    return None


class SudokuPipeline:
    """Photo -> solution, avec le détecteur YOLO et le classifieur de cases YOLO-cls."""

    def __init__(
        self,
        grid_weights: Optional[Union[str, Path]],
        digit_weights: Union[str, Path],
        device: str = "cpu",
        try_rotations: bool = True,
        repair: bool = True,
    ):
        from sudoku_cv.digit_classifier import DigitClassifier
        from sudoku_cv.grid_detector import GridDetector

        self.detector = GridDetector(grid_weights, device=device) if grid_weights else None
        self.classifier = DigitClassifier(digit_weights, device=device)
        self.try_rotations = try_rotations
        self.repair = repair

    def run(self, image: np.ndarray) -> PipelineResult:
        res = PipelineResult(status="no_grid")
        t0 = time.perf_counter()

        bbox = None
        if self.detector is not None:
            det = self.detector.detect(image)
            if det is not None:
                bbox, res.bbox_conf = det
                res.bbox = bbox
        t1 = time.perf_counter()
        res.timings["detect"] = t1 - t0

        cells = read_cells(image, bbox=bbox)
        t2 = time.perf_counter()
        res.timings["warp"] = t2 - t1
        if cells is None:
            res.timings["total"] = t2 - t0
            return res

        # Lecture dans chaque orientation, on garde la plus vraisemblable
        best = None
        for k in range(4 if self.try_rotations else 1):
            flat = np.ascontiguousarray(np.rot90(cells.flat, k))
            batch = [preprocess_cell(c) for c in split_cells(flat)] if k else cells.cells
            probs = self.classifier.predict_probs(batch)
            score = _grid_score(probs)
            if best is None or score > best[0]:
                best = (score, k, flat, probs)
        _, res.rotation, res.flat, probs = best
        res.points = np.rot90(cells.points, res.rotation).copy()
        res.raw_grid = probs.argmax(axis=1).reshape(9, 9).tolist()
        res.confidence = probs.max(axis=1).reshape(9, 9)
        t3 = time.perf_counter()
        res.timings["classify"] = t3 - t2

        if self.repair:
            fixed = repair_grid(probs)
            if fixed is not None:
                res.grid, res.repaired = fixed
        elif count_solutions(res.raw_grid) == 1:
            res.grid = res.raw_grid
        if res.grid is not None:
            res.solution = solve(res.grid)
        res.status = "solved" if res.solution is not None else "unsolvable"
        res.timings["solve"] = time.perf_counter() - t3
        res.timings["total"] = time.perf_counter() - t0
        return res


def render_solution(image: np.ndarray, res: PipelineResult, color=(40, 160, 0)) -> np.ndarray:
    """Écrit les chiffres trouvés dans les cases vides, directement sur la photo.

    Chaque chiffre est centré dans sa case (moyenne des 4 intersections de la grille), avec
    une taille proportionnelle à la case : la perspective et la courbure sont respectées.
    """
    out = image.copy()
    if res.solution is None or res.points is None:
        return out
    pts = res.points
    for r in range(9):
        for c in range(9):
            if res.grid[r][c]:
                continue
            quad = np.array([pts[r, c], pts[r, c + 1], pts[r + 1, c + 1], pts[r + 1, c]])
            cx, cy = quad.mean(axis=0)
            side = 0.5 * (np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[1] - quad[2]))
            text = str(res.solution[r][c])
            scale = side / 40
            thick = max(1, int(side / 15))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            cv2.putText(out, text, (int(cx - tw / 2), int(cy + th / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thick, cv2.LINE_AA)
    return out
