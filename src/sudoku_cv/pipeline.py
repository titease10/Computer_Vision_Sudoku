"""Enchaînement des étapes : photo -> cases -> chiffres -> solution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from sudoku_cv.cells import preprocess_cell, split_cells
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
