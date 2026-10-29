"""Découpage de la grille redressée en 81 cases et prétraitement des cases."""
from __future__ import annotations

import cv2
import numpy as np

CELL_SIZE = 64  # taille d'entrée du classifieur YOLO-cls


def split_cells(warped: np.ndarray, margin_ratio: float = 0.12) -> list[np.ndarray]:
    """Découpe une grille redressée (carrée) en 81 cases, ordre ligne par ligne.

    margin_ratio : part de la case rognée de chaque côté pour enlever les traits de grille.
    """
    h, w = warped.shape[:2]
    cells = []
    for r in range(9):
        for c in range(9):
            y1, y2 = round(r * h / 9), round((r + 1) * h / 9)
            x1, x2 = round(c * w / 9), round((c + 1) * w / 9)
            my, mx = round(margin_ratio * (y2 - y1)), round(margin_ratio * (x2 - x1))
            cells.append(warped[y1 + my : y2 - my, x1 + mx : x2 - mx])
    return cells


def preprocess_cell(cell: np.ndarray, size: int = CELL_SIZE) -> np.ndarray:
    """Case -> image BGR size x size en niveaux de gris, prête pour le classifieur.

    On garde le canal le plus sombre (min BGR) : une encre colorée (rouge, bleue) y est
    aussi contrastée qu'une encre noire, ce qui colle aux données d'entraînement noir/blanc.
    """
    dark = cell if cell.ndim == 2 else cell.min(axis=2)
    dark = cv2.resize(dark, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(dark, cv2.COLOR_GRAY2BGR)


def cell_has_ink(cell: np.ndarray, min_contrast: int = 30, min_fill: float = 0.03) -> bool:
    """Heuristique simple « case non vide » (sert au pré-étiquetage, pas en production).

    Un pixel est de l'encre s'il est plus sombre d'au moins `min_contrast` niveaux que le
    fond de la case (médiane). On travaille sur le canal le plus sombre (min BGR) pour que
    l'encre colorée (chiffres rouges sur écran) ressorte autant que l'encre noire ; les
    chiffres fantômes du verso, gris très clair, restent sous le seuil.
    """
    dark = (cell if cell.ndim == 2 else cell.min(axis=2)).astype(np.int16)
    h, w = dark.shape
    center = dark[h // 6 : h - h // 6, w // 6 : w - w // 6]
    ink = center < np.median(dark) - min_contrast
    return float(ink.mean()) >= min_fill
