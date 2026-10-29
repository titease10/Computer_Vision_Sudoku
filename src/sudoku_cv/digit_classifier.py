"""Classifieur de cases YOLO-cls : 10 classes, '0' = case vide, '1'..'9' = chiffres."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from ultralytics import YOLO

from sudoku_cv.cells import CELL_SIZE


class DigitClassifier:
    def __init__(self, weights: Union[str, Path], device: str = "cpu"):
        self.model = YOLO(str(weights), task="classify")
        self.device = device
        # Les noms de classes sont les noms de dossiers '0'..'9'
        self.class_to_digit = {i: int(name) for i, name in self.model.names.items()}

    def predict_probs(self, cells: list[np.ndarray]) -> np.ndarray:
        """81 cases -> matrice (81, 10) de probabilités, colonne d = chiffre d (0 = vide)."""
        results = self.model.predict(cells, imgsz=CELL_SIZE, device=self.device, verbose=False)
        probs = np.zeros((len(cells), 10), dtype=np.float32)
        for k, r in enumerate(results):
            p = r.probs.data.cpu().numpy()
            for cls, digit in self.class_to_digit.items():
                probs[k, digit] = p[cls]
        return probs

    def predict_grid(self, cells: list[np.ndarray]) -> tuple[list[list[int]], np.ndarray]:
        """81 cases -> (grille 9x9 lue, confiance de chaque case (81,))."""
        probs = self.predict_probs(cells)
        digits = probs.argmax(axis=1)
        grid = digits.reshape(9, 9).tolist()
        return grid, probs.max(axis=1)
