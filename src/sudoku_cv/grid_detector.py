"""Détecteur de grille YOLO detect (1 classe : sudoku)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from ultralytics import YOLO


class GridDetector:
    def __init__(self, weights: Union[str, Path], device: str = "cpu", conf: float = 0.25):
        self.model = YOLO(str(weights), task="detect")
        self.device = device
        self.conf = conf

    def detect_all(self, image: np.ndarray, imgsz: int = 640) -> list[tuple[np.ndarray, float]]:
        """Image BGR -> toutes les grilles [(bbox [x1, y1, x2, y2] en pixels, confiance)],
        de la plus sûre à la moins sûre (une double page de magazine en contient deux)."""
        result = self.model.predict(image, imgsz=imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        return [(boxes[k], float(confs[k])) for k in np.argsort(-confs)]

    def detect(self, image: np.ndarray, imgsz: int = 640) -> Optional[tuple[np.ndarray, float]]:
        """Image BGR -> (bbox, confiance) de la grille la plus sûre, ou None si aucune grille."""
        dets = self.detect_all(image, imgsz)
        return dets[0] if dets else None
