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

    def detect(self, image: np.ndarray, imgsz: int = 640) -> Optional[tuple[np.ndarray, float]]:
        """Image BGR -> (bbox [x1, y1, x2, y2] en pixels, confiance) de la grille la plus sûre,
        ou None si aucune grille n'est détectée."""
        result = self.model.predict(image, imgsz=imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        if len(result.boxes) == 0:
            return None
        best = int(result.boxes.conf.argmax())
        return result.boxes.xyxy[best].cpu().numpy(), float(result.boxes.conf[best])
