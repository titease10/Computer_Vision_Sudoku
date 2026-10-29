"""Redresse des photos de sudoku et en extrait les 81 cases.

Usage :
    python scripts/extract_cells.py data/external/cv-sudoku-solver/data/sudoku_images data/debug

Pour chaque photo, écrit dans le dossier de sortie :
    corners/<nom>.jpg   photo réduite : quadrilatère (rouge) + 100 intersections (vert)
    warped/<nom>.png    grille redressée 450x450
    cells/<nom>_<r><c>.png  les 81 cases 64x64 (à trier ensuite dans cells_cls/{0..9})
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sudoku_cv.cells import cell_has_ink, preprocess_cell, split_cells  # noqa: E402
from sudoku_cv.warp import dewarp_grid, find_grid_corners  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    for sub in ("corners", "warped", "cells"):
        (args.output_dir / sub).mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    failed = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            failed.append(path.name)
            continue
        corners = find_grid_corners(image)
        if corners is None:
            failed.append(path.name)
            print(f"[ECHEC] {path.name} : grille non trouvée")
            continue

        scale = 800 / max(image.shape[:2])
        preview = cv2.resize(image, None, fx=scale, fy=scale)
        cv2.polylines(preview, [np.int32(corners * scale)], True, (0, 0, 255), 3)
        warped, points = dewarp_grid(image, corners)
        for x, y in points.reshape(-1, 2) * scale:
            cv2.circle(preview, (int(x), int(y)), 3, (0, 200, 0), -1)
        cv2.imwrite(str(args.output_dir / "corners" / f"{path.stem}.jpg"), preview)

        cv2.imwrite(str(args.output_dir / "warped" / f"{path.stem}.png"), warped)

        cells = split_cells(warped)
        for i, cell in enumerate(cells):
            cv2.imwrite(str(args.output_dir / "cells" / f"{path.stem}_{i // 9}{i % 9}.png"), preprocess_cell(cell))

        grid = "".join("x" if cell_has_ink(c) else "." for c in cells)
        print(f"[OK]    {path.name} : {grid.count('x')} cases remplies")

    print(f"\n{len(paths) - len(failed)}/{len(paths)} grilles redressées")
    if failed:
        print("Échecs :", ", ".join(failed))


if __name__ == "__main__":
    main()
