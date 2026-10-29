"""Pré-étiquette les vraies cases remplies (data/cells_unlabeled) avec un modèle entraîné.

Étiquetage semi-automatique : le modèle propose, l'humain vérifie et corrige.
Deux aides à la vérification :
  * une planche d'images par classe prédite (cases numérotées), pour repérer d'un coup d'œil
    un chiffre rangé dans la mauvaise colonne, et une planche des cases les moins sûres ;
  * un contrôle par les règles du sudoku : pour chaque photo, la grille reconstituée
    (cases vides = 0, cases remplies = prédiction) doit être valide et soluble. Une
    étiquette fausse crée souvent un doublon.

Sorties :
    labels/real_cells_prelabels.json     {fichier: {"pred": d, "conf": p}}
    data/debug/prelabel/classe_<d>.png   planches par classe prédite
    data/debug/prelabel/moins_surs.png   les cases de plus faible confiance

Usage :
    python scripts/prelabel_cells.py --weights runs/cells/cells_v1/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sudoku_cv.digit_classifier import DigitClassifier  # noqa: E402
from sudoku_cv.solver import is_valid_grid, solve  # noqa: E402

TILE = 64
COLS = 16


def contact_sheet(items: list[tuple[str, np.ndarray]], path: Path) -> None:
    """Planche d'images numérotées (le numéro renvoie à l'ordre de la liste)."""
    rows = (len(items) + COLS - 1) // COLS
    sheet = np.full((rows * (TILE + 14), COLS * TILE, 3), 255, np.uint8)
    for k, (_, img) in enumerate(items):
        r, c = divmod(k, COLS)
        y, x = r * (TILE + 14), c * TILE
        sheet[y + 14 : y + 14 + TILE, x : x + TILE] = img
        cv2.putText(sheet, str(k), (x + 2, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1)
    cv2.imwrite(str(path), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/cells/cells_v1/weights/best.pt")
    parser.add_argument("--cells", type=Path, default=ROOT / "data/cells_unlabeled")
    parser.add_argument("--out", type=Path, default=ROOT / "labels/real_cells_prelabels.json")
    parser.add_argument("--sheets", type=Path, default=ROOT / "data/debug/prelabel")
    args = parser.parse_args()

    paths = sorted(args.cells.glob("*.png"))
    images = [cv2.imread(str(p)) for p in paths]
    probs = DigitClassifier(args.weights).predict_probs(images)
    preds, confs = probs.argmax(axis=1), probs.max(axis=1)

    labels = {p.name: {"pred": int(d), "conf": round(float(c), 4)} for p, d, c in zip(paths, preds, confs)}
    args.out.write_text(json.dumps(labels, indent=1), encoding="utf-8")

    # Planches par classe prédite, triées par confiance croissante
    args.sheets.mkdir(parents=True, exist_ok=True)
    by_class: dict[int, list] = defaultdict(list)
    for p, img, d, c in zip(paths, images, preds, confs):
        by_class[int(d)].append((c, p.name, img))
    for d, items in sorted(by_class.items()):
        items.sort(key=lambda t: t[0])
        contact_sheet([(n, im) for _, n, im in items], args.sheets / f"classe_{d}.png")
        (args.sheets / f"classe_{d}.txt").write_text(
            "\n".join(f"{k}\t{n}\t{c:.3f}" for k, (c, n, _) in enumerate(items)), encoding="utf-8"
        )
    order = np.argsort(confs)[:32]
    contact_sheet([(paths[i].name, images[i]) for i in order], args.sheets / "moins_surs.png")
    (args.sheets / "moins_surs.txt").write_text(
        "\n".join(f"{k}\t{paths[i].name}\tpred={preds[i]}\t{confs[i]:.3f}" for k, i in enumerate(order)),
        encoding="utf-8",
    )

    # Contrôle par les règles du sudoku, photo par photo
    grids: dict[str, list[int]] = defaultdict(lambda: [0] * 81)
    for p, d in zip(paths, preds):
        photo, idx = p.stem.rsplit("_", 1)
        grids[photo][int(idx)] = int(d)
    print(f"{len(paths)} cases pré-étiquetées, {int((preds == 0).sum())} prédites vides")
    print(f"Confiance < 0.9 : {int((confs < 0.9).sum())}   < 0.99 : {int((confs < 0.99).sum())}")
    print("\nContrôle sudoku par photo :")
    for photo in sorted(grids, key=lambda s: int(s)):
        g = [grids[photo][r * 9 : r * 9 + 9] for r in range(9)]
        valid = is_valid_grid(g)
        status = "valide + soluble" if valid and solve(g) else ("valide, INSOLUBLE" if valid else "DOUBLON")
        print(f"  photo {photo:>2} : {sum(v > 0 for v in grids[photo]):2d} chiffres  {status}")


if __name__ == "__main__":
    main()
