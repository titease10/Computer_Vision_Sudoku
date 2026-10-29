"""Construit le dataset YOLO detect de la grille : data/grid_det/{images,labels}/{train,val,test}.

Une seule classe : 0 = sudoku. La bbox est le rectangle englobant des 4 coins de la grille.

Sources :
  * wichtounet/sudoku_dataset : 202 photos avec les 4 coins annotés (outlines_sorted.csv)
    et les chiffres de la grille (.dat) ;
  * rg1990 : 24 photos (17 exclue) sans annotation -> coins = intersections extrêmes trouvées
    par notre redressement (dewarp_grid), À VÉRIFIER visuellement (planche de contrôle).

Découpage PAR GRILLE (toutes les photos d'une même grille dans le même ensemble) :
  * rg1990 : on reprend le découpage du classifieur (labels/rg1990_grids.json) pour garder
    un jeu de test de bout en bout propre ; ses photos train vont en train ;
  * wichtounet : grilles regroupées par leurs chiffres, réparties ~70/15/15 en nombre de photos ;
  * découpage GLOBAL : une grille présente dans les deux datasets (rg1990 24 et 25 sont aussi
    dans wichtounet) suit le découpage rg1990.

Sorties en plus des images/labels :
  data/grid_det/data.yaml           configuration Ultralytics
  data/grid_det/corners.json        les 4 coins (pixels de l'image enregistrée) + source + grille
  data/debug/grid_det_check_*.jpg   planches de contrôle (bbox verte, coins rouges)

Usage :
    python scripts/build_grid_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sudoku_cv.warp import dewarp_grid, find_grid_corners, order_corners  # noqa: E402

WICH = ROOT / "data/external/wichtounet"
RG = ROOT / "data/external/cv-sudoku-solver/data"
MAX_SIDE = 1280
SPLITS = ("train", "val", "test")


def read_dat_grid(path: Path) -> str:
    lines = path.read_text(errors="ignore").splitlines()[2:11]
    return "".join(lines).replace(" ", "")


def split_groups(groups: dict[str, list[str]], fractions=(0.70, 0.15, 0.15), seed: int = 0) -> dict[str, str]:
    """Répartit des groupes (grille -> photos) dans les ensembles, en visant des fractions
    du nombre de PHOTOS. Glouton : groupes les plus gros d'abord, vers l'ensemble le plus
    en retard sur sa cible."""
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))
    total = sum(len(v) for v in groups.values())
    filled = dict.fromkeys(SPLITS, 0)
    assign = {}
    for k in keys:
        split = max(SPLITS, key=lambda s: fractions[SPLITS.index(s)] * total - filled[s])
        assign[k] = split
        filled[split] += len(groups[k])
    return assign


def save_sample(image: np.ndarray, corners: np.ndarray, out: Path, split: str, name: str) -> dict:
    """Réduit l'image, écrit image + label YOLO, renvoie les infos (coins à l'échelle)."""
    scale = min(1.0, MAX_SIDE / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    pts = order_corners(corners) * scale
    pts[:, 0] = pts[:, 0].clip(0, w - 1)
    pts[:, 1] = pts[:, 1].clip(0, h - 1)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / "labels" / split / f"{name}.txt").write_text(
        f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n"
    )
    return {"split": split, "size": [w, h], "corners": pts.round(1).tolist()}


def check_sheet(out: Path, infos: dict[str, dict], names: list[str], path: Path, cols: int = 10) -> None:
    tiles = []
    for name in names:
        info = infos[name]
        img = cv2.imread(str(out / "images" / info["split"] / f"{name}.jpg"))
        pts = np.array(info["corners"])
        (x1, y1), (x2, y2) = pts.min(axis=0), pts.max(axis=0)
        t = max(2, img.shape[1] // 250)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), t)
        cv2.polylines(img, [pts.astype(np.int32)], True, (0, 0, 255), t)
        img = cv2.resize(img, (200, 200))
        cv2.putText(img, name[-12:], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 0), 1)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.full((200, 200, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data/grid_det")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    for sub in ("images", "labels"):
        for split in SPLITS:
            (args.out / sub / split).mkdir(parents=True)
    infos: dict[str, dict] = {}

    # --- wichtounet ---
    outlines = {Path(r["filepath"]).stem: r for r in csv.DictReader((WICH / "outlines_sorted.csv").open())}
    groups: dict[str, list[str]] = defaultdict(list)
    for stem in outlines:
        dat = WICH / "images" / f"{stem}.dat"
        if dat.exists():
            groups[read_dat_grid(dat)].append(stem)
    # Découpage GLOBAL par grille : certaines grilles rg1990 (photos 24, 25) sont aussi dans
    # wichtounet ; une grille déjà placée par le découpage rg1990 impose son ensemble.
    rg_labels = json.loads((ROOT / "labels/rg1990_grids.json").read_text(encoding="utf-8"))
    forced = {grid: split for split in ("train", "test") for grid in rg_labels[split].values()}
    assign = split_groups({g: s for g, s in groups.items() if g not in forced}, seed=args.seed)
    assign.update({g: forced[g] for g in groups if g in forced})
    for grid, stems in groups.items():
        for stem in stems:
            image = cv2.imread(str(WICH / "images" / f"{stem}.jpg"))
            r = outlines[stem]
            corners = np.array([[float(r[f"p{k}_x"]), float(r[f"p{k}_y"])] for k in range(1, 5)])
            name = f"w_{stem}"
            infos[name] = save_sample(image, corners, args.out, assign[grid], name)
            infos[name].update(source="wichtounet", grid=grid)

    # --- rg1990 : coins issus de notre redressement ---
    for photo in sorted((RG / "sudoku_images").glob("*.jpg"), key=lambda p: int(p.stem)):
        if photo.name == "17.jpg":
            continue  # coin bas-gauche mal trouvé (page très courbée)
        split = "test" if photo.name in rg_labels["test"] else "train"
        image = cv2.imread(str(photo))
        corners = find_grid_corners(image)
        _, pts = dewarp_grid(image, corners)
        extremes = np.array([pts[0, 0], pts[0, 9], pts[9, 9], pts[9, 0]])
        name = f"rg_{photo.stem}"
        infos[name] = save_sample(image, extremes, args.out, split, name)
        infos[name].update(source="rg1990", grid=rg_labels[split][photo.name])

    (args.out / "corners.json").write_text(json.dumps(infos, indent=1), encoding="utf-8")
    (args.out / "data.yaml").write_text(
        f"path: {args.out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: sudoku\n",
        encoding="utf-8",
    )

    # --- statistiques + planches de contrôle ---
    debug = ROOT / "data/debug"
    debug.mkdir(parents=True, exist_ok=True)
    for src in ("rg1990", "wichtounet"):
        names = sorted(n for n, i in infos.items() if i["source"] == src)
        for k in range(0, len(names), 60):
            check_sheet(args.out, infos, names[k : k + 60], debug / f"grid_det_check_{src}_{k // 60}.jpg")
    print(f"Grilles wichtounet distinctes : {len(groups)}")
    for split in SPLITS:
        per_src = {s: sum(1 for i in infos.values() if i["split"] == split and i["source"] == s) for s in ("wichtounet", "rg1990")}
        n_grids = len({i["grid"] for i in infos.values() if i["split"] == split})
        print(f"{split:5s} : {sum(per_src.values()):3d} images ({per_src}), {n_grids} grilles distinctes")
    # Contrôle anti-fuite : aucune grille dans deux ensembles
    seen: dict[str, set] = defaultdict(set)
    for i in infos.values():
        seen[i["grid"]].add(i["split"])
    print("Grilles présentes dans plusieurs ensembles :", sum(len(s) > 1 for s in seen.values()))


if __name__ == "__main__":
    main()
