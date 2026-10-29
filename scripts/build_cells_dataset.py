"""Construit le dataset YOLO-cls des cases : data/cells_cls/{train,val,test}/{0..9}/.

Sources :
  * chiffres 1-9 : polices imprimées du repo rg1990 (Chars74K), sans les polices exclues
    par l'auteur (data/font_exclude), rendues « comme une case photographiée »
    (fond gris variable, traits de grille, flou, bruit, basse résolution...) ;
  * classe 0 (vide) : cases vides synthétiques (dont chiffres fantômes du verso) +
    cases vides réelles extraites des photos d'entraînement (heuristique cell_has_ink) ;
  * test : les 81 cases des photos de test, étiquetées avec labels/rg1990_grids.json.

Split train/val par police (une police n'est jamais à la fois en train et en val).
Les cases réelles non vides des photos d'entraînement vont dans data/cells_unlabeled/
pour un pré-étiquetage ultérieur.

Usage :
    python scripts/build_cells_dataset.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sudoku_cv.cells import CELL_SIZE, cell_has_ink  # noqa: E402
from sudoku_cv.pipeline import read_cells  # noqa: E402

RG = ROOT / "data/external/cv-sudoku-solver/data"
# Photos sans vérité terrain et trop déformées pour fournir des cases vides fiables
SKIP_PHOTOS = {"17.jpg"}


def load_fonts(rg_dir: Path) -> dict[int, dict[int, np.ndarray]]:
    """{index de police: {chiffre: image en niveaux de gris}} sans les polices exclues."""
    excluded = {p.stem.split("-")[-1] for p in (rg_dir / "font_exclude").glob("*.png")}
    fonts: dict[int, dict[int, np.ndarray]] = {}
    for digit in range(1, 10):
        for p in (rg_dir / "digit_images" / f"Sample{digit + 1:03d}").glob("*.png"):
            font_id = p.stem.split("-")[-1]
            if font_id in excluded:
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                fonts.setdefault(int(font_id), {})[digit] = img
    return fonts


def _digit_mask(glyph: np.ndarray) -> np.ndarray | None:
    """Image noir sur blanc -> masque d'encre (float 0..1) rogné au chiffre."""
    ink = 1.0 - glyph.astype(np.float32) / 255.0
    ys, xs = np.where(ink > 0.5)
    if len(ys) == 0:
        return None
    return ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _place(canvas: np.ndarray, mask: np.ndarray, value: float, height_ratio: float, rng) -> None:
    """Dessine le masque dans la case, centré avec un léger décalage, couleur `value`."""
    s = canvas.shape[0]
    h = max(4, int(height_ratio * s))
    w = max(2, min(s - 2, int(mask.shape[1] * h / mask.shape[0])))
    m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
    angle = rng.uniform(-6, 6)
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m = cv2.warpAffine(m, rot, (w, h))
    y0 = int((s - h) / 2 + rng.uniform(-0.08, 0.08) * s)
    x0 = int((s - w) / 2 + rng.uniform(-0.08, 0.08) * s)
    y0, x0 = int(np.clip(y0, 0, s - h)), int(np.clip(x0, 0, s - w))
    region = canvas[y0 : y0 + h, x0 : x0 + w]
    region[:] = region * (1 - m) + value * m


def _background(s: int, rng) -> np.ndarray:
    base = rng.uniform(120, 245)
    gy, gx = np.mgrid[0:s, 0:s] / s
    gradient = rng.uniform(-25, 25) * gx + rng.uniform(-25, 25) * gy
    return np.clip(base + gradient, 0, 255).astype(np.float32)


def _grid_lines(canvas: np.ndarray, rng) -> None:
    """Restes de traits de grille sur les bords (rognage imparfait)."""
    s = canvas.shape[0]
    for side in range(4):
        if rng.random() < 0.3:
            t = int(rng.integers(1, 5))
            v = rng.uniform(20, 120)
            off = int(rng.integers(0, 4))
            if side == 0:
                canvas[off : off + t, :] = v
            elif side == 1:
                canvas[s - off - t : s - off, :] = v
            elif side == 2:
                canvas[:, off : off + t] = v
            else:
                canvas[:, s - off - t : s - off] = v


def _degrade(canvas: np.ndarray, rng) -> np.ndarray:
    s = canvas.shape[0]
    if rng.random() < 0.4:  # basse résolution (photos lointaines, écrans)
        small = int(rng.integers(14, 32))
        canvas = cv2.resize(cv2.resize(canvas, (small, small), interpolation=cv2.INTER_AREA), (s, s))
    sigma = rng.uniform(0, 1.6)
    if sigma > 0.3:
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigma)
    canvas = canvas + rng.normal(0, rng.uniform(0, 8), canvas.shape)
    img = np.clip(canvas, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))])
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return img


def render_digit_cell(mask: np.ndarray, rng, s: int = CELL_SIZE) -> np.ndarray:
    canvas = _background(s, rng)
    bg = float(canvas.mean())
    if rng.random() < 0.15:  # chiffre fantôme du verso derrière un vrai chiffre
        _place(canvas, mask, bg - rng.uniform(5, 25), rng.uniform(0.5, 0.8), rng)
    ink = rng.uniform(0, min(110, bg - 50))
    _place(canvas, mask, ink, rng.uniform(0.5, 0.85), rng)
    _grid_lines(canvas, rng)
    return _degrade(canvas, rng)


def _oblique_line(canvas: np.ndarray, rng) -> None:
    """Trait oblique ou courbe traversant un coin de la case : reste d'un bord de grille
    mal redressé ou bord de page (erreur observée avec cells_v1 : vide lu comme un 7)."""
    s = canvas.shape[0]
    side_a, side_b = rng.choice(4, size=2, replace=False)

    def point_on(side: int) -> tuple[float, float]:
        t = rng.uniform(0, s)
        return [(t, 0.0), (t, float(s)), (0.0, t), (float(s), t)][side]

    (x0, y0), (x2, y2) = point_on(side_a), point_on(side_b)
    cx, cy = (x0 + x2) / 2 + rng.uniform(-10, 10), (y0 + y2) / 2 + rng.uniform(-10, 10)
    ts = np.linspace(0, 1, 30)[:, None]
    pts = (1 - ts) ** 2 * [x0, y0] + 2 * (1 - ts) * ts * [cx, cy] + ts**2 * [x2, y2]  # Bézier
    cv2.polylines(canvas, [pts.astype(np.int32)], False, rng.uniform(20, 120), int(rng.integers(1, 4)))


def render_empty_cell(ghost_masks: list[np.ndarray], rng, s: int = CELL_SIZE, oblique: bool = False) -> np.ndarray:
    canvas = _background(s, rng)
    if oblique and rng.random() < 0.25:
        _oblique_line(canvas, rng)
    if rng.random() < 0.4:  # chiffre fantôme : à peine plus sombre que le fond
        bg = float(canvas.mean())
        ghost = ghost_masks[int(rng.integers(len(ghost_masks)))]
        _place(canvas, ghost, bg - rng.uniform(5, 28), rng.uniform(0.5, 0.85), rng)
    if rng.random() < 0.1:  # petite tache / poussière
        cv2.circle(canvas, tuple(int(v) for v in rng.integers(8, s - 8, 2)), int(rng.integers(1, 3)), rng.uniform(60, 140), -1)
    _grid_lines(canvas, rng)
    return _degrade(canvas, rng)


def _save(img: np.ndarray, path: Path) -> None:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data/cells_cls")
    parser.add_argument("--unlabeled", type=Path, default=ROOT / "data/cells_unlabeled")
    parser.add_argument("--labels", type=Path, default=ROOT / "labels/rg1990_grids.json")
    parser.add_argument("--variants", type=int, default=2, help="rendus par image de police")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--real-digits", action="store_true", help="ajoute les vraies cases chiffres (photos train)")
    parser.add_argument("--real-repeat", type=int, default=3, help="copies de chaque vraie case chiffre")
    parser.add_argument("--oblique", action="store_true", help="traits obliques/courbes dans les vides synthétiques")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    for d in (args.out, args.unlabeled):
        if d.exists():
            shutil.rmtree(d)
    for split in ("train", "val", "test"):
        for c in range(10):
            (args.out / split / str(c)).mkdir(parents=True)
    args.unlabeled.mkdir(parents=True)

    # --- chiffres 1-9 synthétiques, split par police ---
    fonts = load_fonts(RG)
    font_ids = sorted(fonts)
    rng.shuffle(font_ids)
    n_val = int(len(font_ids) * args.val_fraction)
    split_of = {fid: ("val" if k < n_val else "train") for k, fid in enumerate(font_ids)}
    counts = {"train": [0] * 10, "val": [0] * 10, "test": [0] * 10}
    masks_by_split: dict[str, list[np.ndarray]] = {"train": [], "val": []}
    for fid in font_ids:
        split = split_of[fid]
        for digit, glyph in fonts[fid].items():
            mask = _digit_mask(glyph)
            if mask is None:
                continue
            masks_by_split[split].append(mask)
            for v in range(args.variants):
                _save(render_digit_cell(mask, rng), args.out / split / str(digit) / f"f{fid}_{v}.png")
                counts[split][digit] += 1

    # --- classe 0 synthétique : autant que la moyenne d'une classe chiffre ---
    for split in ("train", "val"):
        n_empty = sum(counts[split][1:]) // 9
        for k in range(n_empty):
            _save(render_empty_cell(masks_by_split[split], rng, oblique=args.oblique), args.out / split / "0" / f"syn_{k}.png")
        counts[split][0] += n_empty

    # --- photos réelles (découpage par grille, voir labels/rg1990_grids.json) ---
    all_labels = json.loads(args.labels.read_text(encoding="utf-8"))
    test_labels, train_labels = all_labels["test"], all_labels.get("train", {})
    photos = sorted((RG / "sudoku_images").glob("*.jpg"), key=lambda p: int(p.stem))
    for photo in photos:
        if photo.name in SKIP_PHOTOS:
            continue
        grid = read_cells(cv2.imread(str(photo)))
        if grid is None:
            print(f"[ECHEC] {photo.name}")
            continue
        if photo.name in test_labels:
            for i, (cell, label) in enumerate(zip(grid.cells, test_labels[photo.name])):
                cv2.imwrite(str(args.out / "test" / label / f"{photo.stem}_{i:02d}.png"), cell)
                counts["test"][int(label)] += 1
            continue
        if photo.name in train_labels:
            for i, (cell, label) in enumerate(zip(grid.cells, train_labels[photo.name])):
                if label == "0":
                    cv2.imwrite(str(args.out / "train" / "0" / f"real_{photo.stem}_{i:02d}.png"), cell)
                    counts["train"][0] += 1
                elif args.real_digits:
                    # Peu nombreuses face au synthétique : on les répète (Ultralytics les
                    # augmente différemment à chaque époque)
                    for k in range(args.real_repeat):
                        cv2.imwrite(str(args.out / "train" / label / f"real_{photo.stem}_{i:02d}_{k}.png"), cell)
                        counts["train"][int(label)] += 1
            continue
        # photos sans étiquettes : vides -> classe 0 (train), remplies -> à étiqueter
        for i, cell in enumerate(grid.cells):
            if cell_has_ink(cell):
                cv2.imwrite(str(args.unlabeled / f"{photo.stem}_{i:02d}.png"), cell)
            else:
                cv2.imwrite(str(args.out / "train" / "0" / f"real_{photo.stem}_{i:02d}.png"), cell)
                counts["train"][0] += 1

    print(f"Polices : {len(font_ids)} ({n_val} en val)")
    for split, c in counts.items():
        print(f"{split:5s} total={sum(c):6d}  par classe={c}")
    print(f"Cases réelles à étiqueter : {len(list(args.unlabeled.glob('*.png')))} dans {args.unlabeled}")
    recipe = {"real_digits": args.real_digits, "real_repeat": args.real_repeat, "oblique": args.oblique}
    (args.out / "stats.json").write_text(json.dumps({"n_fonts": len(font_ids), "n_val_fonts": n_val, "recipe": recipe, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
