"""Photos synthétiques de sudokus pour la démo : une page imprimée, posée et photographiée.

`render_page` dessine une page (une ou plusieurs grilles, un titre) ; `photograph` la pose sur un
fond, de travers, avec les défauts d'une vraie prise de vue (flou, bruit, ombre, JPEG).
Les réglages de `PRESETS` vont du scan propre à la photo dégradée, pour montrer où l'application tient et où elle lâche.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sudoku_cv.solver import Grid

FONT_CANDIDATES = ("arial.ttf", "times.ttf", "georgia.ttf", "DejaVuSans.ttf", "DejaVuSerif.ttf", "LiberationSans-Regular.ttf")
CELL = 56  # côté d'une case dans la page, en pixels
INK = (28, 28, 30)


@dataclass(frozen=True)
class Shot:
    """Réglages de la prise de vue. Les angles sont en degrés, les autres valeurs en fraction de la taille."""
    tilt: float = 0.0  # rotation de la page
    quarter_turns: int = 0  # page posée à 90°, 180°, 270°
    perspective: float = 0.0  # décalage aléatoire de chaque coin
    blur: float = 0.0  # écart-type du flou gaussien (pixels)
    noise: float = 0.0  # écart-type du bruit (niveaux de gris, sur 255)
    shadow: float = 0.0  # 0 à 1 : force du dégradé d'ombre
    contrast: float = 1.0
    jpeg: int = 95  # qualité de compression
    cut: float = 0.0  # fraction de la page qui dépasse du cadre


PRESETS = {
    "scan": Shot(),
    "easy": Shot(tilt=4, perspective=0.02, blur=0.6, noise=3, jpeg=90),
    "tilted": Shot(tilt=14, perspective=0.07, blur=0.8, noise=4, shadow=0.15, jpeg=85),
    "sideways": Shot(tilt=6, quarter_turns=1, perspective=0.05, blur=0.8, noise=4, jpeg=88),
    "upside_down": Shot(tilt=5, quarter_turns=2, perspective=0.05, blur=0.8, noise=4, jpeg=88),
    "shadow": Shot(tilt=6, perspective=0.05, blur=1.0, noise=5, shadow=0.55, jpeg=80),
    "blurry": Shot(tilt=5, perspective=0.05, blur=3.0, noise=6, jpeg=70),
    "dark_noisy": Shot(tilt=8, perspective=0.08, blur=1.5, noise=22, shadow=0.4, contrast=0.55, jpeg=60),
    "cut": Shot(tilt=3, perspective=0.03, blur=0.6, noise=3, cut=0.2),
}


def _fonts(size: int, rng: random.Random) -> ImageFont.FreeTypeFont:
    """Une police système au hasard, sinon celle livrée avec Pillow (toujours disponible)."""
    for name in rng.sample(FONT_CANDIDATES, len(FONT_CANDIDATES)):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def render_grid(grid: Grid, rng: random.Random, cell: int = CELL) -> np.ndarray:
    """Une grille imprimée, fond blanc : traits fins entre les cases, épais entre les blocs."""
    side = cell * 9
    img = Image.new("RGB", (side + 8, side + 8), "white")
    draw = ImageDraw.Draw(img)
    for i in range(10):
        w = 4 if i % 3 == 0 else 1
        p = 4 + i * cell
        draw.line([(p, 4), (p, 4 + side)], fill=INK, width=w)
        draw.line([(4, p), (4 + side, p)], fill=INK, width=w)
    font = _fonts(int(cell * 0.62), rng)
    for r in range(9):
        for c in range(9):
            if grid[r][c]:
                draw.text((4 + c * cell + cell / 2, 4 + r * cell + cell / 2 + 1), str(grid[r][c]),
                          fill=INK, font=font, anchor="mm")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def render_page(grids: Sequence[Grid], rng: random.Random, title: str = "SUDOKU") -> np.ndarray:
    """Page de journal : un titre puis les grilles l'une sous l'autre (deux côte à côte si trois ou quatre)."""
    tiles = [render_grid(g, rng) for g in grids]
    gh, gw = tiles[0].shape[:2]
    cols = 1 if len(tiles) <= 2 else 2
    rows = math.ceil(len(tiles) / cols)
    margin, head = 40, 80
    page = np.full((head + rows * (gh + margin) + margin, cols * (gw + margin) + margin, 3), (240, 244, 246), np.uint8)
    font = _fonts(34, rng)
    pil = Image.fromarray(page[..., ::-1])
    ImageDraw.Draw(pil).text((margin, 24), title, fill=INK, font=font)
    page = np.ascontiguousarray(np.asarray(pil)[..., ::-1])
    for k, tile in enumerate(tiles):
        r, c = divmod(k, cols)
        y, x = head + r * (gh + margin), margin + c * (gw + margin)
        page[y:y + gh, x:x + gw] = tile
    return page


def blank_page(rng: random.Random, size: tuple[int, int] = (560, 460)) -> np.ndarray:
    """Page de texte sans aucune grille, pour vérifier que l'application ne voit pas de sudoku partout."""
    h, w = size
    page = np.full((h, w, 3), (240, 244, 246), np.uint8)
    for y in range(40, h - 30, 26):
        x2 = rng.randint(w // 2, w - 40)
        cv2.line(page, (40, y), (x2, y), (150, 150, 155), 7)
    return page


def _quad(page_shape: tuple[int, int], canvas: tuple[int, int], shot: Shot, rng: random.Random) -> np.ndarray:
    """Où tombent les 4 coins de la page dans l'image finale."""
    ph, pw = page_shape[:2]
    ch, cw = canvas
    turns = shot.quarter_turns % 4
    if turns % 2:
        ph, pw = pw, ph
    scale = 0.86 * min(cw / pw, ch / ph)
    w, h = pw * scale, ph * scale
    corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]], np.float32) / 2
    a = math.radians(rng.uniform(-shot.tilt, shot.tilt))
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
    corners = corners @ rot.T + (cw / 2, ch / 2)
    corners += np.random.default_rng(rng.randrange(2**32)).uniform(-1, 1, (4, 2)).astype(np.float32) * (shot.perspective * min(w, h))
    if shot.cut:  # la page déborde du cadre sur la droite
        corners[:, 0] += cw + shot.cut * w - corners[:, 0].max()
    return corners.astype(np.float32)


def photograph(page: np.ndarray, shot: Shot, rng: random.Random, canvas: tuple[int, int] = (960, 960)) -> np.ndarray:
    """Pose la page sur un bureau et applique les défauts de la prise de vue."""
    turns = shot.quarter_turns % 4
    page = np.ascontiguousarray(np.rot90(page, -turns)) if turns else page
    ch, cw = canvas
    desk = np.array([rng.randint(40, 120), rng.randint(60, 130), rng.randint(70, 150)], np.uint8)  # bois, tissu…
    img = np.empty((ch, cw, 3), np.uint8)
    img[:] = desk
    ph, pw = page.shape[:2]
    quad = _quad(page.shape, canvas, shot, rng)
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    homography = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(page, homography, (cw, ch), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), homography, (cw, ch))
    out = np.where(mask[..., None] > 127, warped, img).astype(np.float32)

    if shot.shadow:  # dégradé orienté au hasard, comme une main ou une lampe sur le côté
        a = rng.uniform(0, math.tau)
        yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
        t = ((xx - cw / 2) * math.cos(a) + (yy - ch / 2) * math.sin(a)) / (max(cw, ch) / 2)
        out *= (1 - shot.shadow * np.clip(t * 0.5 + 0.5, 0, 1) ** 1.5)[..., None]
    out = (out - 128) * shot.contrast + 128
    if shot.blur:
        out = cv2.GaussianBlur(out, (0, 0), shot.blur)
    if shot.noise:
        out += np.random.default_rng(rng.randrange(2**32)).normal(0, shot.noise, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, shot.jpeg])
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR) if ok else out
