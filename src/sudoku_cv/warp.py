"""Localisation des 4 coins de la grille et redressement (vue de dessus).

Utilisé après le détecteur YOLO : on cherche les coins précis dans la bbox de la grille.
Sans bbox, la recherche se fait sur toute l'image (mode OpenCV seul).
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

WARP_SIZE = 450  # 9 cases de 50 px

# Taille max de travail pour la recherche de contours (accélère sur les photos 12 Mpx)
_WORK_MAX_SIDE = 1000


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordonne 4 points en haut-gauche, haut-droite, bas-droite, bas-gauche."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _find_quad(gray: np.ndarray, min_area_ratio: float) -> Optional[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    min_area = min_area_ratio * gray.shape[0] * gray.shape[1]
    # 1er essai avec dilatation : referme les petites coupures des traits de la grille.
    # 2e essai sans : sur une petite image, la dilatation peut souder la grille au bord de la
    # page (ombre de la reliure) et le contour n'est plus un quadrilatère.
    for mask in (cv2.dilate(thresh, np.ones((3, 3), np.uint8)), thresh):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            if cv2.contourArea(cnt) < min_area:
                break
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32)
    return None


def find_grid_corners(
    image: np.ndarray,
    bbox: Optional[Sequence[float]] = None,
    min_area_ratio: float = 0.1,
) -> Optional[np.ndarray]:
    """Retourne les 4 coins ordonnés (TL, TR, BR, BL) de la grille, en pixels de `image`.

    bbox : (x1, y1, x2, y2) issue du détecteur YOLO. Si aucun quadrilatère n'est trouvé
    dans la bbox, on retombe sur les coins de la bbox ; sans bbox on retourne None.
    """
    h, w = image.shape[:2]
    x0, y0 = 0, 0
    region = image
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # Marge de 5 % : la bbox YOLO peut rogner les bords de la grille
        mx, my = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
        x0, y0 = int(max(0, x1 - mx)), int(max(0, y1 - my))
        xe, ye = int(min(w, x2 + mx)), int(min(h, y2 + my))
        region = image[y0:ye, x0:xe]

    gray = region if region.ndim == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, _WORK_MAX_SIDE / max(gray.shape[:2]))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    quad = _find_quad(gray, min_area_ratio)
    if quad is not None:
        return order_corners(quad / scale + np.array([x0, y0], dtype=np.float32))
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        return order_corners(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
    return None


def _fit_line(samples: list[tuple[float, float]], expected: float) -> np.poly1d:
    """Ajuste pos = f(t) (degré 2) sur les points d'un trait ; rejette les points aberrants."""
    if len(samples) < 8:
        return np.poly1d([expected])
    t, p = np.array(samples, dtype=np.float64).T
    coeffs = np.polyfit(t, p, 2)
    keep = np.abs(np.polyval(coeffs, t) - p) < 4
    if keep.sum() >= 8:
        coeffs = np.polyfit(t[keep], p[keep], 2)
    return np.poly1d(coeffs)


def _track_lines(mask: np.ndarray, pad: int, cell: float, axis: int) -> list[np.poly1d]:
    """Suit les 10 traits de la grille dans un masque de traits (horizontaux si axis=0).

    Pour le trait k, on cherche, le long du trait, la position de réponse maximale dans une
    fenêtre autour de sa position théorique pad + k*cell, puis on ajuste une parabole.
    """
    m = mask if axis == 0 else mask.T  # on se ramène toujours à des traits horizontaux
    n = m.shape[0]
    win = int(0.4 * cell)
    lines = []
    for k in range(10):
        expected = pad + k * cell
        lo, hi = max(0, int(expected - win)), min(n, int(expected + win) + 1)
        samples = []
        for t in range(pad, n - pad, 4):
            column = m[lo:hi, max(0, t - 2) : t + 3].sum(axis=1)
            if column.max() > 0:
                samples.append((t, lo + float(np.argmax(column))))
        lines.append(_fit_line(samples, expected))
    return lines


def dewarp_grid(
    image: np.ndarray, corners: np.ndarray, size: int = WARP_SIZE, pad_ratio: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Redressement fin qui corrige la courbure des pages (livres, feuilles pliées).

    1. Redressement grossier par les 4 coins, avec une marge pour ne rien couper.
    2. Suivi des 10 traits horizontaux et 10 verticaux, calcul des 100 intersections.
    3. Remappage case par case (interpolation bilinéaire entre intersections).

    Retourne (grille aplatie size x size, intersections 10x10x2 en pixels de `image`,
    ordre [ligne, colonne, (x, y)]) — les intersections servent au rendu de la solution.
    """
    pad = int(pad_ratio * size)
    full = size + 2 * pad
    cell = (size - 1) / 9
    dst = np.array(
        [[pad, pad], [pad + size - 1, pad], [pad + size - 1, pad + size - 1], [pad, pad + size - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(order_corners(corners), dst)
    coarse = cv2.warpPerspective(image, m, (full, full), borderMode=cv2.BORDER_REPLICATE)

    dark = coarse if coarse.ndim == 2 else coarse.min(axis=2)
    binary = cv2.adaptiveThreshold(dark, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 7)
    k = int(0.6 * cell)  # plus long que les traits des chiffres
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    h_lines = _track_lines(horiz, pad, cell, axis=0)  # y = f(x)
    v_lines = _track_lines(vert, pad, cell, axis=1)  # x = f(y)

    pts = np.zeros((10, 10, 2), dtype=np.float32)
    for i, fh in enumerate(h_lines):
        for j, fv in enumerate(v_lines):
            x, y = pad + j * cell, pad + i * cell
            for _ in range(5):  # point fixe : intersection des deux courbes
                y = fh(x)
                x = fv(y)
            pts[i, j] = (x, y)

    # Carte de remappage : chaque pixel de sortie -> position bilinéaire dans sa case
    u = np.arange(size, dtype=np.float32) / cell
    jj = np.minimum(u.astype(int), 8)
    a = u - jj
    J, I = np.meshgrid(jj, jj)
    A, B = np.meshgrid(a, a)
    A, B = A[..., None], B[..., None]
    p00, p01, p10, p11 = pts[I, J], pts[I, J + 1], pts[I + 1, J], pts[I + 1, J + 1]
    src = (1 - B) * ((1 - A) * p00 + A * p01) + B * ((1 - A) * p10 + A * p11)
    src = np.ascontiguousarray(src, dtype=np.float32)
    flat = cv2.remap(coarse, src, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    m_inv = np.linalg.inv(m)
    pts_image = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), m_inv).reshape(10, 10, 2)
    return flat, pts_image


def warp_grid(
    image: np.ndarray, corners: np.ndarray, size: int = WARP_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Redresse la grille en un carré size x size.

    Retourne (image redressée, matrice M image->carré, matrice inverse carré->image).
    """
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    src = order_corners(corners)
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, m, (size, size))
    return warped, m, m_inv
