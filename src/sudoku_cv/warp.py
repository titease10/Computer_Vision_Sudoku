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
    refine: bool = True,
) -> Optional[np.ndarray]:
    """Retourne les 4 coins ordonnés (TL, TR, BR, BL) de la grille, en pixels de `image`.

    bbox : (x1, y1, x2, y2) issue du détecteur YOLO. Si aucun quadrilatère n'est trouvé
    dans la bbox, on retombe sur les coins de la bbox ; sans bbox on retourne None.
    refine : affine ensuite les coins avec les traits de la grille (`refine_corners`).
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
        corners = quad / scale + np.array([x0, y0], dtype=np.float32)
    elif bbox is not None:
        x1, y1, x2, y2 = bbox
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    else:
        return None
    return refine_corners(image, corners) if refine else order_corners(corners)


def _line_masks(coarse: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Masques des traits horizontaux et verticaux d'une grille redressée."""
    dark = coarse if coarse.ndim == 2 else coarse.min(axis=2)
    binary = cv2.adaptiveThreshold(dark, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 7)
    k = int(0.6 * cell)  # plus long que les traits des chiffres
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    return horiz, vert


def _peaks(profile: np.ndarray, min_value: float, merge_dist: float) -> list[tuple[float, int]]:
    """Plages contiguës où le profil dépasse min_value : (centre pondéré, largeur).

    Un trait fin peut être coupé en deux plages voisines par le seuillage : les plages
    distantes de moins de merge_dist sont fusionnées en un seul trait.
    """
    above = profile > min_value
    peaks, k = [], 0
    while k < len(profile):
        if above[k]:
            j = k
            while j < len(profile) and above[j]:
                j += 1
            idx = np.arange(k, j)
            center, width = float((idx * profile[k:j]).sum() / profile[k:j].sum()), j - k
            if peaks and center - peaks[-1][0] < merge_dist:
                prev, prev_w = peaks[-1]
                peaks[-1] = ((prev * prev_w + center * width) / (prev_w + width), prev_w + width)
            else:
                peaks.append((center, width))
            k = j
        else:
            k += 1
    return peaks


def _regular_tens(peaks: list[tuple[float, int]], cell: float) -> list[tuple[float, list[float]]]:
    """Ensembles de 10 pics qui peuvent être les 10 traits d'une grille :
    [(coût, positions)], du meilleur au moins bon.

    Deux façons de proposer des candidats :
    - 10 pics consécutifs : suit une perspective forte (espacement qui grandit) ;
    - un treillis : pour chaque couple (premier, dernier trait), on place les 8 traits
      intermédiaires attendus et on prend le pic le plus proche de chacun. Les pics
      parasites intercalés (barre d'un 1 ou d'un 7, au milieu d'une case) sont sautés.

    Coût = écart à une courbe de degré 2 (l'espacement peut varier régulièrement avec la
    perspective) + 0,1 par pic sauté (un vrai trait sauté est suspect) − un bonus si les
    traits 0, 3, 6, 9 (bords des blocs 3x3) sont plus épais : c'est ce qui départage une
    grille voisine collée, presque aussi régulière.
    """
    pos = np.array([p[0] for p in peaks])
    width = np.array([p[1] for p in peaks], dtype=np.float64)
    k = np.arange(10)
    candidates = {tuple(range(i, i + 10)) for i in range(len(peaks) - 9)}
    for i in range(len(peaks)):
        for j in range(i + 9, len(peaks)):
            step = (pos[j] - pos[i]) / 9
            if 0.6 * cell < step < 1.4 * cell:
                idx = tuple(int(np.argmin(np.abs(pos - (pos[i] + n * step)))) for n in k)
                if len(set(idx)) == 10 and np.abs(pos[list(idx)] - (pos[i] + k * step)).max() < 0.35 * step:
                    candidates.add(idx)
    tens = []
    for idx in candidates:
        idx = list(idx)
        p, w = pos[idx], width[idx]
        step = (p[-1] - p[0]) / 9
        if not 0.6 * cell < step < 1.4 * cell or np.any(np.diff(p) < 0.4 * step):
            continue
        residual = np.sqrt(np.mean((np.polyval(np.polyfit(k, p, 2), k) - p) ** 2)) / step
        if residual > 0.15:
            continue
        skipped = (idx[-1] - idx[0] + 1) - 10
        thick = w[[0, 3, 6, 9]].mean() - w[[1, 2, 4, 5, 7, 8]].mean()
        tens.append((residual + 0.1 * skipped - 0.5 * thick / w.mean(), list(p)))
    return sorted(tens, key=lambda t: t[0])


def _grid_lines(
    mask: np.ndarray, pad: int, size: int, cell: float
) -> tuple[Optional[tuple[np.poly1d, np.poly1d]], int]:
    """Premier et dernier des 10 traits verticaux d'un masque de traits : x = f(y).

    On coupe le masque en 3 bandes horizontales ; dans chaque bande, les traits (presque
    verticaux) donnent des pics du profil des colonnes. On garde la meilleure suite de 10
    pics par bande, puis on ajuste une droite à travers les 3 bandes. Plus robuste que les
    composantes connexes : deux traits qui se touchent restent séparés dans une bande.

    Renvoie aussi le nombre de bandes qui CONFIRMENT les coins actuels : une suite de 10
    traits commence et finit là où le redressement grossier place les bords de la grille.
    """
    half = int(cell / 2)
    ys, firsts, lasts, agree = [], [], [], 0
    for frac in (0.2, 0.5, 0.8):
        yc = int(pad + frac * size)
        band = mask[yc - half : yc + half] > 0
        tens = _regular_tens(_peaks(band.sum(axis=0).astype(np.float64), 0.3 * band.shape[0], cell / 4), cell)
        if not tens:
            continue
        agree += any(abs(t[0] - pad) < cell / 3 and abs(t[-1] - (pad + size - 1)) < cell / 3 for _, t in tens)
        ys.append(yc)
        firsts.append(tens[0][1][0])
        lasts.append(tens[0][1][-1])
    if len(ys) < 2:
        return None, agree
    return (np.poly1d(np.polyfit(ys, firsts, 1)), np.poly1d(np.polyfit(ys, lasts, 1))), agree


def refine_corners(image: np.ndarray, corners: np.ndarray, size: int = WARP_SIZE,
                   pad_ratio: float = 0.15) -> np.ndarray:
    """Corrige les 4 coins quand les traits de la grille les contredisent.

    Le contour extérieur peut englober autre chose que la grille (bandeau coloré, grille
    voisine collée, bord de page). On redresse grossièrement avec une marge et on cherche
    les 10 traits dans chaque direction :
    - si les traits confirment les coins (bords de grille là où on les attend), on garde
      les coins du contour, plus précis ;
    - sinon on prend l'intersection des traits extrêmes trouvés.
    Si les traits ne sont pas retrouvés, les coins d'origine sont renvoyés tels quels.
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
    horiz, vert = _line_masks(coarse, cell)
    h, h_agree = _grid_lines(horiz.T, pad, size, cell)  # y = f(x)
    v, v_agree = _grid_lines(vert, pad, size, cell)  # x = f(y)
    if h is None or v is None or (h_agree >= 2 and v_agree >= 2):
        return order_corners(corners)

    quad = []
    for fh, fv in ((h[0], v[0]), (h[0], v[1]), (h[1], v[1]), (h[1], v[0])):
        x, y = float(fv(fh(full / 2))), 0.0
        for _ in range(10):  # point fixe : intersection de deux droites presque perpendiculaires
            y = float(fh(x))
            x = float(fv(y))
        quad.append((x, y))
    quad = cv2.perspectiveTransform(np.array(quad, dtype=np.float32).reshape(-1, 1, 2), np.linalg.inv(m))
    return order_corners(quad.reshape(4, 2))


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

    horiz, vert = _line_masks(coarse, cell)
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
