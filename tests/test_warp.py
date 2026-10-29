import cv2
import numpy as np

from sudoku_cv.cells import cell_has_ink, split_cells
from sudoku_cv.warp import WARP_SIZE, dewarp_grid, find_grid_corners, order_corners, warp_grid


def _synthetic_photo(digits=None):
    """Grille 9x9 dessinée à plat, puis déformée en perspective sur un fond gris."""
    flat = np.full((450, 450, 3), 255, np.uint8)
    for i in range(10):
        t = 4 if i % 3 == 0 else 1
        cv2.line(flat, (i * 50 if i < 9 else 449, 0), (i * 50 if i < 9 else 449, 449), (0, 0, 0), t)
        cv2.line(flat, (0, i * 50 if i < 9 else 449), (449, i * 50 if i < 9 else 449), (0, 0, 0), t)
    for (r, c), d in (digits or {}).items():
        cv2.putText(flat, str(d), (c * 50 + 13, r * 50 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

    target = np.array([[180, 120], [620, 160], [650, 610], [140, 580]], dtype=np.float32)
    src = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, target)
    bg = np.full((800, 800, 3), 170, np.uint8)
    photo = cv2.warpPerspective(flat, m, (800, 800), dst=bg, borderMode=cv2.BORDER_TRANSPARENT)
    return photo, target


def test_order_corners():
    pts = np.array([[10, 90], [90, 90], [10, 10], [90, 10]], dtype=np.float32)
    assert order_corners(pts).tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_find_corners_full_image():
    photo, target = _synthetic_photo()
    corners = find_grid_corners(photo)
    assert corners is not None
    assert np.abs(corners - target).max() < 8


def test_find_corners_with_bbox():
    photo, target = _synthetic_photo()
    bbox = (*target.min(axis=0), *target.max(axis=0))
    corners = find_grid_corners(photo, bbox=bbox)
    assert np.abs(corners - target).max() < 8


def test_bbox_fallback_when_no_grid():
    blank = np.full((300, 300, 3), 200, np.uint8)
    assert find_grid_corners(blank) is None
    corners = find_grid_corners(blank, bbox=(50, 60, 250, 270))
    assert corners.tolist() == [[50, 60], [250, 60], [250, 270], [50, 270]]


def test_warp_split_and_ink():
    digits = {(0, 0): 5, (4, 4): 7, (8, 2): 3}
    photo, _ = _synthetic_photo(digits)
    warped, m, m_inv = warp_grid(photo, find_grid_corners(photo))
    assert warped.shape == (WARP_SIZE, WARP_SIZE, 3)
    assert np.allclose(m @ m_inv, np.eye(3) * (m @ m_inv)[2, 2], atol=1e-6)

    cells = split_cells(warped)
    assert len(cells) == 81
    filled = {i for i, cell in enumerate(cells) if cell_has_ink(cell)}
    assert filled == {r * 9 + c for r, c in digits}


def test_dewarp_grid_intersections_and_cells():
    digits = {(0, 8): 2, (3, 3): 9, (8, 8): 1}
    photo, target = _synthetic_photo(digits)
    flat, pts = dewarp_grid(photo, find_grid_corners(photo))
    assert flat.shape == (WARP_SIZE, WARP_SIZE, 3)
    assert pts.shape == (10, 10, 2)
    # Les 4 intersections extrêmes = les 4 coins de la grille dans la photo
    extremes = np.array([pts[0, 0], pts[0, 9], pts[9, 9], pts[9, 0]])
    assert np.abs(extremes - target).max() < 8

    filled = {i for i, cell in enumerate(split_cells(flat)) if cell_has_ink(cell)}
    assert filled == {r * 9 + c for r, c in digits}
