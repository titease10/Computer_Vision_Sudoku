import numpy as np

from sudoku_cv.pipeline import repair_grid
from test_solver import EASY


def _probs(grid, low=None):
    """Probabilités « classifieur » : 0,95 sur le bon chiffre. `low` = {case: (lu, 2e choix)}
    simule une case mal lue avec une faible confiance."""
    probs = np.full((81, 10), 0.005, dtype=np.float32)
    for k, v in enumerate(np.array(grid).ravel()):
        probs[k, v] = 0.955
    for k, (read, second) in (low or {}).items():
        probs[k] = 0.005
        probs[k, read], probs[k, second] = 0.55, 0.40
    return probs


def test_repair_keeps_correct_grid():
    grid, changes = repair_grid(_probs(EASY))
    assert grid == EASY and changes == []


def test_repair_fixes_empty_read_as_one():
    # case vide (0, 2) lue « 1 » (bord de grille, cas de la photo rg_18) : la grille lue n'a
    # plus une solution unique -> la réparation doit revenir au 2e choix (vide)
    grid, changes = repair_grid(_probs(EASY, low={2: (1, 0)}))
    assert grid == EASY
    assert changes == [(0, 2, 1, 0)]


def test_repair_fixes_two_misreads():
    # le 5 de (0, 0) lu 6, le 9 de (2, 1) lu 4
    grid, changes = repair_grid(_probs(EASY, low={0: (6, 5), 19: (4, 9)}))
    assert grid == EASY
    assert sorted(changes) == [(0, 0, 6, 5), (2, 1, 4, 9)]


def test_repair_gives_up_when_errors_are_confident():
    probs = _probs(EASY)
    probs[0] = 0.005
    probs[0, 6] = 0.955  # 5 lu 6 avec assurance : rien à faire avec 1 seul changement autorisé
    probs[19] = 0.005
    probs[19, 4] = 0.955
    assert repair_grid(probs, max_changes=1) is None
