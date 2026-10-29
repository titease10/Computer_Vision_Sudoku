"""Solveur de sudoku : backtracking avec heuristique MRV (case la plus contrainte d'abord).

Une grille est une liste 9x9 (ou un np.ndarray) d'entiers, 0 = case vide.
"""
from __future__ import annotations

from typing import Optional, Sequence

Grid = list[list[int]]


def _to_list(grid: Sequence[Sequence[int]]) -> Grid:
    g = [[int(v) for v in row] for row in grid]
    if len(g) != 9 or any(len(row) != 9 for row in g):
        raise ValueError("La grille doit être de taille 9x9")
    if any(v < 0 or v > 9 for row in g for v in row):
        raise ValueError("Les valeurs doivent être entre 0 et 9")
    return g


def is_valid_grid(grid: Sequence[Sequence[int]]) -> bool:
    """Vérifie qu'aucun chiffre n'apparaît deux fois dans une ligne, colonne ou bloc.

    Sert à détecter une mauvaise lecture OCR avant de lancer la résolution.
    """
    g = _to_list(grid)
    for i in range(9):
        row = [v for v in g[i] if v]
        col = [g[r][i] for r in range(9) if g[r][i]]
        br, bc = 3 * (i // 3), 3 * (i % 3)
        box = [g[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3) if g[r][c]]
        if len(row) != len(set(row)) or len(col) != len(set(col)) or len(box) != len(set(box)):
            return False
    return True


def solve(grid: Sequence[Sequence[int]]) -> Optional[Grid]:
    """Résout la grille. Retourne la grille complète, ou None si insoluble / invalide."""
    solutions = _search(grid, limit=1)
    return solutions[0] if solutions else None


def count_solutions(grid: Sequence[Sequence[int]], limit: int = 2) -> int:
    """Nombre de solutions, compté jusqu'à `limit` (0 = invalide ou insoluble).

    Une vraie grille de journal a exactement 1 solution : 0 ou 2+ trahit une mauvaise lecture.
    """
    return len(_search(grid, limit=limit))


def _search(grid: Sequence[Sequence[int]], limit: int) -> list[Grid]:
    g = _to_list(grid)
    if not is_valid_grid(g):
        return []
    solutions: list[Grid] = []

    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empties: list[tuple[int, int]] = []
    for r in range(9):
        for c in range(9):
            v = g[r][c]
            if v:
                rows[r].add(v)
                cols[c].add(v)
                boxes[3 * (r // 3) + c // 3].add(v)
            else:
                empties.append((r, c))

    def candidates(r: int, c: int) -> set[int]:
        return set(range(1, 10)) - rows[r] - cols[c] - boxes[3 * (r // 3) + c // 3]

    def backtrack() -> bool:
        """Retourne True quand `limit` solutions ont été trouvées (arrêt de la recherche)."""
        if not empties:
            solutions.append([row[:] for row in g])
            return len(solutions) >= limit
        # MRV : choisir la case vide avec le moins de candidats
        idx = min(range(len(empties)), key=lambda i: len(candidates(*empties[i])))
        r, c = empties.pop(idx)
        b = 3 * (r // 3) + c // 3
        for v in candidates(r, c):
            g[r][c] = v
            rows[r].add(v)
            cols[c].add(v)
            boxes[b].add(v)
            if backtrack():
                return True
            rows[r].discard(v)
            cols[c].discard(v)
            boxes[b].discard(v)
        g[r][c] = 0
        empties.insert(idx, (r, c))
        return False

    backtrack()
    return solutions
