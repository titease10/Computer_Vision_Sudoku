"""Génération de grilles de sudoku pour la démo : valides, invalides, insolubles, ambiguës.

Une grille est une liste 9x9 d'entiers, 0 = case vide (comme dans `solver`). Tout passe par un
`random.Random` fourni par l'appelant, donc un lot se rejoue à l'identique avec la même graine.
"""
from __future__ import annotations

import random

from sudoku_cv.solver import Grid, count_solutions, is_valid_grid

# Nombre d'indices (cases remplies) par niveau
CLUES = {"easy": 40, "medium": 32, "hard": 26}


def random_solution(rng: random.Random) -> Grid:
    """Grille complète valide : on mélange une grille de base sans casser ses propriétés."""
    def shuffled(n: int = 3) -> list[int]:
        return [g * n + i for g in rng.sample(range(n), n) for i in rng.sample(range(n), n)]

    rows, cols = shuffled(), shuffled()
    digits = rng.sample(range(1, 10), 9)
    return [[digits[(3 * (r % 3) + r // 3 + c) % 9] for c in cols] for r in rows]


def make_puzzle(solution: Grid, clues: int, rng: random.Random) -> Grid:
    """Retire des chiffres tant que la solution reste unique, jusqu'à `clues` indices.

    Peut s'arrêter au-dessus de `clues` si retirer une case de plus rendrait la grille ambiguë.
    """
    grid = [row[:] for row in solution]
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    left = 81
    for r, c in cells:
        if left <= clues:
            break
        v, grid[r][c] = grid[r][c], 0
        if count_solutions(grid) == 1:
            left -= 1
        else:
            grid[r][c] = v
    return grid


def make_ambiguous(puzzle: Grid, rng: random.Random) -> Grid:
    """Grille valide mais à plusieurs solutions : on retire des indices jusqu'à perdre l'unicité."""
    grid = [row[:] for row in puzzle]
    givens = [(r, c) for r in range(9) for c in range(9) if grid[r][c]]
    rng.shuffle(givens)
    for r, c in givens:
        grid[r][c] = 0
        if count_solutions(grid) > 1:
            return grid
    raise ValueError("la grille est restée à solution unique")  # ne se produit pas : une grille vide a des milliards de solutions


def make_conflict(puzzle: Grid, rng: random.Random) -> Grid:
    """Grille invalide : un chiffre est recopié dans une case vide de sa ligne, colonne ou bloc."""
    grid = [row[:] for row in puzzle]
    givens = [(r, c) for r in range(9) for c in range(9) if grid[r][c]]
    rng.shuffle(givens)
    for r, c in givens:
        peers = [(rr, cc) for rr in range(9) for cc in range(9)
                 if (rr, cc) != (r, c) and not grid[rr][cc]
                 and (rr == r or cc == c or (rr // 3, cc // 3) == (r // 3, c // 3))]
        if peers:
            rr, cc = rng.choice(peers)
            grid[rr][cc] = grid[r][c]
            return grid
    raise ValueError("aucune case vide où copier un chiffre")


def make_unsolvable(puzzle: Grid, solution: Grid, rng: random.Random) -> Grid:
    """Grille sans doublon visible, mais sans solution : un faux chiffre qui ne se voit qu'en cherchant.

    `puzzle` doit avoir `solution` pour unique solution : tout autre chiffre mis dans une case vide
    mène donc forcément à une impasse.
    """
    grid = [row[:] for row in puzzle]
    empties = [(r, c) for r in range(9) for c in range(9) if not grid[r][c]]
    rng.shuffle(empties)
    for r, c in empties:
        wrong = [v for v in range(1, 10) if v != solution[r][c]]
        rng.shuffle(wrong)
        for v in wrong:
            grid[r][c] = v
            if is_valid_grid(grid):
                return grid
        grid[r][c] = 0
    raise ValueError("aucune case vide où placer un faux chiffre")


def new_puzzle(level: str, rng: random.Random) -> tuple[Grid, Grid]:
    """(grille à trous à solution unique, sa solution)."""
    solution = random_solution(rng)
    return make_puzzle(solution, CLUES[level], rng), solution

