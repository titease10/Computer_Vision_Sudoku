from sudoku_cv.solver import count_solutions, is_valid_grid, solve

EASY = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

# Grille "AI Escargot" (réputée très difficile)
HARD = [
    [1, 0, 0, 0, 0, 7, 0, 9, 0],
    [0, 3, 0, 0, 2, 0, 0, 0, 8],
    [0, 0, 9, 6, 0, 0, 5, 0, 0],
    [0, 0, 5, 3, 0, 0, 9, 0, 0],
    [0, 1, 0, 0, 8, 0, 0, 0, 2],
    [6, 0, 0, 0, 0, 4, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 7],
    [0, 0, 7, 0, 0, 0, 3, 0, 0],
]


def _is_complete_solution(sol, puzzle):
    assert all(v != 0 for row in sol for v in row)
    assert is_valid_grid(sol)
    # Les chiffres donnés doivent être conservés
    assert all(puzzle[r][c] in (0, sol[r][c]) for r in range(9) for c in range(9))
    return True


def test_solve_easy():
    sol = solve(EASY)
    assert sol is not None and _is_complete_solution(sol, EASY)
    assert sol[0] == [5, 3, 4, 6, 7, 8, 9, 1, 2]


def test_solve_hard():
    sol = solve(HARD)
    assert sol is not None and _is_complete_solution(sol, HARD)


def test_input_not_modified():
    puzzle = [row[:] for row in EASY]
    solve(puzzle)
    assert puzzle == EASY


def test_invalid_grid_detected():
    bad = [row[:] for row in EASY]
    bad[0][2] = 5  # doublon dans la ligne 0
    assert not is_valid_grid(bad)
    assert solve(bad) is None


def test_unsolvable_grid():
    # Valide en apparence mais la case (0,8) n'a aucun candidat
    g = [[0] * 9 for _ in range(9)]
    g[0][:8] = [1, 2, 3, 4, 5, 6, 7, 8]
    g[1][8] = 9
    assert is_valid_grid(g)
    assert solve(g) is None


def test_empty_grid_solvable():
    sol = solve([[0] * 9 for _ in range(9)])
    assert sol is not None and is_valid_grid(sol)


def test_count_solutions():
    assert count_solutions(EASY) == 1
    assert count_solutions(HARD) == 1
    assert count_solutions([[0] * 9 for _ in range(9)]) == 2  # compté jusqu'à la limite
    bad = [row[:] for row in EASY]
    bad[0][2] = 5
    assert count_solutions(bad) == 0
