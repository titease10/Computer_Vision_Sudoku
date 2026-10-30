import random

import numpy as np

from sudoku_cv import generator, synth
from sudoku_cv.demo_batch import SPECS, Case, build_batch, judge
from sudoku_cv.solver import count_solutions, is_valid_grid, solve


def test_random_solution_is_complete_and_valid():
    grid = generator.random_solution(random.Random(0))
    assert all(v for row in grid for v in row) and is_valid_grid(grid)


def test_puzzle_has_a_unique_solution_and_few_clues():
    rng = random.Random(1)
    solution = generator.random_solution(rng)
    puzzle = generator.make_puzzle(solution, 30, rng)
    assert count_solutions(puzzle) == 1 and solve(puzzle) == solution
    assert sum(v != 0 for row in puzzle for v in row) <= 34


def test_conflict_is_invalid():
    puzzle, _ = generator.new_puzzle("medium", random.Random(2))
    assert not is_valid_grid(generator.make_conflict(puzzle, random.Random(2)))


def test_unsolvable_looks_valid_but_has_no_solution():
    rng = random.Random(3)
    puzzle, solution = generator.new_puzzle("medium", rng)
    grid = generator.make_unsolvable(puzzle, solution, rng)
    assert is_valid_grid(grid) and count_solutions(grid) == 0


def test_ambiguous_has_several_solutions():
    rng = random.Random(4)
    puzzle, _ = generator.new_puzzle("medium", rng)
    assert count_solutions(generator.make_ambiguous(puzzle, rng)) == 2


def test_same_seed_same_batch():
    a, b = build_batch(7), build_batch(7)
    assert len(a) == len(SPECS)
    assert all(np.array_equal(x.image, y.image) and x.solutions == y.solutions for x, y in zip(a, b))


def test_photos_are_images_and_only_solvable_cases_have_a_solution():
    for case in build_batch(5):
        assert case.image.ndim == 3 and case.image.shape[2] == 3
        assert bool(case.solutions) == (case.expect == "solve")


def test_every_preset_renders():
    rng = random.Random(6)
    page = synth.render_page([generator.new_puzzle("easy", rng)[0]], rng)
    for shot in synth.PRESETS.values():
        assert synth.photograph(page, shot, rng).shape == (960, 960, 3)


def _case(expect, solutions, stress=False):
    return Case("c", "cat", "", np.zeros((1, 1, 3), np.uint8), expect, solutions, stress)


def test_judge():
    sol = [[1] * 9 for _ in range(9)]
    other = [[2] * 9 for _ in range(9)]
    assert judge(_case("solve", [sol]), "solved", sol)[0] == "ok"
    assert judge(_case("solve", [sol]), "solved", other)[0] == "wrong"
    assert judge(_case("solve", [sol]), "unsolvable", None)[0] == "miss"
    assert judge(_case("solve", [sol], stress=True), "no_grid", None)[0] == "limit"
    assert judge(_case("reject", []), "unsolvable", None)[0] == "ok"
    assert judge(_case("reject", []), "solved", sol)[0] == "wrong"
    assert judge(_case("reject", []), "solved", sol, repaired=True)[0] == "ok"
