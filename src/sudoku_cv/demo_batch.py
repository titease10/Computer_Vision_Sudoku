"""Lot de photos de démo : chaque cas sait ce que l'application devrait répondre.

`build_batch` fabrique les photos (valides, invalides, plusieurs grilles, dégradées…) ; `judge` compare
la réponse du pipeline à l'attendu, pour voir d'un coup d'œil ce qui marche et où sont les limites.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from sudoku_cv import generator, synth
from sudoku_cv.solver import Grid


@dataclass
class Case:
    name: str
    category: str
    note: str  # ce que le cas cherche à montrer
    image: np.ndarray
    expect: str  # "solve" : une solution juste est attendue ; "reject" : l'application doit refuser la grille ou la corriger
    solutions: list[Grid] = field(default_factory=list)  # solutions justes acceptées
    stress: bool = False  # cas volontairement dur : un échec est une limite, pas un bug


@dataclass(frozen=True)
class Spec:
    category: str
    note: str
    make: Callable[[random.Random], tuple[np.ndarray, list[Grid]]]  # -> (photo, solutions justes)
    expect: str = "solve"
    stress: bool = False


def _solvable(level: str, shot: str, title: str = "SUDOKU"):
    def make(rng):
        puzzle, solution = generator.new_puzzle(level, rng)
        page = synth.render_page([puzzle], rng, title)
        return synth.photograph(page, synth.PRESETS[shot], rng), [solution]
    return make


def _broken(kind: str, shot: str = "easy"):
    def make(rng):
        puzzle, solution = generator.new_puzzle("medium", rng)
        if kind == "conflict":
            grid = generator.make_conflict(puzzle, rng)
        elif kind == "unsolvable":
            grid = generator.make_unsolvable(puzzle, solution, rng)
        else:
            grid = generator.make_ambiguous(puzzle, rng)
        return synth.photograph(synth.render_page([grid], rng), synth.PRESETS[shot], rng), []
    return make


def _several(n: int, valid: int, shot: str = "easy"):
    """Page de `n` grilles dont `valid` sont justes (les autres ont un doublon)."""
    def make(rng):
        grids, solutions = [], []
        for k in range(n):
            puzzle, solution = generator.new_puzzle(rng.choice(["easy", "medium"]), rng)
            if k < valid:
                grids.append(puzzle)
                solutions.append(solution)
            else:
                grids.append(generator.make_conflict(puzzle, rng))
        order = rng.sample(range(n), n)  # la grille juste n'est pas toujours la première
        page = synth.render_page([grids[i] for i in order], rng, "SUDOKU DU JOUR")
        return synth.photograph(page, synth.PRESETS[shot], rng), solutions
    return make


def _no_grid(rng):
    return synth.photograph(synth.blank_page(rng), synth.PRESETS["easy"], rng), []


SPECS = [
    Spec("Clean", "Printed grid, almost flat: the easy case.", _solvable("easy", "scan")),
    Spec("Clean", "Medium grid, light tilt.", _solvable("medium", "easy")),
    Spec("Clean", "Hard grid (few clues): the solver does the work.", _solvable("hard", "easy")),
    Spec("Tilted", "Page photographed at an angle, with perspective.", _solvable("medium", "tilted")),
    Spec("Tilted", "Same, with a hard grid.", _solvable("hard", "tilted")),
    Spec("Rotated", "Grid on its side: the 4 orientations are tested.", _solvable("medium", "sideways")),
    Spec("Rotated", "Grid upside down.", _solvable("easy", "upside_down")),
    Spec("Shadow", "Strong shadow across the page.", _solvable("medium", "shadow")),
    Spec("Several grids", "Two grids on the page, only one is valid: the app must pick it.", _several(2, 1)),
    Spec("Several grids", "Three grids, all valid: it solves one of them.", _several(3, 3)),
    Spec("Duplicate digit", "A digit appears twice in a row, column or block: the app must refuse it or correct it.",
         _broken("conflict"), expect="reject"),
    Spec("Duplicate digit", "Same, on a tilted photo.", _broken("conflict", "tilted"), expect="reject"),
    Spec("No solution", "No visible duplicate, but a wrong digit: only the search finds the dead end.",
         _broken("unsolvable"), expect="reject"),
    Spec("Several solutions", "Too few clues: valid but ambiguous, the app must refuse it or correct it.",
         _broken("ambiguous"), expect="reject"),
    Spec("No grid", "A page of text without any sudoku.", _no_grid, expect="reject"),
    Spec("Limit", "Heavy blur: digits barely readable.", _solvable("easy", "blurry"), stress=True),
    Spec("Limit", "Dark, noisy and low contrast.", _solvable("easy", "dark_noisy"), stress=True),
    Spec("Limit", "Grid cut by the edge of the photo.", _solvable("easy", "cut"), stress=True),
]


def build_batch(seed: int, repeat: int = 1) -> list[Case]:
    """Les `SPECS` fabriquées `repeat` fois avec des grilles différentes ; même graine, même lot."""
    rng = random.Random(seed)
    cases = []
    for n in range(repeat):
        for spec in SPECS:
            image, solutions = spec.make(random.Random(rng.randrange(2**32)))
            cases.append(Case(name=f"{spec.category} #{len(cases) + 1}", category=spec.category, note=spec.note,
                              image=image, expect=spec.expect, solutions=solutions, stress=spec.stress))
    return cases


def judge(case: Case, status: str, solution: Optional[Grid], repaired: bool = False) -> tuple[str, str]:
    """Compare la réponse du pipeline à l'attendu. Retourne (verdict, explication).

    Verdicts : "ok", "limit" (cas dur non résolu), "miss" (cas normal non résolu), "wrong" (solution
    fausse, ou grille invalide résolue sans avoir été corrigée : le pire cas).
    Sur une grille invalide, refuser ou corriger puis résoudre (`repaired`) sont deux bonnes réponses.
    """
    if solution is not None and any(solution == s for s in case.solutions):
        return "ok", "solved correctly"
    if solution is not None and case.expect == "reject" and repaired:
        return "ok", "grid corrected, then solved"
    if solution is not None:
        return "wrong", "solved, but not the grid in the photo" if case.expect == "solve" \
            else "solved an invalid grid without correcting it"
    if case.expect == "reject":
        return "ok", {"no_grid": "no grid found, as expected"}.get(status, "refused, as expected")
    if case.stress:
        return "limit", "not solved: limit of the app" if status == "unsolvable" else "no grid found: limit of the app"
    return "miss", "not solved" if status == "unsolvable" else "no grid found"


def outcome(case: Case, res) -> tuple[str, str]:
    """`judge` sur un PipelineResult."""
    return judge(case, res.status, res.solution if res.status == "solved" else None, bool(res.repaired))
