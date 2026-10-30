# 1. Le solveur

Code : `src/sudoku_cv/solver.py`. Tests : `tests/test_solver.py`.

C'est la partie la plus simple du projet, et on l'a écrite en premier parce qu'elle ne dépend
de rien. Deux fonctions : `is_valid_grid(grid)` vérifie qu'aucun chiffre n'apparaît deux fois
dans une ligne, une colonne ou un bloc 3×3 (les cases vides valent 0 et sont ignorées), et
`solve(grid)` renvoie la grille complète, ou `None` si elle est invalide ou sans solution. La
grille d'entrée n'est pas modifiée.

## Backtracking et MRV

On remplit les cases vides une par une. Pour chaque case, on essaie un chiffre encore autorisé
et on continue ; à la première impasse (une case sans aucun chiffre possible), on revient en
arrière et on passe au chiffre suivant.

Pour savoir vite quels chiffres sont autorisés, on garde les chiffres déjà présents dans chaque
ligne, colonne et bloc, sous forme d'ensembles. Les candidats d'une case sont alors
`{1..9} − ligne − colonne − bloc`.

Dans l'ordre naturel des cases, la recherche peut être longue. On choisit donc à chaque fois la
case qui a le moins de candidats (heuristique MRV, *Minimum Remaining Values*) : une case à un
seul candidat se remplit sans risque, et une case à zéro candidat révèle l'impasse tout de
suite. La grille « AI Escargot », réputée parmi les plus difficiles, se résout ainsi en quelques
millisecondes.

## Le solveur contrôle aussi la lecture

Si le classifieur lit un 3 à la place d'un 8, la grille contient souvent un doublon, et
`is_valid_grid` le voit. C'est ce qui permet ensuite de réparer la lecture (voir
[doc 6](06_pipeline_evaluation.md)). Sans ce contrôle, le backtracking pourrait aussi tourner
longtemps sur une grille impossible.

Une grille avec plusieurs solutions (des chiffres pris pour des cases vides, par exemple) serait
quand même « résolue » par `solve`, avec une solution qui n'est pas forcément celle du journal.
On a donc ajouté `count_solutions`, qui s'arrête à 2 : le pipeline exige une solution unique.

## Tests

| Test | Ce qu'il vérifie |
|---|---|
| grille facile | solution correcte, chiffres donnés conservés |
| AI Escargot | grille difficile résolue |
| entrée non modifiée | `solve` travaille sur une copie |
| doublon | `is_valid_grid` = False et `solve` = None |
| grille insoluble mais sans doublon | `solve` = None (une case n'a aucun candidat) |
| grille vide | une solution valide existe |
