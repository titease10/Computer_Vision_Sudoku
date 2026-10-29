# Étape 1 — Le solveur

Fichier : `src/sudoku_cv/solver.py` — tests : `tests/test_solver.py`

## Ce qui a été fait

Deux fonctions :

- `is_valid_grid(grid)` : vérifie qu'aucun chiffre n'apparaît deux fois dans une même ligne,
  colonne ou bloc 3×3 (les cases vides valent 0 et sont ignorées).
- `solve(grid)` : renvoie la grille complète, ou `None` si la grille est invalide ou sans
  solution. La grille d'entrée n'est pas modifiée.

## Explication

### Backtracking (retour sur trace)

On remplit les cases vides une par une. Pour une case, on essaie chaque chiffre encore
autorisé ; on continue récursivement ; si on arrive à une impasse (une case sans aucun
chiffre possible), on revient en arrière et on essaie le chiffre suivant.

Pour que « chiffre autorisé » soit rapide, on garde trois listes d'ensembles :
chiffres déjà présents dans chaque ligne, chaque colonne et chaque bloc. Les candidats d'une
case sont `{1..9} − ligne − colonne − bloc`.

### Heuristique MRV (« Minimum Remaining Values »)

Au lieu de remplir les cases dans l'ordre, on choisit à chaque fois la case qui a **le moins
de candidats**. Si une case n'a qu'un candidat, on le pose sans se tromper ; si une case n'en
a aucun, on détecte l'impasse immédiatement. Cela réduit énormément l'arbre de recherche : la
grille « AI Escargot », réputée parmi les plus difficiles, est résolue en quelques
millisecondes.

## Pourquoi `is_valid_grid` est importante pour la vision

Le solveur sert aussi de **contrôle qualité de la lecture**. Si le classifieur lit un 3 à la
place d'un 8, la grille contient souvent un doublon : `is_valid_grid` le détecte, et on
pourra alors corriger en reprenant les cases les moins sûres (étape « pipeline »). Sans ce
contrôle, le backtracking pourrait tourner longtemps sur une grille impossible.

## Tests

| Test | Ce qu'il vérifie |
|---|---|
| grille facile | solution correcte, chiffres donnés conservés |
| AI Escargot | grille difficile résolue |
| entrée non modifiée | `solve` travaille sur une copie |
| doublon | `is_valid_grid` = False et `solve` = None |
| grille insoluble mais sans doublon | `solve` = None (une case n'a aucun candidat) |
| grille vide | une solution valide existe |

## Problèmes / limites

- Une grille avec **plusieurs solutions** (trop peu de chiffres lus, par exemple si des
  chiffres sont pris pour des cases vides) sera « résolue » quand même : le solveur renvoie la
  première solution trouvée, qui n'est pas forcément celle du journal. Piste : compter les
  solutions (arrêt à 2) et signaler une lecture suspecte si la grille n'en a pas exactement une.
