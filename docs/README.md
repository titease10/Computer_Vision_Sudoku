# Documentation — Solveur de Sudoku par Computer Vision

Projet scolaire : à partir d'une **photo d'une grille de sudoku sur papier**, lire la grille
uniquement par vision par ordinateur, la résoudre et afficher la solution sur la photo.

## Sommaire

| # | Document | Contenu |
|---|---|---|
| — | [journal.md](journal.md) | Journal chronologique : pistes explorées, problèmes, décisions |
| 1 | [01_solveur.md](01_solveur.md) | Résolution du sudoku (backtracking + MRV) |
| 2 | [02_donnees.md](02_donnees.md) | Sources de données, vérité terrain, découpage train/val/test |
| 3 | [03_redressement.md](03_redressement.md) | Localisation de la grille, redressement, pages courbées, découpage en cases |
| 4 | [04_classifieur_chiffres.md](04_classifieur_chiffres.md) | Lecture des cases avec YOLO-cls, dataset synthétique, MLflow |
| 5 | [05_detection_grille.md](05_detection_grille.md) | Détection de la grille avec YOLO detect : datasets, découpage, pièges |

## Vue d'ensemble du pipeline

```
Photo
  │
  ├─[1] Détection de la grille (YOLO detect)             → bbox de la grille         (dataset prêt)
  ├─[2] 4 coins (OpenCV, contours)                         → quadrilatère               (fait)
  ├─[3] Redressement fin (suivi des 20 traits de grille)   → grille plate 450×450       (fait)
  ├─[4] Découpage 9×9                                      → 81 cases 64×64             (fait)
  ├─[5] Classification des cases (YOLO-cls, 10 classes)    → grille lue (0 = vide)      (v2 : 99,8 %)
  ├─[6] Résolution (backtracking + MRV)                    → grille complète            (fait)
  └─[7] Rendu : chiffres manquants reprojetés sur la photo                              (à venir)
```

**Pourquoi ce découpage en étapes ?** Chaque étape a une entrée et une sortie simples, se teste
séparément et se mesure séparément. Quand le résultat final est faux, on sait localiser
l'étape fautive (grille mal trouvée ? chiffre mal lu ?), ce qu'un modèle « bout-en-bout »
unique ne permettrait pas.

**Où intervient l'apprentissage ?** Seulement là où les règles écrites à la main échouent :
trouver la grille dans une scène quelconque (YOLO detect) et reconnaître un chiffre
(YOLO-cls). Le reste (géométrie, résolution) est déterministe : il n'a pas besoin de données
et il est exact.

## Organisation du code

```
src/sudoku_cv/
  solver.py            résolution + validation d'une grille
  warp.py              coins, redressement 4 points, redressement fin (dewarp)
  cells.py             découpage en cases, prétraitement, heuristique « case remplie »
  digit_classifier.py  enveloppe du modèle YOLO-cls
  pipeline.py          enchaînement photo -> cases
scripts/
  extract_cells.py         debug visuel du redressement sur un dossier de photos
  build_cells_dataset.py   construction du dataset de cases
  train_cells.py           entraînement YOLO-cls + MLflow
labels/rg1990_grids.json   vérité terrain des photos de test
tests/                     tests unitaires (pytest)
```

## Commandes utiles

```bash
.venv/Scripts/python -m pytest                                   # tests
.venv/Scripts/python scripts/extract_cells.py <dossier_photos> data/debug
.venv/Scripts/python scripts/build_cells_dataset.py
.venv/Scripts/python scripts/train_cells.py --epochs 15 --name cells_v1
.venv/Scripts/mlflow ui --backend-store-uri sqlite:///mlflow.db  # http://127.0.0.1:5000
```
