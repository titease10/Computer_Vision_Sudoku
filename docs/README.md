# Documentation

La présentation générale du projet est dans le [README à la racine](../README.md). Ici, on
garde le détail de chaque étape : ce qu'on a essayé, les mesures, et ce qui n'a pas marché.

Pour une première lecture, le plus parlant est le [journal](journal.md), écrit au fil du projet.

| # | Document | Contenu |
|---|---|---|
| 0 | [00_note_de_cadrage.md](00_note_de_cadrage.md) | Note de cadrage (rendu du 30 octobre) |
| — | [journal.md](journal.md) | Journal chronologique : pistes explorées, problèmes, décisions |
| 1 | [01_solveur.md](01_solveur.md) | Résolution du sudoku (backtracking + MRV) |
| 2 | [02_donnees.md](02_donnees.md) | Sources de données, vérité terrain, découpage train/val/test |
| 3 | [03_redressement.md](03_redressement.md) | Localisation de la grille, redressement, pages courbées, découpage en cases |
| 4 | [04_classifieur_chiffres.md](04_classifieur_chiffres.md) | Lecture des cases avec YOLO-cls, dataset synthétique, MLflow |
| 5 | [05_detection_grille.md](05_detection_grille.md) | Détection de la grille avec YOLO detect : datasets, découpage, pièges, résultats |
| 6 | [06_pipeline_evaluation.md](06_pipeline_evaluation.md) | Pipeline complet, orientation, réparation, évaluation de bout en bout, Model Registry |
| 7 | [07_application.md](07_application.md) | Application Streamlit (photo ou caméra, grille modifiable) |
| 8 | [08_explication_complete.md](08_explication_complete.md) | Explication de chaque étape : fonctionnement, notions, fonctions appelées, limites, glossaire |

## Le pipeline

```
Photo
  ├─ détection de la grille (YOLO detect)        cadre de la grille        grid_v1 : mAP50 0,995
  ├─ 4 coins (OpenCV, contours)                  quadrilatère
  ├─ redressement fin (suivi des 20 traits)      grille plate 450×450
  ├─ découpage 9×9                               81 cases 64×64
  ├─ classification (YOLO-cls, 10 classes)       grille lue (0 = vide)     cells_v2 : 99,8 %
  ├─ orientation (4 rotations) + réparation      grille lue corrigée
  ├─ résolution (backtracking + MRV)             grille complète
  └─ rendu                                       chiffres écrits sur la photo

41 photos de test : 97,6 % des sudokus résolus, 0 solution fausse.
```

Avec ce découpage, quand le résultat final est faux, on peut dire quelle étape a lâché (grille
mal trouvée, chiffre mal lu). Un modèle unique de bout en bout ne le permettrait pas.

## Organisation du code

```
src/sudoku_cv/
  solver.py            résolution et validation d'une grille
  warp.py              coins, redressement 4 points, redressement fin (dewarp)
  cells.py             découpage en cases, prétraitement, heuristique « case remplie »
  digit_classifier.py  enveloppe du modèle YOLO-cls
  grid_detector.py     enveloppe du modèle YOLO detect
  pipeline.py          enchaînement photo -> solution, réparation, rendu
scripts/
  extract_cells.py         debug visuel du redressement sur un dossier de photos
  build_cells_dataset.py   construction du dataset de cases
  train_cells.py           entraînement YOLO-cls + MLflow
  build_grid_dataset.py    construction du dataset de détection de grille
  train_grid.py            entraînement YOLO detect + MLflow (--resume pour reprendre)
  evaluate.py              évaluation de bout en bout + MLflow (pipeline_eval)
  register_models.py       enregistrement des modèles dans le Model Registry (alias champion)
labels/rg1990_grids.json   vérité terrain des photos de test
app/streamlit_app.py       application
tests/                     tests unitaires (pytest)
```

## Commandes

```bash
.venv/Scripts/python -m pytest                                   # tests
.venv/Scripts/python scripts/extract_cells.py <dossier_photos> data/debug
.venv/Scripts/python scripts/build_cells_dataset.py
.venv/Scripts/python scripts/train_cells.py --epochs 15 --name cells_v1
.venv/Scripts/python scripts/train_grid.py --epochs 60
.venv/Scripts/python scripts/evaluate.py --renders
.venv/Scripts/python scripts/register_models.py                 # alias champion dans le registry
.venv/Scripts/streamlit run app/streamlit_app.py               # http://localhost:8501
.venv/Scripts/mlflow ui --backend-store-uri sqlite:///mlflow.db  # http://127.0.0.1:5000
```
