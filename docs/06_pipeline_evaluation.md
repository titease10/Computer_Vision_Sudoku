# 6. Pipeline complet et évaluation

`src/sudoku_cv/pipeline.py` enchaîne toutes les briques :

```
photo ─► YOLO detect (cadre) ─► coins OpenCV dans le cadre ─► redressement fin (dewarp)
      ─► 81 cases ─► YOLO-cls ×4 orientations ─► meilleure orientation
      ─► réparation par les règles du sudoku ─► solveur ─► chiffres écrits sur la photo
```

`SudokuPipeline(grid_weights, digit_weights).run(image)` renvoie un `PipelineResult` : statut
(`solved`, `unsolvable`, `no_grid`), cadre, rotation, lecture brute, lecture réparée, confiance par
case, solution, temps de chaque étape. `render_solution(image, result)` écrit les chiffres sur la
photo, et `scripts/evaluate.py` mesure le tout sur le jeu de test (expérience MLflow `pipeline_eval`).

La première mesure de bout en bout donnait 85,4 %. Deux ajouts ont fait la différence.

## Choisir l'orientation

Certaines photos wichtounet montrent la grille tournée de 90° ou à l'envers (journal photographié de
côté). Les coins sont ordonnés selon la photo, donc la grille redressée garde cette rotation : le
classifieur lit des chiffres couchés et se trompe sur environ 30 cases (w_image1041, 1043, 1083).

On classe donc les 81 cases dans les 4 orientations (0°, 90°, 180°, 270°) et on garde celle dont la
confiance moyenne sur les cases lues comme chiffres est la plus haute. À l'envers, les 2, 3, 4, 5 et 7
deviennent méconnaissables et la confiance chute. Les cases vides ne comptent pas : elles sont sûres
dans toutes les orientations et masqueraient la différence.

Coût : 4 × 81 petites images 64×64, soit environ +45 ms par photo.

## Réparer avec les règles

Une vraie grille de journal a exactement une solution. Si la lecture brute a un doublon ou n'a pas de
solution unique, au moins une case est mal lue. `repair_grid` prend alors les 8 cases les moins sûres
et essaie d'en remplacer 1 ou 2 par le 2ᵉ ou le 3ᵉ choix du classifieur. Les combinaisons sont testées
de la plus probable à la moins probable (coût = perte de log-vraisemblance), et la première qui donne
une solution unique est retenue. Un correcteur orthographique fait à peu près pareil.

Le risque est de tomber par hasard sur une mauvaise grille à solution unique. On le surveille avec
`wrong_solution_rate`, car une réponse fausse donnée avec assurance est pire qu'un aveu d'échec.

Tests : `tests/test_pipeline.py` (grille correcte inchangée, une erreur réparée, deux erreurs réparées,
abandon si les erreurs sont lues avec assurance).

## Résultats sur 41 photos de test

30 photos wichtounet et 11 photos rg1990, jamais vues par l'un ou l'autre modèle. On ajoute les
briques une à une pour voir ce que chacune apporte :

| Configuration | Grille trouvée | Cases justes | Grilles lues parfaitement | Sudokus résolus | Solutions fausses | Temps |
|---|---|---|---|---|---|---|
| OpenCV seul, lecture brute | 97,6 % | 94,9 % | 85,4 % | 85,4 % | 0 % | 0,19 s |
| + détecteur YOLO | 100 % | 96,9 % | 85,4 % | 85,4 % | 0 % | 0,12 s |
| + choix de l'orientation | 100 % | 99,4 % | 92,7 % | 92,7 % | 0 % | 0,17 s |
| + réparation (pipeline complet) | 100 % | 99,4 % | 92,7 % | 97,6 % | 0 % | 0,16 s |

Par source, avec le pipeline complet : wichtounet 96,7 % (29/30), rg1990 100 % (11/11). L'objectif de
90 % est atteint, avec 40 photos sur 41.

![Exemples de rendus](img/pipeline_rendus.jpg)

*De gauche à droite : rg_1 (cas normal), w_image1041 (grille tournée de 90°), rg_18 (page courbée, une
case réparée), w_image1083 (page à l'envers). En vert, les chiffres trouvés.*

YOLO ne change pas le nombre de sudokus résolus sur ce test. Il trouve quand même la grille sur 100 %
des photos (contre 97,6 %) et améliore les cases justes (94,9 % → 96,9 %), grâce à de meilleurs coins
(95,1 % contre 90,2 %, voir [doc 5](05_detection_grille.md)). Il accélère aussi OpenCV, qui cherche
dans une zone plus petite.

L'orientation règle les 3 grilles tournées. La réparation règle 2 photos à une seule case fausse :
`rg_18`, où une case vide est lue « 1 » parce que le bord de la grille entre dans la case, et
`w_image46`, où un 1 est lu « 4 ». Aucune réparation n'a produit de mauvaise solution sur ce test.
Ce n'est pas une garantie : le [journal](journal.md) raconte une photo personnelle où c'est arrivé.

Reste `w_image17`, une grille coupée par le bord de la photo. Le détecteur la trouve, mais une partie
des cases est hors de l'image : 17 cases fausses, la plupart des chiffres lus comme vides. La
réparation abandonne et le pipeline répond « insoluble », ce qui est le comportement voulu.

## Photo à plusieurs grilles

Une double page de magazine contient souvent 2 grilles. Le pipeline récupère toutes les détections
YOLO, les essaie de la plus sûre à la moins sûre et garde la première résolue (voir le
[journal](journal.md)).

## Limites

- 41 photos et 27 grilles distinctes : chaque photo pèse 2,4 points. Les quelques photos personnelles
  testées ont déjà révélé des cas absents de ce jeu, donc il en faudrait beaucoup plus.
- Les chiffres de la solution sont écrits droits par rapport à la photo, même quand la grille est
  tournée. C'est plus lisible, mais ce n'est pas le sens de la grille.
- Une erreur de lecture qui donne quand même une grille à solution unique ne peut pas être détectée.

## Commandes

```bash
.venv/Scripts/python scripts/evaluate.py              # les 4 configurations
.venv/Scripts/python scripts/evaluate.py --configs full --renders   # + photos annotées dans runs/eval/
```

## Model Registry MLflow

Les deux modèles retenus sont enregistrés dans le Model Registry (`scripts/register_models.py`), avec
l'alias `champion` :

| Modèle du registry | Version | Run d'origine | Rôle |
|---|---|---|---|
| `sudoku-grid-detector` | v1 | `grid_v1` | cadre de la grille |
| `sudoku-digit-classifier` | v1 | `cells_v2` | contenu des cases |

L'application et l'évaluation chargent `models:/<nom>@champion` (`SudokuPipeline.from_registry(...)`),
sans chemin de fichier ni id de run en dur. Pour changer de modèle, on enregistre une nouvelle version
et on déplace l'alias, sans toucher au code et en gardant l'historique.

Les poids Ultralytics (`.pt`) ne sont pas un format que MLflow connaît. On enregistre donc le dossier
`weights/` du run comme version, et on télécharge `best.pt` au chargement.

`evaluate.py --configs full --from-registry` redonne 97,6 % des sudokus résolus, comme avec les chemins
locaux.
