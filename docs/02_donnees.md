# 2. Les données

## Le dépôt rg1990

On est parti de `rg1990/cv-sudoku-solver` (licence MIT), cloné dans
`data/external/cv-sudoku-solver/` et non versionné. On n'utilise que ses données : son code
Python n'est pas exécuté.

| Dossier | Contenu | Utilisation |
|---|---|---|
| `data/sudoku_images/` | 25 photos JPG 2736×3648 | test du redressement, cases réelles, vérité terrain |
| `data/digit_images/Sample002…010` | 9 × 1016 images 128×128 de chiffres imprimés (dataset Chars74K « Fnt ») ; `Sample002` = 1 … `Sample010` = 9 | chiffres 1-9 du classifieur |
| `data/font_exclude/` | 134 images, soit 134 numéros de police écartés par l'auteur (polices fantaisie) | on les exclut aussi |
| `data/sudoku_puzzles/` | 3 fichiers texte, ~30 000 grilles de 81 caractères | tests du solveur, grilles synthétiques |

Le dépôt contenait moins que prévu. Les photos ne sont pas annotées, car l'auteur trouve la
grille par contours OpenCV, sans apprentissage : pour entraîner YOLO detect, il fallait donc
d'autres annotations (voir [doc 5](05_detection_grille.md)). MNIST n'y est pas non plus, l'auteur
le téléchargeait via Keras. Et 25 photos suffisent pour développer et tester, mais pas pour
entraîner un détecteur.

Ces photos montrent bien les difficultés réelles : pages de livre courbées (4, 16, 17, 19…),
chiffres du verso visibles par transparence (photo 1), 3 photos d'écran avec des chiffres rouges
flous et en basse résolution (23, 24, 25), un doigt, un fond encombré, une page voisine.

## Vérité terrain

`labels/rg1990_grids.json` contient les 81 cases de chaque photo de test (0 = vide), lues à la
main sur la grille redressée. Chaque grille saisie passe ensuite dans `is_valid_grid` et `solve`,
et les 5 premières étaient valides et solubles. Une faute de frappe donnerait très probablement
un doublon ou une grille insoluble, donc ce contrôle attrape l'essentiel des erreurs de saisie.

Pour les photos d'entraînement, l'étiquetage est semi-automatique (`scripts/prelabel_cells.py`) :
`cells_v1` propose, on vérifie sur des planches d'images et en contrôlant que la grille lue a
exactement une solution. Il a fallu 1 correction manuelle (photo 23, case (2,0) : 5). Les
étiquettes finales sont dans le même fichier JSON (clés `train` et `test`).

## Découpage train / val / test

On découpe par grille, parce que rg1990 contient plusieurs photos de la même page :

| Grille | Photos |
|---|---|
| A | 1, 22 |
| B | 4, 5 |
| C | 6, 7, 8, 9 |
| D | 18, 19 |
| E | 11, 12 |

| Ensemble | Contenu | Rôle |
|---|---|---|
| test | photos 1, 22, 4, 5, 6, 7, 8, 9, 18, 19, 24 (11 photos, 891 cases, 5 grilles) | mesure finale, jamais vues à l'entraînement |
| train | photos 2, 3, 10-16, 20, 21, 23, 25 (13 photos, toutes les cases étiquetées) + synthétique | apprentissage |
| val | synthétique, polices jamais vues en train | suivi pendant l'entraînement |

Le test couvre les cas difficiles : verso visible (A), pages courbées (B, D), écran rouge (24),
et un cas plus ordinaire (C). La photo 17 est écartée partout : la grille y est trop déformée
(voir [doc 3](03_redressement.md)) et ses cases ne sont pas fiables.

Le train et la val sont séparés par police (132 polices sur 882 réservées à la val). Si une
police était des deux côtés, la validation mesurerait la mémorisation de cette police. Les photos
de test ne donnent aucune case à l'entraînement, pas même les cases vides.

## L'erreur de départ : une fuite de données

Notre premier découpage était fait par photo (test = 1, 4, 8, 19, 24). En pré-étiquetant les
photos d'entraînement, on a vu que 6 d'entre elles montraient les mêmes grilles que le test : même
page, autre angle. `cells_v1` avait donc appris des cases vides de ces pages, et son score de test
était trop optimiste. On a refait le découpage par grille et réentraîné une référence propre,
`cells_v1_clean`.

## Limite

11 photos mais seulement 5 grilles différentes, c'est peu pour conclure. C'est pourquoi le test
final de bout en bout utilise aussi les photos de wichtounet (voir [doc 6](06_pipeline_evaluation.md)).
