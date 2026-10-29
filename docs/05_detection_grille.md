# Étape 5 — Détecter la grille : YOLO detect

Fichiers : `scripts/build_grid_dataset.py` (dataset) — entraînement : à venir

## Pourquoi un détecteur, alors qu'OpenCV trouve déjà la grille ?

`find_grid_corners` cherche « le plus grand quadrilatère de l'image ». Ça marche sur nos 25
photos, mais ça casse dès qu'un autre rectangle est plus grand : page entière, autre grille
(page voisine, solutions du jour précédent), tableau, écran… Un détecteur appris reconnaît
**une grille de sudoku** (9×9 cases, chiffres), pas juste un rectangle. On combine ensuite :
YOLO donne la zone, OpenCV y cherche les coins précis (voir étape 3).

## Dataset

### Sources

| Source | Images | Annotation |
|---|---|---|
| [wichtounet/sudoku_dataset](https://github.com/wichtounet/sudoku_dataset) (CC-BY-4.0) | 202 | 4 coins annotés par l'auteur (`outlines_sorted.csv`) |
| rg1990 | 24 (sans la 17) | **automatique** : intersections extrêmes trouvées par `dewarp_grid`, **vérifiées à l'œil** |

Recherche de datasets : voir `journal.md` (Roboflow Universe, Hugging Face, AICrowd comparés).

### Format

- **1 classe** : `0 = sudoku`.
- bbox = rectangle englobant des 4 coins ; format YOLO `classe cx cy w h` normalisé.
- images réduites à **1280 px** de côté max (les photos rg1990 font 12 Mpx : les décoder à
  chaque époque ralentirait beaucoup l'entraînement CPU ; YOLO travaille de toute façon en 640).
- les 4 coins sont gardés dans `data/grid_det/corners.json` → permet de tester plus tard une
  variante **YOLO-pose** qui prédit directement les 4 coins (piste de recherche).

### Découpage — attention aux doublons de grilles

| Ensemble | Images | dont wichtounet | dont rg1990 | Grilles distinctes |
|---|---|---|---|---|
| train | 156 | 143 | 13 | 43 |
| val | 29 | 29 | 0 | 21 |
| test | 41 | 30 | 11 | 27 |

Deux pièges trouvés et corrigés :

1. **Photos multiples de la même grille dans wichtounet** : 203 photos, seulement
   **77 grilles distinctes** (jusqu'à 18 photos de la même page). Le découpage officiel de
   l'auteur met 15 grilles à la fois en train et en test. → regroupement par grille (les
   chiffres du `.dat` servent d'identifiant), répartition ~70/15/15 du nombre de photos.
2. **Grilles communes aux deux datasets** : les photos rg1990 **24 et 25** montrent des grilles
   présentes aussi dans wichtounet (7 photos). Avec un découpage séparé par dataset, la grille
   de la photo de test 24 se retrouvait en validation via wichtounet. → découpage **global** :
   une grille déjà placée par le découpage rg1990 (étape 4) impose son ensemble à toutes ses
   photos. Contrôle automatique en fin de script : **0 grille dans plusieurs ensembles**.

Les photos de test rg1990 sont les mêmes que pour le classifieur de chiffres : on garde ainsi
un jeu de test **de bout en bout** (photo → solution) jamais vu par aucun des deux modèles.

### Vérification des annotations

Planches de contrôle `data/debug/grid_det_check_*.jpg` (bbox en vert, coins en rouge) :
les 226 annotations ont été regardées. Les 24 annotations automatiques rg1990 sont toutes
correctes (la photo 17, dont le coin est faux, est exclue).

### Observations sur les données (problèmes à anticiper)

- **Cadrage serré** : beaucoup de photos wichtounet sont cadrées sur la grille (bbox ≈ toute
  l'image). Le modèle risque d'apprendre « la grille = toute l'image ». Atténuation : les
  photos rg1990 (scènes larges) et l'augmentation **mosaic** d'Ultralytics (4 images
  assemblées → grilles plus petites, à des positions variées).
- **Grilles tournées de 90°** (wichtounet image1024, 1031-1041) : sans effet sur la détection
  (une bbox n'a pas d'orientation), mais **la lecture des chiffres devra gérer la rotation**
  (piste : essayer les 4 rotations et garder celle où le classifieur est le plus confiant /
  où la grille est valide).
- Beaucoup d'images en **640×480, floues**, chiffres **rouges**, papier journal jauni,
  cases **surlignées** (bandes bleues ou vertes).
- **Peu de grilles distinctes** (77 + 20) : le test (27 grilles) donne une mesure correcte
  pour la détection, mais il faudra des photos personnelles pour valider en conditions réelles.
