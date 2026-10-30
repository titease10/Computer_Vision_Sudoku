# 5. Détecter la grille avec YOLO

Code : `scripts/build_grid_dataset.py`, `scripts/train_grid.py`.

`find_grid_corners` cherche le plus grand quadrilatère de l'image. Ça marche sur nos 25 photos, mais
ça casse dès qu'un autre rectangle est plus grand : page entière, grille voisine, solutions du jour
précédent, tableau, écran. Un détecteur entraîné reconnaît une grille de sudoku, pas seulement un
rectangle. YOLO donne donc la zone, et OpenCV y cherche les coins précis (voir
[doc 3](03_redressement.md)).

## Le dataset

| Source | Images | Annotation |
|---|---|---|
| [wichtounet/sudoku_dataset](https://github.com/wichtounet/sudoku_dataset) (CC-BY-4.0) | 202 | 4 coins annotés par l'auteur (`outlines_sorted.csv`) |
| rg1990 | 24 (sans la 17) | automatique : intersections extrêmes trouvées par `dewarp_grid`, vérifiées à l'œil |

Les autres sources regardées (Roboflow Universe, Hugging Face, AICrowd) sont comparées dans le
[journal](journal.md).

Il y a une seule classe (`0 = sudoku`). Le cadre est le rectangle englobant des 4 coins, au format
YOLO `classe cx cy w h` normalisé. Les images sont réduites à 1280 px de côté max : décoder du 12 Mpx
à chaque époque ralentirait beaucoup l'entraînement sur CPU, et YOLO travaille de toute façon en 640.
Les 4 coins restent dans `data/grid_det/corners.json`, au cas où on testerait une variante YOLO-pose
qui les prédirait directement.

## Le découpage

| Ensemble | Images | dont wichtounet | dont rg1990 | Grilles distinctes |
|---|---|---|---|---|
| train | 156 | 143 | 13 | 43 |
| val | 29 | 29 | 0 | 21 |
| test | 41 | 30 | 11 | 27 |

La fuite de données de rg1990 ([doc 2](02_donnees.md)) se retrouve ici sous deux formes.

D'abord, wichtounet a 203 photos pour seulement 77 grilles distinctes (jusqu'à 18 photos de la même
page), et le découpage officiel met 15 grilles à la fois en train et en test. On regroupe donc par
grille, les chiffres du `.dat` servant d'identifiant, avec une répartition ~70/15/15 du nombre de
photos.

Ensuite, les photos rg1990 24 et 25 montrent des grilles aussi présentes dans wichtounet (7 photos).
Avec un découpage séparé par dataset, la grille de la photo de test 24 se retrouvait en validation.
Le découpage est donc global : une grille déjà placée par le découpage rg1990 impose son ensemble à
toutes ses photos. Le script vérifie à la fin qu'il y a 0 grille dans plusieurs ensembles.

Les photos de test rg1990 sont celles du classifieur, ce qui donne un test de bout en bout jamais vu
par aucun des deux modèles.

Les 226 annotations ont été regardées sur des planches de contrôle
(`data/debug/grid_det_check_*.jpg`, cadre en vert, coins en rouge). Les 24 annotations automatiques
rg1990 sont correctes ; la photo 17, dont le coin est faux, est exclue.

## Ce qu'on a remarqué dans les photos

- Beaucoup de photos wichtounet sont cadrées serré, et le modèle pourrait apprendre que la grille,
  c'est toute l'image. Les scènes larges de rg1990 et l'augmentation mosaic d'Ultralytics (4 images
  assemblées, donc des grilles plus petites et mieux réparties) limitent ce risque.
- Des grilles sont tournées de 90° (wichtounet image1024, 1031-1041). C'est sans effet sur la
  détection, mais pas sur la lecture : c'est ce qui a mené au test des 4 orientations
  ([doc 6](06_pipeline_evaluation.md)).
- Beaucoup d'images sont en 640×480 et floues, avec des chiffres rouges, du papier journal jauni, des
  cases surlignées en bleu ou en vert.
- Il y a peu de grilles distinctes (77 + 20). Le test (27 grilles) donne une mesure correcte de la
  détection, mais pas une validation en conditions réelles.

## Résultats de `grid_v1` (yolo11n, 60 époques, CPU)

Run MLflow `grid_v1`, expérience `grid_detection`. L'entraînement a été interrompu à l'époque 34 puis
repris (voir le [journal](journal.md)). Durée totale ≈ 30 min sur CPU.

![Courbes d'entraînement](img/grid_v1_courbes.png)

| Ensemble | Précision | Rappel | mAP50 | mAP50-95 |
|---|---|---|---|---|
| val (29 images) | 0,985 | 1,00 | 0,995 | 0,913 |
| test (41 images) | 0,997 | 1,00 | 0,995 | 0,921 |

La grille est trouvée sur toutes les images de test. Le mAP50-95, plus exigeant sur l'ajustement du
cadre, progresse peu après l'époque 38 (0,910) : 60 époques suffisent.

Le mAP ne dit pas tout, car la suite du pipeline a besoin des coins. On compte donc une image comme
réussie si ses 4 coins sont à moins de 3 % du côté de la grille :

| Méthode | Réussite | Erreur médiane | wichtounet (30) | rg1990 (11) |
|---|---|---|---|---|
| A. OpenCV seul (référence) | 90,2 % | 1,18 % | 90,0 % | 90,9 % |
| B. YOLO + OpenCV | 95,1 % | 1,12 % | 96,7 % | 90,9 % |
| C. YOLO seul (coins du cadre) | 9,8 % | 4,80 % | 13,3 % | 0 % |

B règle 2 des 4 échecs d'OpenCV seul : la grille collée à la grille des solutions (w_image1077), car
la recherche limitée au cadre ne peut plus fusionner la voisine, et la page à l'envers (w_image1083).

C ne suffit pas, car un cadre est un rectangle droit et une grille photographiée n'en est pas un. Le
cadre seul ne sert que de dernier repli.

B a encore 2 échecs. `rg_18` (4,9 %) est une page courbée : aucun quadrilatère ne colle parfaitement,
et c'est `dewarp_grid` qui compense ensuite. `w_image17` (4,0 %) est une grille coupée par le bord de
la photo. OpenCV seul n'y trouvait aucun contour fermé, et YOLO donne des coins approximatifs, juste
au-dessus du seuil. De toute façon, une grille qui sort du cadre ne peut pas être lue en entier.

L'objectif de détection (mAP50 > 0,95) est atteint. Le pipeline enchaîne donc YOLO, OpenCV dans le
cadre, puis les coins du cadre en repli.
