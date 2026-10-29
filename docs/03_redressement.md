# Étape 3 — Localiser, redresser et découper la grille

Fichiers : `src/sudoku_cv/warp.py`, `src/sudoku_cv/cells.py`, `scripts/extract_cells.py`
Tests : `tests/test_warp.py`

Objectif : passer d'une photo prise de travers à une image carrée 450×450 où la grille est
parfaitement alignée, pour pouvoir la couper en 9×9 cases régulières.

## 3.1 Trouver les 4 coins — `find_grid_corners`

### Méthode

1. Niveaux de gris, puis **réduction à 1000 px max** (les photos font 12 Mpx : traiter en
   pleine résolution est lent sans rien apporter pour trouver un contour).
2. **Flou gaussien** 7×7 : supprime le grain du papier et le bruit du capteur.
3. **Seuillage adaptatif** : chaque pixel est comparé à la moyenne de son voisinage, et non à
   un seuil global. Indispensable car l'éclairage varie sur la photo (ombres, dégradés).
4. **Dilatation** 3×3 : referme les petites coupures dans les traits de la grille.
5. **Contours externes**, triés par aire décroissante ; on garde le premier qui, simplifié par
   `approxPolyDP`, est un **quadrilatère convexe** d'au moins 10 % de l'image.
6. Les 4 points sont remis à l'échelle de la photo d'origine et **ordonnés** (haut-gauche,
   haut-droite, bas-droite, bas-gauche) : le plus petit `x+y` est en haut à gauche, le plus
   grand en bas à droite, et `y−x` départage les deux autres.

### Lien avec YOLO (étape à venir)

La fonction accepte une **bbox** (venant du détecteur YOLO) : la recherche se fait alors
seulement dans cette zone (+5 % de marge). Si aucun quadrilatère n'y est trouvé, on utilise
les coins de la bbox. YOLO apporte la robustesse (trouver la grille dans une scène chargée),
OpenCV la précision (le coin exact au pixel près).

## 3.2 Redressement simple par 4 points — `warp_grid`

`getPerspectiveTransform` calcule l'homographie qui envoie les 4 coins sur les 4 coins d'un
carré 450×450 (9 cases de 50 px), puis `warpPerspective` applique la transformation.
La matrice inverse est conservée pour reprojeter la solution sur la photo.

### Problème rencontré : les pages courbées

Une homographie suppose que la grille est **plane**. Sur un livre ouvert, la page est
courbée : après redressement, les traits de la grille restent **courbes**. En découpant en
cases régulières, des morceaux de traits tombent dans les cases.

Mesure (nombre de cases détectées comme remplies, vérité comptée à la main) :

| Photo | Redressement 4 points | Vérité |
|---|---|---|
| 4 | 35 | 28 |
| 19 | 35 | 28 |
| 17 | 47 | ? (beaucoup moins) |

## 3.3 Redressement fin — `dewarp_grid` (solution retenue)

Idée : au lieu de se fier à 4 points, on **retrouve les 20 traits de la grille** (10
horizontaux, 10 verticaux) et leurs **100 intersections**, puis on redresse **case par
case**.

1. Redressement grossier par 4 points, avec **10 % de marge** autour du carré (sinon un coin
   mal placé coupe une partie de la grille, qu'on ne pourrait plus retrouver).
2. Seuillage adaptatif, puis **ouverture morphologique** avec un élément long et fin
   (horizontal pour un masque, vertical pour l'autre) : ne survivent que les segments plus
   longs que 60 % d'une case, c'est-à-dire les traits de la grille ; les chiffres
   disparaissent.
3. Pour chaque trait k, on parcourt l'image tous les 4 px et on cherche, dans une fenêtre de
   ±40 % de case autour de la position théorique, la position de réponse maximale.
4. On ajuste une **parabole** (degré 2) sur ces points, avec un **rejet des points
   aberrants** (écart > 4 px) puis un second ajustement. Une parabole suffit pour une page
   qui se courbe régulièrement.
5. Les intersections trait horizontal × trait vertical se calculent par itération de point
   fixe (`y = h(x)`, `x = v(y)`, 5 itérations).
6. `cv2.remap` : chaque pixel de la grille de sortie est placé par **interpolation
   bilinéaire** entre les 4 intersections de sa case.

Les 100 intersections sont aussi renvoyées en coordonnées de la photo d'origine : elles
serviront à dessiner la solution au bon endroit, même sur une page courbée.

### Résultats

| Photo | 4 points | Redressement fin | Vérité |
|---|---|---|---|
| 19 | 35 | **28** | 28 ✅ |
| 16 | 34 | **27** | — |
| 4 | 35 | 32 | 28 |
| 8 | 28 | 29 | 28 |
| 17 | 47 | 34 | — |

25 grilles trouvées sur 25.

### Problèmes restants

- **Photo 17** : courbure forte **et** coin bas-gauche mal trouvé par le contour ; le bas de
  la grille reste déformé. Pistes : coins fournis par YOLO ; suivi des traits de proche en
  proche (au lieu d'une fenêtre fixe autour de la position théorique) ; polynôme de degré 3.
- Photos 4 et 8 : quelques cases vides comptées comme remplies (restes de traits sur les
  bords des cases), voir 3.5.

## 3.4 Découpage en cases — `split_cells`, `preprocess_cell`

- La grille 450×450 est coupée en 9×9 ; on **rogne 12 %** de chaque côté de la case pour
  enlever les traits.
- `preprocess_cell` : redimensionne en 64×64 et garde le **canal le plus sombre** (min des
  canaux B, G, R) au lieu du niveau de gris classique. Raison : une encre rouge en niveaux
  de gris donne un gris moyen (~110 sur un fond à ~200), peu contrasté ; dans le canal le
  plus sombre elle vaut ~70, aussi contrastée qu'une encre noire.

## 3.5 Heuristique « case remplie » — `cell_has_ink`

Règle simple, utilisée pour le **pré-tri** des cases (pas pour la lecture finale) : une case
est remplie si au moins 3 % de sa zone centrale est **plus sombre d'au moins 30 niveaux** que
le fond de la case (médiane).

### Pistes essayées

| Version | Règle | Problème |
|---|---|---|
| v1 | pixel < 0,6 × fond (niveaux de gris) | photos 23-25 (chiffres rouges) : **0** case remplie trouvée |
| **v2 (retenue)** | pixel < fond − 30, sur le canal le plus sombre | — |

Choix du seuil, mesuré sur les 25 photos (seuils 30 / 40 / 50 / 60) :

| Photo | 30 | 40 | 50 | 60 | Vérité |
|---|---|---|---|---|---|
| 1 (verso visible) | 27 | 27 | 27 | 27 | 27 |
| 23 (rouge) | **27** | 25 | 5 | 1 | 27 |
| 25 (rouge) | **26** | 19 | 2 | 0 | 26 |

Les chiffres fantômes du verso sont ~25 niveaux plus sombres que le fond, les chiffres rouges
flous ~60, les chiffres noirs ~180 : un seuil de 30 sépare bien les cas.

## Tests (photo synthétique)

Une grille est dessinée, quelques chiffres placés, puis l'image est déformée en perspective
sur un fond gris. On vérifie : ordre des coins, coins retrouvés à moins de 8 px (avec et sans
bbox), repli sur la bbox, redressement + découpage + cases remplies exactes, et les 100
intersections du redressement fin.

Debug visuel : `scripts/extract_cells.py` écrit pour chaque photo le contour (rouge) et les
intersections (vert) dans `data/debug/corners/`, la grille redressée dans
`data/debug/warped/`.
