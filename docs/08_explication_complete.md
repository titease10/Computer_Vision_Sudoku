# Explication complète du projet, étape par étape

Ce document explique comment le projet fonctionne de la photo à la solution : ce que fait chaque
étape, les notions utilisées et les fonctions appelées. Pour l'historique des essais et des
problèmes, voir [journal.md](journal.md) ; pour le détail de chaque étape, les docs 01 à 07.

**Sommaire**

- [Vue d'ensemble](#vue-densemble)
- [Étape 1 : détecter la grille](#étape-1--détecter-la-grille-yolo-detect)
- [Étape 2 : trouver les 4 coins](#étape-2--trouver-les-4-coins-opencv)
- [Étape 3 : vérifier les coins](#étape-3--vérifier-les-coins-avec-les-traits)
- [Étape 4 : redresser la grille](#étape-4--redresser-la-grille)
- [Étape 5 : découper en 81 cases](#étape-5--découper-en-81-cases)
- [Étape 6 : lire les cases](#étape-6--lire-chaque-case-yolo-cls)
- [Étape 7 : choisir l'orientation](#étape-7--choisir-lorientation)
- [Étape 8 : réparer la lecture](#étape-8--réparer-la-lecture)
- [Étape 9 : résoudre](#étape-9--résoudre-le-sudoku)
- [Étape 10 : afficher](#étape-10--écrire-la-solution-sur-la-photo)
- [Avant le pipeline : données et entraînement](#avant-le-pipeline--données-et-entraînement)
- [Évaluation et MLflow](#évaluation-et-mlflow)
- [Application Streamlit](#application-streamlit)
- [Limites à connaître](#limites-à-connaître)
- [Glossaire](#glossaire)

---

## Vue d'ensemble

```
Photo
  │
  ├─ 1. Détecter la grille           YOLO detect    → rectangle (bbox)
  ├─ 2. Trouver les 4 coins          OpenCV         → quadrilatère
  ├─ 3. Vérifier les coins           OpenCV         → quadrilatère corrigé si besoin
  ├─ 4. Redresser                    OpenCV         → grille plate 450×450 + 100 intersections
  ├─ 5. Découper                     NumPy/OpenCV   → 81 images 64×64
  ├─ 6. Lire les cases               YOLO-cls       → 81 × 10 probabilités
  ├─ 7. Choisir l'orientation        logique        → meilleure des 4 rotations
  ├─ 8. Réparer la lecture           règles sudoku  → grille à solution unique
  ├─ 9. Résoudre                     backtracking   → grille complète
  └─ 10. Écrire la solution          OpenCV         → photo annotée
```

**L'idée directrice :** on utilise l'apprentissage seulement là où des règles écrites à la main
échouent, c'est-à-dire pour reconnaître une grille dans une scène et lire un chiffre. La géométrie
et la résolution sont déterministes : elles n'ont pas besoin de données et se vérifient facilement.

### Qui appelle quoi

```
SudokuPipeline.run(image)                                 pipeline.py
├─ GridDetector.detect_all(image)                         grid_detector.py  (étape 1)
└─ pour chaque grille détectée : _run_grid(image, bbox, conf)
   ├─ read_cells(image, bbox)
   │  ├─ find_grid_corners(image, bbox)                   warp.py           (étape 2)
   │  │  ├─ _find_quad(gray, min_area_ratio)
   │  │  └─ refine_corners(image, corners)                                  (étape 3)
   │  │     ├─ _line_masks(coarse, cell)
   │  │     ├─ _grid_lines(mask, pad, size, cell)
   │  │     │  ├─ _peaks(profile, min_value, merge_dist)
   │  │     │  └─ _regular_tens(peaks, cell)
   │  │     └─ order_corners(pts)
   │  ├─ dewarp_grid(image, corners)                                        (étape 4)
   │  │  ├─ _line_masks(coarse, cell)
   │  │  └─ _track_lines(mask, pad, cell, axis)
   │  │     └─ _fit_line(samples, expected)
   │  ├─ split_cells(flat)                                cells.py          (étape 5)
   │  └─ preprocess_cell(cell)
   ├─ DigitClassifier.predict_probs(cells)  × 4 rotations digit_classifier.py (étape 6)
   ├─ _grid_score(probs)                    × 4 rotations pipeline.py       (étape 7)
   ├─ repair_grid(probs)                                                    (étape 8)
   │  ├─ count_solutions(grid)                            solver.py
   │  └─ is_valid_grid(grid)
   └─ solve(grid)                                                           (étape 9)

render_solution(image, result)                            pipeline.py       (étape 10)
```

`run` renvoie un objet `PipelineResult` : statut (`solved`, `unsolvable` ou `no_grid`), bbox,
rotation, lecture brute, lecture réparée, confiances, solution, intersections et temps de calcul.

---

## Étape 1 : détecter la grille (YOLO detect)

**Fichier :** `src/sudoku_cv/grid_detector.py`  
**Entrée :** la photo en BGR. **Sortie :** une liste de `(bbox [x1, y1, x2, y2], confiance)`.

**Rôle :** trouver où se trouve la grille, même dans une scène chargée. Chercher simplement « le
plus grand rectangle » échoue dès qu'une page, un tableau ou une seconde grille est présent.

**Comment ça marche :**
1. L'image est redimensionnée à 640 px par Ultralytics (`imgsz=640`).
2. Le réseau `yolo11n` la traverse en une seule passe et propose des rectangles, chacun avec un score.
3. Les propositions sous `conf=0.25` sont écartées, et la suppression des non-maxima (NMS) retire
   les doublons qui se chevauchent.
4. `detect_all` trie les grilles de la plus sûre à la moins sûre. Une double page de magazine peut
   en contenir deux : le pipeline les essaie dans cet ordre et garde la première résolue.

Si YOLO ne trouve rien, le pipeline continue quand même, avec OpenCV sur toute l'image.

**Notions :** détection d'objets, boîte englobante, score de confiance, NMS, IoU.

| Fonction | Rôle |
|---|---|
| `GridDetector.__init__` | charge les poids avec `YOLO(weights, task="detect")` |
| `GridDetector.detect_all` | appelle `model.predict` et renvoie toutes les grilles triées |
| `GridDetector.detect` | renvoie seulement la plus sûre (utilisée par `train_grid.py`) |

**Résultat (test, 41 images) :** précision 0,997, rappel 1,00, mAP50 0,995, mAP50-95 0,921.

---

## Étape 2 : trouver les 4 coins (OpenCV)

**Fichier :** `src/sudoku_cv/warp.py`, fonctions `find_grid_corners` et `_find_quad`  
**Entrée :** photo + bbox. **Sortie :** 4 coins dans l'ordre haut-gauche, haut-droite, bas-droite,
bas-gauche.

**Rôle :** une bbox est un rectangle droit, alors qu'une grille photographiée est un quadrilatère
quelconque à cause de la perspective. Il faut donc les vrais coins. La mesure le confirme : les
coins de la bbox seule ne réussissent que 9,8 % des images, contre 95,1 % avec YOLO + OpenCV.

**Comment ça marche :**

| # | Opération | Appel OpenCV | Pourquoi |
|---|---|---|---|
| 1 | Découper la bbox avec 5 % de marge | slicing NumPy | la bbox peut rogner les bords |
| 2 | Passer en niveaux de gris | `cv2.cvtColor` | la couleur n'aide pas à trouver les traits |
| 3 | Réduire à 1000 px maximum | `cv2.resize` (`INTER_AREA`) | plus rapide sur les photos de 12 Mpx |
| 4 | Flou gaussien 7×7 | `cv2.GaussianBlur` | enlève le grain du papier et le bruit |
| 5 | Seuillage adaptatif (bloc 11, C = 2) | `cv2.adaptiveThreshold` | chaque pixel est comparé à son voisinage, ce qui résiste aux ombres |
| 6 | Dilatation 3×3 | `cv2.dilate` | referme les petites coupures des traits |
| 7 | Contours externes, triés par aire | `cv2.findContours`, `cv2.contourArea` | le contour de la grille est parmi les plus grands |
| 8 | Simplification du contour (tolérance 2 % du périmètre) | `cv2.arcLength`, `cv2.approxPolyDP` | réduit le contour à quelques sommets |
| 9 | Garder le premier à 4 sommets, convexe, assez grand | `cv2.isContourConvex` | aire ≥ 10 % de la zone |

Si aucun contour ne convient avec la dilatation, on réessaie sans elle : sur une petite image, la
dilatation peut souder la grille au bord de la page. Si cela échoue aussi, on prend les coins de la
bbox en dernier recours.

**Ordonner les coins (`order_corners`) :** les 4 points sortent dans un ordre quelconque. On utilise
deux calculs simples :

| Coin | Critère |
|---|---|
| haut-gauche | plus petit `x + y` |
| bas-droite | plus grand `x + y` |
| haut-droite | plus petit `y − x` |
| bas-gauche | plus grand `y − x` |

**Notions :** niveaux de gris, filtre gaussien, seuillage adaptatif, morphologie mathématique
(dilatation), contours, approximation polygonale (Douglas-Peucker), convexité.

---

## Étape 3 : vérifier les coins avec les traits

**Fichier :** `warp.py`, fonction `refine_corners` (appelée à la fin de `find_grid_corners`)  
**Entrée :** photo + 4 coins. **Sortie :** les mêmes coins, ou des coins corrigés.

**Rôle :** le contour extérieur peut englober autre chose que la grille, par exemple un bandeau
coloré ou une grille voisine collée. Une grille a pourtant une signature très reconnaissable :
**10 traits régulièrement espacés dans chaque direction**. On s'en sert pour vérifier les coins.

**Comment ça marche :**
1. **Redressement grossier** avec 15 % de marge (`getPerspectiveTransform`, `warpPerspective`).
2. **Masques de traits** (`_line_masks`) : seuillage adaptatif sur le canal le plus sombre, puis
   **ouverture morphologique** (`cv2.morphologyEx`, `MORPH_OPEN`) avec un élément long et fin de
   60 % d'une case. Seuls les segments plus longs que cet élément survivent, donc les traits de
   la grille restent et les chiffres disparaissent.
3. **Profil de projection** (`_grid_lines`) : dans 3 bandes (à 20 %, 50 % et 80 % de la grille), on
   compte les pixels blancs par colonne. Chaque trait donne un pic.
4. **Extraction des pics** (`_peaks`) : on garde les plages au-dessus du seuil, et on fusionne les
   pics à moins d'un quart de case, car un trait fin peut être coupé en deux par le seuillage.
5. **Choix des 10 bons pics** (`_regular_tens`) : on propose des suites de 10 pics et on calcule un
   coût pour chacune. Plus le coût est bas, mieux c'est :

   ```
   coût = écart à une courbe de degré 2       (l'espacement doit être régulier)
        + 0,1 × nombre de pics sautés         (sauter un vrai trait est suspect)
        − 0,5 × surépaisseur des traits 0, 3, 6, 9   (les bords de blocs sont plus épais)
   ```

6. **Décision :** si au moins 2 bandes confirment les coins dans chaque direction, ou si les traits
   ne sont pas retrouvés, on **garde le contour**. Sinon on calcule les intersections des traits
   extrêmes et on les reprojette dans la photo (`cv2.perspectiveTransform`, `np.linalg.inv`).

**Notions :** ouverture morphologique (érosion puis dilatation), élément structurant, profil de
projection, détection de pics, régression polynomiale (`np.polyfit`), fonction de coût.

> **Pourquoi seulement en cas de contradiction ?** Appliqué partout, l'affinage dégradait les bons
> coins (erreur médiane de 1,1 % à 1,9 %) et les sudokus résolus tombaient à 90,2 %. La version
> finale garde 97,6 % et règle le cas difficile. Les 4 essais sont dans le journal.

---

## Étape 4 : redresser la grille

**Fichier :** `warp.py`, fonction `dewarp_grid`  
**Entrée :** photo + 4 coins. **Sortie :** image plate 450×450 (9 cases d'environ 50 px) + 100
intersections dans les coordonnées de la photo.

**Rôle :** obtenir une grille vue de dessus et parfaitement alignée pour pouvoir la découper
régulièrement.

### Pourquoi une homographie ne suffit pas

Une **homographie** est une transformation de plan à plan : une matrice 3×3 à 8 degrés de liberté,
entièrement fixée par 4 correspondances de points. Elle corrige la perspective, mais suppose une
surface **plane**. Sur un livre ouvert, la page est courbée : les traits restent courbes après
redressement et tombent dans les cases. Mesure sur la photo 19 : 35 cases comptées comme remplies
avec les 4 points seuls, 28 (la vérité) avec le redressement fin.

### Comment ça marche

1. **Redressement grossier** par homographie avec 10 % de marge :
   `cv2.getPerspectiveTransform`, puis `cv2.warpPerspective`.
2. **Masques de traits** avec `_line_masks`, comme à l'étape 3.
3. **Suivi des 20 traits** (`_track_lines`) : pour chaque trait, on avance tous les 4 px et on cherche
   la réponse maximale dans une fenêtre de ±40 % de case autour de la position théorique.
4. **Ajustement d'une parabole** (`_fit_line`) sur ces points avec `np.polyfit` (degré 2). Les points
   à plus de 4 px sont rejetés, puis on ajuste à nouveau. S'il y a moins de 8 points, le trait reste
   à sa position théorique.
5. **100 intersections** par itération de point fixe : `y = h(x)`, puis `x = v(y)`, 5 fois. Cela
   converge vite, car les deux courbes sont presque perpendiculaires.
6. **Carte de remappage** : chaque pixel de sortie est placé par **interpolation bilinéaire** entre
   les 4 intersections de sa case :

   ```
   src = (1−B)·[(1−A)·p00 + A·p01] + B·[(1−A)·p10 + A·p11]
   ```

   `A` et `B` sont la position du pixel dans sa case (entre 0 et 1), et `p00…p11` ses 4 coins.
7. **Application** avec `cv2.remap`. La carte indique, pour chaque pixel de sortie, où aller chercher
   sa couleur. Partir de la sortie évite les trous dans l'image.
8. **Retour à la photo** : les intersections sont reprojetées avec la matrice inverse
   (`cv2.perspectiveTransform`). Elles serviront à l'étape 10.

**Notions :** homographie, coordonnées homogènes, régression polynomiale, rejet de points aberrants,
point fixe, interpolation bilinéaire, remappage inverse.

> `warp_grid`, le redressement simple par 4 points, n'est plus utilisé par le pipeline. Il reste
> dans le code pour les tests et la comparaison.

---

## Étape 5 : découper en 81 cases

**Fichier :** `src/sudoku_cv/cells.py`  
**Entrée :** image 450×450. **Sortie :** 81 images 64×64, ligne par ligne.

- **`split_cells`** coupe l'image en 9×9 et rogne **12 %** de chaque côté de la case pour enlever
  les traits de grille.
- **`preprocess_cell`** garde le **canal le plus sombre** (minimum de B, G et R) et redimensionne en
  64×64 avec `cv2.resize`. Enfin, `cv2.cvtColor` recopie ce canal sur 3 canaux, car YOLO attend une
  image à 3 canaux.

**Pourquoi le canal le plus sombre ?** Une encre rouge passée en niveaux de gris classiques reste
peu contrastée (environ 110 sur un fond à 200). Dans le canal le plus sombre, elle vaut environ 70,
donc presque autant qu'une encre noire. Le modèle, entraîné en noir et blanc, lit ainsi aussi les
chiffres colorés.

**Notions :** découpage d'image, canaux de couleur, interpolation par aire (`INTER_AREA`).

---

## Étape 6 : lire chaque case (YOLO-cls)

**Fichier :** `src/sudoku_cv/digit_classifier.py`  
**Entrée :** 81 cases. **Sortie :** matrice `(81, 10)` de probabilités.

**Rôle :** dire ce que contient chaque case, parmi 10 classes : `0` = vide et `1` à `9` = chiffres.

**Comment ça marche :**
1. `DigitClassifier.predict_probs` envoie les cases à `model.predict(..., imgsz=64)`.
2. Le réseau `yolo11n-cls` (environ 1,5 million de paramètres) extrait des caractéristiques avec ses
   couches de convolution, puis produit un score par classe.
3. Une fonction **softmax** transforme ces scores en probabilités dont la somme vaut 1.
4. Le code remet les colonnes dans l'ordre des chiffres avec `class_to_digit`, car l'ordre interne
   du modèle vient des noms de dossiers.

**Pourquoi « vide » est une classe à part entière ?** Une règle de seuil confondrait les chiffres
fantômes du verso, les restes de traits ou les taches avec de vrais chiffres. Le modèle apprend à
les ignorer à partir d'exemples. C'est aussi ce qui permet de garder la matrice de probabilités
complète, indispensable aux étapes 7 et 8.

**Notions :** classification d'images, CNN, softmax, transfert d'apprentissage.

**Résultat (`cells_v2`, 891 vraies cases) :** 99,78 % de cases justes, 100 % des chiffres, 2 erreurs.

---

## Étape 7 : choisir l'orientation

**Fichier :** `pipeline.py`, fonctions `_run_grid` et `_grid_score`

**Problème :** les coins sont ordonnés selon la photo, pas selon la grille. Une grille photographiée
de côté reste donc tournée après redressement, et le classifieur lit des chiffres couchés.

**Comment ça marche :**
1. On tourne la grille plate de 0°, 90°, 180° et 270° (`np.rot90`).
2. On recoupe et on reclasse les 81 cases pour chaque rotation.
3. On calcule un score pour chacune et on garde le plus haut :

   ```
   score = moyenne de log(probabilité du meilleur choix), sur les cases lues comme chiffres
   ```

**Pourquoi exclure les cases vides ?** Elles restent vides dans toutes les orientations, avec une
forte confiance, et masqueraient la différence. Une grille à l'envers donne au contraire des 2, 3,
4, 5 et 7 méconnaissables, donc une confiance plus basse.

Les 100 intersections subissent la même rotation, pour que la solution s'écrive dans les bonnes
cases. **Apport mesuré :** les sudokus résolus passent de 85,4 % à 92,7 %, pour environ 45 ms de plus.

**Notions :** log-vraisemblance, inférence multi-hypothèses.

---

## Étape 8 : réparer la lecture

**Fichier :** `pipeline.py`, fonction `repair_grid`  
**Entrée :** matrice `(81, 10)` de probabilités. **Sortie :** grille + liste des changements, ou `None`.

**Idée :** une vraie grille de journal a **exactement une solution**. Si la lecture n'en a pas, au
moins une case est fausse. Les règles du sudoku servent donc de correcteur, à la manière d'un
correcteur orthographique.

**Comment ça marche :**
1. On teste la lecture brute avec `count_solutions`. S'il y a une seule solution, on ne change rien.
2. Sinon, on prend les **8 cases les moins sûres**.
3. On génère les remplacements de **1 ou 2 cases** par leur 2ᵉ ou 3ᵉ choix du classifieur, soit
   16 + 112 = **128 candidats**.
4. On les trie par coût, c'est-à-dire par perte de log-vraisemblance :

   ```
   coût = somme sur les cases changées de [log p(valeur lue) − log p(nouvelle valeur)]
   ```

5. On garde le premier candidat valide (`is_valid_grid`) qui a une solution unique (`count_solutions`).
   Si aucun ne convient, la fonction renvoie `None` et le statut devient `unsolvable`.

**Apport mesuré :** de 92,7 % à 97,6 % des sudokus résolus sur le test.

**Risque :** la réparation peut tomber sur une grille fausse qui a, par hasard, une solution unique.
C'est arrivé sur une photo personnelle. La vraie correction était alors géométrique (étape 3).
Pour surveiller ce risque, on mesure `wrong_solution_rate`.

**Notions :** contraintes comme contrôle qualité, recherche ordonnée, log-vraisemblance.

---

## Étape 9 : résoudre le sudoku

**Fichier :** `src/sudoku_cv/solver.py`

**Comment ça marche, avec `_search` :**
1. On note les chiffres présents dans chaque ligne, colonne et bloc, avec des ensembles Python.
2. Les candidats d'une case valent `{1..9} − ligne − colonne − bloc`.
3. **Backtracking :** on pose un candidat, on continue récursivement et, en cas d'impasse, on revient
   en arrière et on essaie le suivant.
4. **Heuristique MRV :** on traite d'abord la case qui a le **moins de candidats**. Une case à un seul
   candidat se remplit sans risque, et une case à zéro candidat révèle l'impasse tout de suite. L'arbre
   de recherche se réduit beaucoup : une grille réputée très difficile comme AI Escargot est résolue en
   quelques millisecondes.

Le numéro du bloc se calcule avec `3 * (r // 3) + c // 3`.

| Fonction | Rôle |
|---|---|
| `is_valid_grid` | vérifie l'absence de doublon dans les lignes, colonnes et blocs |
| `solve` | renvoie la première solution, ou `None` (`_search` avec `limit=1`) |
| `count_solutions` | compte les solutions, en s'arrêtant à 2 : c'est suffisant pour savoir si la solution est unique |
| `_search` | implémente le backtracking + MRV |

**Notions :** satisfaction de contraintes, récursivité, backtracking, heuristique MRV.

---

## Étape 10 : écrire la solution sur la photo

**Fichier :** `pipeline.py`, fonction `render_solution`

Pour chaque case vide de la grille lue :
1. Le **centre** de la case est la moyenne de ses 4 intersections.
2. La **taille** du chiffre est proportionnelle au côté de la case (`np.linalg.norm`), ce qui respecte
   la perspective.
3. `cv2.getTextSize` donne les dimensions du texte pour le centrer, puis `cv2.putText` l'écrit en vert.

Grâce aux intersections suivies à l'étape 4, les chiffres tombent au bon endroit même sur une page
courbée. Ils restent cependant écrits droits : sur une grille tournée, ils ne suivent pas le sens de
la grille.

---

## Avant le pipeline : données et entraînement

### Découper par grille, pas par photo

Le même sudoku apparaît sur plusieurs photos (autre angle, autre éclairage). Avec un découpage par
photo, le modèle verrait au test des grilles déjà apprises : c'est une **fuite de données**. Toutes
les photos d'une grille vont donc dans le même ensemble, et `build_grid_dataset.py` vérifie à la fin
qu'aucune grille n'est présente dans deux ensembles.

### Dataset du détecteur : `scripts/build_grid_dataset.py`

| Source | Images | Annotation |
|---|---|---|
| wichtounet | 202 | 4 coins annotés par l'auteur |
| rg1990 | 24 | coins calculés par `dewarp_grid`, vérifiés à l'œil |

La bbox est le rectangle englobant les 4 coins, au format YOLO `classe cx cy w h` normalisé. Le
découpage donne 156 images en train, 29 en val et 41 en test.

### Entraînement du détecteur : `scripts/train_grid.py`

- **Modèle de départ :** `yolo11n.pt`, pré-entraîné (**transfert d'apprentissage**).
- **Réglages :** jusqu'à 60 époques, images 640 px, batch 16, arrêt anticipé après 20 époques sans
  progrès (`patience=20`).
- **Augmentations :** rotation ±10°, légère perspective, pas de retournement vertical.
- **Évaluation :** `evaluate_corners` compare OpenCV seul, YOLO + OpenCV et YOLO seul, car ce qui
  compte au final est la précision des coins. Une image est réussie si ses 4 coins sont à moins de
  3 % du côté de la grille (`corner_error`).

### Dataset du classifieur : `scripts/build_cells_dataset.py`

Il y a très peu de vraies cases, donc on en **fabrique** à partir de 882 polices de chiffres.
`render_digit_cell` et `render_empty_cell` les font ressembler à des cases photographiées :

| Effet | Fonction | Ce que ça imite |
|---|---|---|
| fond gris + dégradé | `_background` | papier, éclairage inégal |
| taille, décalage, rotation ±6° | `_place` | redressement imparfait |
| traits sur les bords | `_grid_lines` | rognage imparfait |
| trait oblique ou courbe | `_oblique_line` | bord mal redressé |
| chiffre très pâle | `_place` | verso visible par transparence |
| basse résolution, flou, bruit, JPEG | `_degrade` | photo lointaine, capteur, compression |

Ce procédé réduit l'**écart de domaine** entre images synthétiques et réelles. Le découpage
train/val se fait **par police** : la validation mesure ainsi la capacité à généraliser, pas la
mémorisation d'une police. Les vraies cases sont ajoutées aussi, après un pré-étiquetage assisté
par le modèle (`scripts/prelabel_cells.py`) et contrôlé par les règles du sudoku.

### Entraînement du classifieur : `scripts/train_cells.py`

- **Modèle de départ :** `yolo11n-cls.pt`, 15 époques, batch 128, images 64 px, sur CPU.
- **Augmentations désactivées, car elles changeraient la classe :**
  - `fliplr=0` : un 2 retourné n'est plus un 2 ;
  - `erasing=0` : effacer un chiffre le transformerait en case « vide » avec une mauvaise étiquette.
- **Évaluation :** `evaluate_real_cells` mesure le résultat sur de vraies cases, et `plot_confusion`
  trace la matrice de confusion.

| Run | Changement | Cases réelles | Grilles parfaites |
|---|---|---|---|
| `cells_v1_clean` | polices + synthétique + vides réels | 99,66 % | 8 / 11 |
| `cells_v2` (retenu) | + vraies cases chiffres + traits obliques | 99,78 % | 10 / 11 |

**Pourquoi viser si haut ?** Une grille a 81 cases : à 99,5 % par case, seules 0,995⁸¹ ≈ 67 % des
grilles sont lues sans erreur. C'est pour cela que l'étape 8 existe.

---

## Évaluation et MLflow

### Évaluation de bout en bout : `scripts/evaluate.py`

La métrique principale est `solved_rate`, la part des photos dont la solution est exacte. On ajoute
les briques une à une pour mesurer ce que chacune apporte (étude d'ablation) :

| Configuration | Grille trouvée | Cases justes | Sudokus résolus | Solutions fausses |
|---|---|---|---|---|
| OpenCV seul | 97,6 % | 94,9 % | 85,4 % | 0 % |
| + YOLO | 100 % | 96,9 % | 85,4 % | 0 % |
| + orientation | 100 % | 99,4 % | 92,7 % | 0 % |
| **+ réparation (complet)** | **100 %** | 99,4 % | **97,6 % (40/41)** | **0 %** |

L'unique échec, `w_image17`, est une grille coupée par le bord de la photo. Le pipeline répond
« insoluble » plutôt que de donner une réponse fausse.

### MLflow

| Fonction de MLflow | Utilisation |
|---|---|
| **Tracking** | chaque entraînement ou évaluation est un run avec ses paramètres, métriques et fichiers (expériences `grid_detection`, `digit_classification`, `pipeline_eval`) |
| **Model Registry** | `scripts/register_models.py` enregistre `sudoku-grid-detector` et `sudoku-digit-classifier` avec l'alias `champion` |

L'application charge `models:/<nom>@champion` via `SudokuPipeline.from_registry` et
`registry_weights`. Pour déployer un meilleur modèle, il suffit de déplacer l'alias, sans toucher au
code.

---

## Application Streamlit

**Fichier :** `app/streamlit_app.py`. Elle montre chaque étape : grille trouvée (`draw_detection`),
grille redressée, tableau modifiable (`st.data_editor`), puis solution (`render_solution`).

- `@st.cache_resource` (`load_pipeline`) charge les modèles une seule fois.
- `st.session_state` garde le résultat, avec pour clé l'empreinte MD5 de la photo et les réglages.
  Corriger une case relance seulement le solveur, pas les modèles.
- Les cases dont la confiance est inférieure à 0,8 sont signalées.

---

## Limites à connaître

Ces points sont utiles à avoir en tête si on présente les résultats.

1. **Le score de 97,6 % est probablement optimiste.** Les modèles n'ont jamais vu les photos de test
   à l'entraînement, mais ces photos ont guidé la conception : les erreurs du test ont orienté
   `cells_v2`, l'orientation, la réparation et le choix entre les 4 essais de `refine_corners`.
   Seules de nouvelles photos donneraient une mesure vraiment indépendante.
2. **Le test est petit.** 41 photos et 27 grilles distinctes : chaque photo pèse 2,4 points.
3. **« 0 solution fausse » ne vaut que pour ce test.** La réparation en a déjà produit une sur une
   photo personnelle avant la correction des coins, et une telle erreur reste possible en théorie.
4. **Les coins de vérité des 11 photos rg1990 du test viennent de notre propre `dewarp_grid`**, avec
   une vérification à l'œil. La mesure de coins sur ces photos n'est donc pas totalement indépendante
   de la méthode évaluée. Ce biais ne touche pas `solved_rate`, qui repose sur les chiffres des grilles.
5. **Cas non gérés :** grille coupée par le bord de la photo et page très courbée (la photo 17 est
   exclue des jeux de données).
6. **Code non utilisé en production :** `warp_grid` sert seulement aux tests, et
   `DigitClassifier.predict_grid` n'est appelée nulle part.

---

## Glossaire

| Terme | Définition |
|---|---|
| **Apprentissage par transfert** | partir de poids déjà entraînés sur une autre tâche plutôt que de zéro |
| **Augmentation** | variations aléatoires des images d'entraînement pour mieux généraliser |
| **Backtracking** | essayer une valeur, continuer, revenir en arrière en cas d'impasse |
| **Bbox** | rectangle droit qui entoure un objet |
| **CNN** | réseau de neurones à convolutions, adapté aux images |
| **Dilatation** | opération morphologique qui épaissit les zones blanches |
| **Écart de domaine** | différence entre les images d'entraînement et les images réelles |
| **Époque** | un passage complet sur les données d'entraînement |
| **Fuite de données** | information du test présente à l'entraînement, qui gonfle le score |
| **Homographie** | transformation entre deux plans, fixée par 4 paires de points |
| **Interpolation bilinéaire** | calcul d'une valeur à partir des 4 points voisins |
| **IoU** | aire d'intersection divisée par aire d'union de deux boîtes |
| **mAP50 / mAP50-95** | précision moyenne de détection, avec un IoU minimal de 0,5 ou en moyenne de 0,5 à 0,95 |
| **MRV** | choisir d'abord la variable qui a le moins de valeurs possibles |
| **NMS** | suppression des détections qui se chevauchent, en gardant la plus sûre |
| **Ouverture morphologique** | érosion puis dilatation : retire les éléments plus petits que l'élément structurant |
| **Point fixe** | répéter un calcul jusqu'à ce que le résultat ne bouge plus |
| **Précision / rappel** | part des détections correctes / part des objets réels retrouvés |
| **Seuillage adaptatif** | seuil calculé pour chaque pixel selon son voisinage |
| **Softmax** | transforme des scores en probabilités dont la somme vaut 1 |
