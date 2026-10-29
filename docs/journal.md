# Journal de bord

Chaque entrée : ce qu'on a essayé, ce qu'on a observé (avec les mesures), ce qu'on a décidé.
Les pistes abandonnées sont gardées : elles expliquent les choix.

---

## 2026-10-29 — Cadrage

**Question :** comment lire une grille de sudoku sur une photo de papier ?

Pistes envisagées :

| Piste | Avantages | Inconvénients | Décision |
|---|---|---|---|
| Tout OpenCV (contours + OCR classique) | pas de données, rapide | fragile : fond chargé, ombres, courbure | gardé pour la géométrie fine |
| YOLO detect qui trouve **les 81 chiffres** directement | un seul modèle | 81 petits objets, beaucoup d'annotations, lourd sur CPU | écarté |
| **YOLO detect (grille) + OpenCV (coins) + YOLO-cls (cases)** | chaque étape simple et mesurable ; classer 64×64 est rapide sur CPU | plusieurs étapes à enchaîner | **retenu** |

Contraintes : pas de GPU (CPU / Colab) → petits modèles (`yolo11n`, `yolo11n-cls`).
Suivi des expériences avec **MLflow**. Interface finale **Streamlit**.

## Étape 1 — Solveur

Backtracking + heuristique MRV. 6 tests, dont une grille très difficile (AI Escargot).
Aucun problème particulier. Le solveur sert aussi de contrôle de la lecture (doublons).

## Données — repo rg1990

Trouvé : 25 photos, ~9 000 chiffres imprimés (Chars74K), 30 000 grilles.
**Problèmes :** pas d'annotations pour YOLO, pas de MNIST, 25 photos seulement.
**Observation importante :** sur la photo 1, les chiffres du **verso** se voient par
transparence → risque que le modèle les lise comme de vrais chiffres. Il faudra des
exemples de « cases vides avec fantôme » dans la classe vide.

## Étape 3 — Redressement

1. **Coins par contours + homographie** : 25/25 grilles trouvées. Mais l'heuristique de
   comptage donne **0** case remplie sur 23, 24 et 25, et **47** sur 17.
2. **Diagnostic 23-25** : chiffres rouges sur écran ; en niveaux de gris ils sont trop peu
   contrastés pour la règle « 0,6 × fond ».
   → Passage au **canal le plus sombre** + **écart absolu de 30 niveaux**. Seuil choisi en
   comparant 30/40/50/60 sur les 25 photos (tableau dans `03_redressement.md`). Résultat :
   27/36/26, égal au comptage manuel.
3. **Diagnostic 4, 17, 19** : pages de livre **courbées**. Une homographie ne peut pas
   aplatir une surface courbe ; des traits de grille tombent dans les cases (35 au lieu de 28).
   → Nouvelle méthode **`dewarp_grid`** : suivi des 20 traits, ajustement de paraboles,
   100 intersections, remappage case par case. Photo 19 : 35 → **28** (exact).
   Photo 4 : 35 → 32. Photo 17 : 47 → 34, toujours imparfait (coin bas-gauche faux).
4. Bug rencontré : `cv2.remap` refusait la carte (float64 au lieu de float32 après les
   calculs numpy) → conversion explicite.

**Reste à faire :** photo 17 (courbure forte) ; YOLO devra fournir une bbox plus fiable.

## Étape 6 — Classifieur de cases (en cours)

- Vérité terrain de 5 photos de test saisie à la main ; **contrôle par le solveur** : les 5
  grilles sont valides et solubles (une faute de saisie aurait probablement créé un doublon).
- `font_exclude` du repo : ce sont des **numéros de police** (134) à exclure pour les 9
  chiffres, pas des fichiers isolés.
- Dataset : 882 polices × 9 chiffres rendues comme des cases photographiées (fond, flou,
  bruit, traits, basse résolution, JPEG) ; classe vide = synthétique (40 % avec chiffre
  fantôme) + 1 019 vraies cases vides.
- **Choix d'augmentations Ultralytics désactivées :** `fliplr` (un 2 retourné n'est plus un 2)
  et `erasing` (effacer une partie d'un chiffre peut le rendre « vide »).
- Run MLflow `cells_v1` (15 époques, 19 min CPU) : val 99,28 % ; **vraies cases 99,51 %
  (2 erreurs / 405), 3 grilles parfaites sur 5**.
- Erreur 1 (photo 4, coin) : trait **oblique** laissé par le redressement → lu comme un 7.
- Erreur 2 (photo 24) : 6 rouge très flou → lu comme un 8.
- Constat : 0,995⁸¹ ≈ 67 % de grilles parfaites → il faut ~99,9 % par case **et** une
  correction par les règles du sudoku. Pistes v2 dans `04_classifieur_chiffres.md`.

## Étape 6b — Pré-étiquetage des vraies cases et découverte d'une fuite de données

**Objectif :** ajouter de vraies cases chiffres à l'entraînement (le modèle v1 n'a vu que des
chiffres synthétiques).

**Méthode — étiquetage semi-automatique** (`scripts/prelabel_cells.py`) : le modèle v1
propose une étiquette pour les 520 cases remplies des photos d'entraînement ; on vérifie avec
deux outils au lieu de tout saisir :
1. planches d'images par classe prédite, triées par confiance (`data/debug/prelabel/`) ;
2. **contrôle par les règles du sudoku** : la grille reconstituée de chaque photo doit être
   valide, soluble, et avoir une **solution unique** (ajout de `count_solutions` au solveur).

**Résultat :** 17 photos sur 19 passent directement. Les 2 autres :
- photo 23 : un 5 rouge flou lu comme 6 → grille **insoluble** → corrigé à la main ;
- photo 18 : bord de page vertical lu comme des « 1 » → **doublon** ;
après correction, chaque grille d'entraînement a exactement 1 solution.

**Découverte : fuite de données.** En regardant la photo 18, on reconnaît **la même grille
que la photo de test 19**. Comparaison systématique des grilles lues :

| Grille de test | Autres photos de la même grille |
|---|---|
| 1 | 22 |
| 4 | 5 |
| 8 | 6, 7, 9 |
| 19 | 18 |
| 24 | — |

Le jeu rg1990 contient plusieurs photos de la même page (angles, éclairages différents). Le
découpage train/test **par photo** était donc faux : le modèle v1 avait appris les cases
vides (avec leurs chiffres fantômes) de 5 photos dont la grille est en test.

**Décision :** découpage **par grille** (toutes les photos d'une grille dans le même
ensemble). Le test passe de 5 photos / 405 cases à **11 photos / 891 cases**, avec une vérité
terrain gratuite (même grille). Les 13 photos restantes (2, 3, 10-16, 20, 21, 23, 25) sont
en train, entièrement étiquetées.

**Comparaison honnête :** le score de v1 n'est plus comparable. On entraîne
`cells_v1_clean` (même recette, découpage propre) comme **référence**, puis `cells_v2`
(+ vraies cases chiffres + traits obliques) : une seule chose change entre les deux runs.

Pour information, v1 (avec fuite) évaluée sur le nouveau test : 99,55 %, 4 erreurs, 8 grilles
parfaites sur 11 (dont 2 nouvelles erreurs : le bord de page de la photo 18 lu « 1 »).

## Recherche d'un dataset annoté pour le détecteur de grille (étape 4)

**Besoin :** des photos où la grille est annotée, pour entraîner YOLO detect sans annoter à
la main (~1-2 min par photo).

| Source | Contenu | Verdict |
|---|---|---|
| **wichtounet/sudoku_dataset** (GitHub, CC-BY-4.0) | 203 photos de journaux (smartphones, 640×480 à 1600×1200), **4 coins de la grille** pour 202 photos (`outlines_sorted.csv`), **chiffres de chaque grille** (`.dat`) | **retenu** |
| Roboflow Universe « SudokuDetection » (mlworkspace) | ~100-185 images | page inaccessible (403) sans compte ; à revoir si besoin |
| Roboflow « 2 » (brecht) | classes sudoku + 4 coins | contenu non vérifié |
| Lexski (Hugging Face) | photos + captures, chiffres par case | pas de coins |
| AICrowd sudoku | 10 001 images synthétiques | pas des photos |

**Vérifications faites sur wichtounet :**
- coins tracés sur 6 photos : ils tombent exactement sur les coins de la grille, ordre
  haut-gauche, haut-droite, bas-droite, bas-gauche → convertibles directement en bbox YOLO ;
- **même piège que rg1990 :** 203 photos mais seulement **77 grilles distinctes** (un groupe
  de 18 photos de la même grille). Le découpage officiel train/test de l'auteur met
  **15 grilles à la fois en train et en test** → on refera un découpage par grille.

**Bonus :** les chiffres de chaque grille donnent une vérité terrain pour ~200 grilles, soit
~16 000 vraies cases étiquetées : utile pour le classifieur et pour l'évaluation de bout en
bout.

**Limite :** beaucoup de photos sont cadrées serré sur la grille (peu de fond, peu de
« scène ») ; pour que YOLO apprenne à trouver une grille dans une scène encombrée, il faudra
garder aussi des photos plus larges (rg1990, les nôtres).

## Étape 6c — Résultats v1_clean vs v2

- `cells_v1_clean` (recette v1, découpage propre) : 99,66 %, 3 erreurs, 8/11 grilles parfaites.
- `cells_v2` (+ vraies cases chiffres ×3 + traits obliques) : **99,78 %, 2 erreurs, 10/11
  grilles parfaites, 100 % des chiffres**. Les 2 erreurs visées (trait oblique → 7, 6 rouge
  flou → 8) sont corrigées.
- Observation : la val synthétique baisse un peu pendant que le test réel monte → on juge
  sur le réel.
- Les 2 erreurs restantes (photo 18) viennent du **redressement** (bord de grille dans la
  case, lu comme un 1), pas du classifieur. Elles créent un doublon → réparables par la
  correction sudoku.
- Temps d'entraînement : ~33 s/époque pour v2 contre ~80 s pour v1_clean : les époques de
  v1_clean ont été ralenties par d'autres calculs lancés en parallèle (construction du dataset
  de détection). À retenir : ne pas mesurer des temps quand le CPU est partagé.

## Étape 5 (détection) — référence OpenCV et lancement de YOLO

- Référence **OpenCV seul** sur le test détection (41 images) : 4 coins à moins de 3 % du
  côté de la grille sur **90,2 %** des images (erreur médiane 1,2 %). Run MLflow
  `opencv_baseline` (expérience `grid_detection`).
- Les 4 échecs : grille **fusionnée avec la grille des solutions** voisine (w_image1077),
  grille **coupée par le bord de la photo** (w_image17, aucun contour fermé), page **à
  l'envers** non trouvée (w_image1083), page courbée (rg_18).
- Lancement de `grid_v1` (yolo11n, 60 époques max, arrêt anticipé après 20 époques sans
  progrès, rotation ±10°, perspective, pas de retournement vertical).

### Problème : entraînement `grid_v1` interrompu (époque 34/60)

- L'entraînement s'est arrêté vers 12:55 à l'époque 34 (session fermée), sans erreur dans le
  log. Le run MLflow est resté à l'état `RUNNING`.
- À l'époque 33 : mAP50 val = 0,995, mAP50-95 = 0,907 (en hausse régulière, donc
  l'entraînement n'était pas terminé).
- Solution : option `--resume` ajoutée à `scripts/train_grid.py`. Elle recharge
  `weights/last.pt` (poids, optimiseur, numéro d'époque) avec `train(resume=True)` et
  rattache la suite au **même run MLflow**, ce qui garde des courbes continues.
- Leçon : un long entraînement CPU doit pouvoir être repris. Ultralytics sauvegarde
  `last.pt` à chaque époque, ce qui le permet sans rien perdre.
- Effet de bord découvert à la reprise : l'intégration MLflow d'Ultralytics refuse d'écrire
  dans un run dont les paramètres ont changé (`model` : `yolo11n.pt` devient `last.pt`) et
  affiche « Not tracking this run ». Les époques 34 à 60 n'étaient donc pas dans MLflow.
  Correction : `backfill_epoch_metrics()` recopie les époques manquantes depuis `results.csv`
  (mêmes noms, step = époque - 1), et les poids et courbes sont ajoutés au run à la main.
  Les 60 époques sont maintenant visibles dans MLflow.

## Étape 5 (détection) — résultats `grid_v1`

- Test : mAP50 = 0,995, mAP50-95 = 0,921, rappel 100 %.
- Coins : YOLO + OpenCV réussit **95,1 %** (contre 90,2 % pour OpenCV seul), bbox seule 9,8 %.
- Échecs restants : page courbée (rg_18), grille coupée par le bord (w_image17).

## Étape 7-8 — Pipeline complet et évaluation de bout en bout

- Première mesure, lecture brute : 85,4 % des sudokus résolus (35/41).
- **Problème 1 : grilles tournées.** 3 photos de journal tournées de 90° ou à l'envers :
  environ 30 cases fausses chacune. Les coins sont ordonnés selon la photo, pas selon la
  grille. Solution : classer dans les 4 orientations et garder la plus confiante (confiance
  calculée sur les cases chiffres seulement). On passe à 92,7 %.
- **Problème 2 : une case fausse suffit à tout casser** (rg_18 : bord de grille lu « 1 » ;
  w_image46 : 1 lu « 4 »). Solution : réparation par les règles (2ᵉ choix des cases les moins
  sûres jusqu'à obtenir une solution unique). On passe à **97,6 % (40/41), 0 solution fausse**.
- Piste écartée : réparer en essayant toutes les cases. Trop de combinaisons, et plus de
  risque de tomber par hasard sur une mauvaise grille valide. Les 8 cases les moins sûres
  suffisent : les erreurs du classifieur ont presque toujours une confiance basse.
- Échec restant : w_image17, grille coupée par le bord de la photo. Le pipeline répond
  « insoluble » au lieu d'une solution fausse.

## Test sur une photo personnelle : double page de magazine (échec puis correction)

Photo : capture d'écran 585×398 d'un magazine ouvert, **deux grilles** (à gauche une grille
vierge, à droite une grille remplie au crayon avec des notes dans les cases).

**Diagnostic, étape par étape :**
1. Détection YOLO : correcte, les **deux** grilles sont trouvées (confiance 0,93 chacune).
2. Coins : **faux**. Dans la bbox de gauche, le seuillage puis la dilatation **soudent la
   grille au bord de la page** (ombre de la reliure à droite, coin de la page en bas à
   gauche). Le contour a alors 6 sommets, il est rejeté, et on retombe sur les coins de la
   bbox, trop imprécis.
3. Conséquence : la grille redressée est décalée d'environ une colonne, tous les chiffres
   sont lus dans la mauvaise case, et la grille est déclarée insoluble.

**Corrections :**
- `_find_quad` : si aucun quadrilatère n'est trouvé avec la dilatation, 2ᵉ essai **sans
  dilatation**. Le contour de la grille redevient un quadrilatère propre. La dilatation reste
  le 1er essai, donc les photos qui marchaient déjà ne changent pas.
- Pipeline : `GridDetector.detect_all` renvoie **toutes** les grilles détectées. Le pipeline
  les essaie de la plus sûre à la moins sûre et garde la **première résolue**. Avant, seule
  la plus sûre était essayée.

**Résultat :** la grille de gauche est lue sans erreur (81/81 cases vérifiées à l'œil) et
résolue. Pas de régression sur le test : 97,6 % des sudokus résolus, coins 95,1 %, chiffres
identiques au centième près.

**Pistes écartées :**
- Approcher l'enveloppe convexe du contour soudé par 4 sommets : on obtient bien un
  quadrilatère, mais il englobe le coin de la page. Il est donc faux, et rien ne permettait
  de s'en rendre compte.
- Utiliser directement la bbox YOLO : c'est exactement le repli qui a échoué (rectangle droit
  alors que la grille est en perspective).

**Leçon :** ce cas n'existait pas dans le jeu de test (une seule grille par photo, photos
plus grandes). Les photos personnelles révèlent des situations que le jeu public ne couvre
pas.
