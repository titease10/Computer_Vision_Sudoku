# Étape 4 — Lire les cases : classifieur YOLO-cls

Fichiers : `scripts/build_cells_dataset.py`, `scripts/train_cells.py`,
`src/sudoku_cv/digit_classifier.py`, `src/sudoku_cv/pipeline.py`

## Formulation du problème

Chaque case 64×64 est classée parmi **10 classes** : `0` = case vide, `1`…`9` = chiffres.
On traite « vide » comme une classe à part entière plutôt qu'avec une règle de seuil : le
modèle apprend lui-même à ignorer les chiffres fantômes du verso, les restes de traits, les
taches.

**Modèle :** `yolo11n-cls` (Ultralytics), pré-entraîné sur ImageNet, ~1,5 M de paramètres.
Le plus petit de la famille : il s'entraîne sur CPU en ~20 min et classe 81 cases en une
fraction de seconde.

## Construction du dataset — `build_cells_dataset.py`

| Classe | Source | Train | Val |
|---|---|---|---|
| 1-9 | 882 polices Chars74K (repo rg1990, sans les 134 exclues), **2 rendus** par image | 1 500 / classe | 264 / classe |
| 0 (vide) | cases vides synthétiques | 1 500 | 264 |
| 0 (vide) | **vraies cases vides** des photos d'entraînement | 1 019 | — |
| **test** | **405 vraies cases** des photos 1, 4, 8, 19, 24, étiquetées à la main | | |

### Rendre une police « comme une case photographiée »

Les images Chars74K sont des chiffres noirs parfaits sur fond blanc, très différents d'une
case photographiée. Sans adaptation, le modèle apprendrait sur des images qui ne
ressemblent pas à celles qu'il verra (**écart de domaine**). Chaque rendu tire au hasard :

| Effet | Imite |
|---|---|
| fond gris 120-245 + dégradé linéaire | papier, éclairage inégal, ombre |
| encre 0-110 (toujours ≥ 50 niveaux sous le fond) | encre noire, imprimée grise, encre colorée |
| hauteur du chiffre 50-85 % de la case, décalage ±8 %, rotation ±6° | redressement imparfait |
| traits sur les bords (30 % par côté) | rognage imparfait de la grille |
| chiffre fantôme à 5-28 niveaux du fond (40 % des vides, 15 % des chiffres) | verso visible par transparence |
| réduction à 14-32 px puis agrandissement (40 %) | photo lointaine, écran |
| flou gaussien, bruit, compression JPEG | mise au point, capteur, compression |

Montage (haut : synthétique, classes 0 à 9 ; bas : vraies cases de test ; dernière ligne :
vraies cases vides) :

![synthétique vs réel](img/cells_montage_synth_vs_reel.png)

### Choix des augmentations d'Ultralytics

Ultralytics ajoute ses propres augmentations. Deux ont été **désactivées** car elles changent
la classe :

- `fliplr` (miroir horizontal) : un 2 ou un 3 retourné n'est plus un chiffre valide ;
- `erasing` (effacement aléatoire d'un rectangle) : peut effacer le chiffre et produire une
  image « vide » étiquetée « chiffre ».

## Suivi avec MLflow

- Stockage : `sqlite:///mlflow.db` (+ artefacts dans `mlruns/`), expérience
  `digit_classification`.
- Le script ouvre lui-même le run ; le callback MLflow d'Ultralytics s'y greffe et y écrit
  les hyperparamètres, la loss et la précision à chaque époque, les poids et les graphiques.
- On y ajoute ensuite **l'évaluation sur les vraies cases** (la mesure qui compte) : métriques
  `real_*`, matrice de confusion `real_confusion.png`, liste des erreurs `real_errors.txt`.
- Visualiser : `.venv/Scripts/mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Résultats — run `cells_v1`

Entraînement : 15 époques, batch 128, 64×64, CPU (≈ 75 s par époque, 19 min au total).

| Mesure | Valeur |
|---|---|
| Précision validation (synthétique, polices inconnues) | 99,28 % |
| **Précision sur les vraies cases (405)** | **99,51 %** (2 erreurs) |
| … sur les chiffres (147) | 99,32 % |
| … sur les vides (258) | 99,61 % |
| **Grilles lues sans aucune erreur** | **3 / 5** |

Matrice de confusion sur les vraies cases :

![confusion](img/cells_v1_confusion_reel.png)

### Analyse des 2 erreurs

![erreurs](img/cells_v1_erreurs.png)

*(gauche : photo 4, case (0,0) ; milieu : photo 24, case (8,2) ; droite : un 8 normal, pour
comparaison)*

1. **Photo 4, case (0,0) — vide lue comme 7.** La case contient un **trait diagonal** : le
   bord de la grille, courbé, mal corrigé dans le coin par le redressement (étape 3). Une
   barre oblique ressemble au jambage d'un 7. Le dataset ne contient que des traits
   horizontaux ou verticaux sur les bords.
   *Cause : redressement imparfait dans un coin + aucun exemple de trait oblique.*
2. **Photo 24, case (8,2) — 6 lu comme 8.** Chiffre **rouge sur écran, très flou**, la boucle
   du haut du 6 est presque fermée. Le modèle n'a vu aucun vrai chiffre d'écran à
   l'entraînement (seulement des vides réels).
   *Cause : écart de domaine (flou d'écran) + aucune vraie case chiffre en train.*

### Pourquoi 99,5 % ne suffit pas

Une grille a 81 cases : avec 99,5 % par case, la probabilité de lire une grille sans erreur
est 0,995⁸¹ ≈ **67 %**, ce qui correspond bien aux 3 grilles parfaites sur 5 observées.
Pour atteindre 90 % de grilles parfaites, il faut ~99,9 % par case. D'où deux leviers :
améliorer le modèle, **et** corriger les erreurs avec les règles du sudoku (étape pipeline).

## Pistes pour la v2

| Piste | Erreur visée |
|---|---|
| Cases vides synthétiques avec **traits obliques / courbes** | erreur 1 |
| **Pré-étiqueter les 520 vraies cases remplies** avec v1, corriger à la main, les ajouter au train (dont les chiffres rouges des photos 23 et 25) | erreur 2, écart de domaine |
| Plus de flou et de basse résolution dans les rendus | erreur 2 |
| Correction par le solveur : si la grille lue est invalide ou insoluble, essayer le 2ᵉ choix des cases les moins sûres | les deux |
| Améliorer le redressement des coins (étape 3) | erreur 1 |

**Limite de l'évaluation :** 405 cases de 5 photos, c'est peu. Une erreur de plus ou de
moins change la précision de 0,25 point. Il faudra élargir le jeu de test avec de nouvelles
photos.

---

## Correction de l'évaluation : fuite de données (voir `02_donnees.md`)

Le test de `cells_v1` (5 photos) était biaisé : 6 photos d'entraînement montraient les
mêmes grilles que le test. Nouveau découpage **par grille** : test = **11 photos, 891 cases**.
Pour comparer proprement, on réentraîne la recette v1 sur le découpage propre
(`cells_v1_clean`), puis on ne change **qu'une chose à la fois** pour la v2.

## Résultats — v1_clean vs v2 (même test : 891 vraies cases, 11 photos)

| Run | Recette | Val (synth., meilleure époque) | **Cases réelles** | Chiffres | Vides | Erreurs | **Grilles parfaites** |
|---|---|---|---|---|---|---|---|
| `cells_v1` *(fuite, pour info)* | polices + synthétique + vides réels | 99,28 % | 99,55 % | 99,68 % | 99,48 % | 4 | 8 / 11 |
| `cells_v1_clean` | idem, découpage par grille | 99,24 % | 99,66 % | 99,68 % | 99,65 % | 3 | 8 / 11 |
| **`cells_v2`** | + **vraies cases chiffres** (×3) + **traits obliques** dans les vides | 99,09 % | **99,78 %** | **100 %** | 99,65 % | **2** | **10 / 11** |

Lecture :

- **Les deux erreurs visées sont corrigées** : le trait oblique de la photo 4 (plus lu comme
  un 7) et le 6 rouge flou de la photo 24 (plus lu comme un 8). **100 % des chiffres** réels
  sont bien lus.
- La précision de **validation baisse légèrement** (99,24 → 99,09 %) alors que le **test réel
  progresse** : la validation est synthétique ; ajouter des vraies cases rapproche le modèle
  des vraies photos, pas des images synthétiques. C'est la preuve qu'il faut juger sur des
  **données réelles** et pas sur la validation synthétique.
- Avec 1 à 2 erreurs d'écart, les différences entre runs sont **petites en valeur absolue** :
  sur 891 cases, une erreur = 0,11 point. La conclusion « v2 > v1_clean » est cohérente avec
  l'analyse des erreurs (chaque erreur corrigée l'est pour la raison visée), mais un test plus
  grand serait nécessaire pour la chiffrer finement.

![confusion v2](img/cells_v2_confusion_reel.png)

## Les 2 erreurs restantes de v2 : un problème de redressement, pas de lecture

![erreurs v2](img/cells_v2_erreurs.png)

*(photo 18, cases (0,0) et (1,0), lues « 1 » ; à droite un vrai 1)*

Ces deux cases contiennent le **bord extérieur de la grille** : sur la photo 18 (page
courbée), le coin haut-gauche est mal aligné et la colonne de gauche est décalée. Un trait
vertical au milieu d'une case ressemble effectivement à un 1 : le classifieur n'a pas « tort »
au vu de l'image. Le correctif est en amont (étape 3 : redressement / coins fournis par YOLO)
ou en aval : deux « 1 » dans la même colonne et le même bloc créent un **doublon**, que la
correction par les règles du sudoku (étape pipeline) pourra détecter et réparer.

**Modèle retenu pour la suite : `cells_v2`** (`runs/cells/cells_v2/weights/best.pt`).
