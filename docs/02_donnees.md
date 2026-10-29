# Étape 2 — Les données

## Sources

### Repo `rg1990/cv-sudoku-solver` (licence MIT)

Cloné dans `data/external/cv-sudoku-solver/` (non versionné). Contenu réellement trouvé :

| Dossier | Contenu | Utilisation |
|---|---|---|
| `data/sudoku_images/` | **25 photos** JPG 2736×3648 | test du redressement, cases réelles, vérité terrain |
| `data/digit_images/Sample002…010` | 9 × 1016 images 128×128 de chiffres imprimés (dataset Chars74K « Fnt ») ; `Sample002` = 1 … `Sample010` = 9 | chiffres 1-9 du classifieur |
| `data/font_exclude/` | 134 images → **134 numéros de police** écartés par l'auteur (polices fantaisie) | on les exclut aussi |
| `data/sudoku_puzzles/` | 3 fichiers texte, ~30 000 grilles de 81 caractères | tests du solveur, futures grilles synthétiques |

Écarts constatés par rapport à ce qu'on attendait :

- **Pas d'annotations** sur les photos : l'auteur trouve la grille par contours OpenCV, sans
  apprentissage. Pour entraîner YOLO detect il faudra annoter nous-mêmes.
- **MNIST (manuscrit) n'est pas dans le repo** : l'auteur le téléchargeait via Keras.
- **Seulement 25 photos** : suffisant pour développer et tester, trop peu pour entraîner un
  détecteur seul.

Le code Python du repo n'est **pas exécuté** : on n'utilise que ses données.

### Ce que montrent les 25 photos (difficultés réelles)

- pages de **livre courbées** (4, 16, 17, 19…) ;
- chiffres du **verso visibles par transparence** (chiffres « fantômes » gris clair, photo 1) ;
- 3 photos d'**écran** avec des chiffres **rouges** flous et basse résolution (23, 24, 25) ;
- doigt, fond encombré, page voisine visible.

## Vérité terrain

Fichier `labels/rg1990_grids.json` : pour chaque photo de test, les 81 cases (0 = vide),
lues à la main sur la grille redressée.

**Contrôle de la saisie :** chaque grille transcrite est passée dans `is_valid_grid` et
`solve` — les 5 sont valides et solubles. Une faute de transcription produirait très
probablement un doublon ou une grille insoluble, c'est donc un bon garde-fou.

## Découpage train / val / test

**Découpage par grille, pas par photo.** Le jeu rg1990 contient plusieurs photos de la même
page (grilles A à E ci-dessous). Toutes les photos d'une même grille sont dans le même
ensemble ; sinon le modèle verrait au test des cases qu'il a déjà apprises (voir
« Problème : fuite de données »).

| Grille | Photos |
|---|---|
| A | 1, 22 |
| B | 4, 5 |
| C | 6, 7, 8, 9 |
| D | 18, 19 |
| E | 11, 12 |

| Ensemble | Contenu | Rôle |
|---|---|---|
| **test** | photos 1, 22, 4, 5, 6, 7, 8, 9, 18, 19, 24 (**11 photos, 891 cases**, 5 grilles) | mesure finale, jamais vues à l'entraînement |
| train | photos 2, 3, 10-16, 20, 21, 23, 25 (13 photos, toutes les cases étiquetées) + synthétique | apprentissage |
| val | synthétique, **polices jamais vues** en train | suivi pendant l'entraînement |

Les grilles de test couvrent les cas difficiles (verso visible : A ; pages courbées : B, D ;
écran rouge : 24 ; cas « normal » : C).

La photo **17** est écartée partout : la grille y est trop déformée par la courbure (voir
étape 3), ses cases ne sont pas fiables.

### Problème : fuite de données (data leakage)

Si une même police de caractères apparaît en train et en val, la validation mesure la
mémorisation de la police, pas la généralisation. Le split train/val se fait donc **par
police** (132 polices sur 882 réservées à la val), pas par image.

De même, les photos de test ne fournissent **aucune** case à l'entraînement, même pas les
cases vides.

**Erreur commise puis corrigée :** le premier découpage était fait par photo (test = 1, 4, 8,
19, 24). En pré-étiquetant les photos d'entraînement, on a constaté que 6 d'entre elles
montraient les mêmes grilles que le test (même page, autre angle). Le modèle `cells_v1` a donc
été entraîné avec des cases vides de ces pages : son score de test est **optimiste**. Le
découpage a été refait par grille et une référence propre (`cells_v1_clean`) réentraînée.

## Étiquetage des photos d'entraînement

Semi-automatique (`scripts/prelabel_cells.py`) : le modèle `cells_v1` propose, on vérifie
(planches d'images + contrôle « la grille lue a exactement une solution »). 1 correction
manuelle (photo 23, case (2,0) : 5). Les étiquettes finales sont dans
`labels/rg1990_grids.json` (clés `train` et `test`).

## Problèmes ouverts

- Il faudra **plus de photos** (les nôtres + datasets publics) pour le détecteur YOLO et pour
  un test plus solide : 11 photos mais seulement **5 grilles différentes** en test, c'est
  peu pour conclure.
