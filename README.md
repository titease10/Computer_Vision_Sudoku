# Sudoku papier

On prend en photo une grille de sudoku de journal ou de livre, et les chiffres manquants
s'écrivent directement sur la photo. Projet de computer vision de Vincent, Titien et Aymeric.

![L'application sur une page de livre courbée](docs/img/app_resolu.png)

Sur 41 photos jamais vues pendant l'entraînement, 40 sudokus sont résolus et aucune solution
n'est fausse. La photo ratée montre une grille coupée par le bord de l'image : l'application
répond qu'elle ne peut pas la résoudre.

## Comment ça marche

La photo passe par une suite d'étapes courtes, et chacune se teste à part :

1. **Trouver la grille.** Un YOLO (`yolo11n`) entraîné sur 156 photos encadre la grille, même
   s'il y a une page voisine ou un tableau à côté.
2. **Trouver ses coins.** Dans ce cadre, OpenCV cherche le contour de la grille. Le cadre seul
   ne suffit pas : c'est un rectangle droit, alors qu'une grille photographiée de biais est un
   quadrilatère quelconque.
3. **La mettre à plat.** On suit les 20 traits de la grille et on redresse case par case, ce
   qui marche aussi sur une page de livre courbée.
4. **Lire les 81 cases.** Un second YOLO (`yolo11n-cls`) classe chaque case : vide ou 1 à 9.
   La grille est lue dans les 4 orientations et on garde la plus crédible, parce que beaucoup
   de photos sont prises de côté.
5. **Résoudre.** Une vraie grille a une seule solution. Si ce n'est pas le cas de la lecture,
   c'est qu'une case est mal lue : on essaie alors le 2ᵉ choix du classifieur sur les cases
   les moins sûres. Le solveur (backtracking) termine le travail.

On a réservé l'apprentissage aux deux étapes que des règles écrites à la main ne savent pas
faire : reconnaître une grille dans une scène et reconnaître un chiffre. La géométrie et la
résolution restent déterministes.

## Ce que chaque étape apporte

Mesuré sur les 41 photos de test, en ajoutant les briques une à une :

| Configuration | Cases justes | Sudokus résolus | Solutions fausses |
|---|---|---|---|
| OpenCV seul, lecture brute | 94,9 % | 85,4 % | 0 % |
| + détecteur YOLO | 96,9 % | 85,4 % | 0 % |
| + choix de l'orientation | 99,4 % | 92,7 % | 0 % |
| + réparation par les règles | 99,4 % | 97,6 % | 0 % |

Le test reste petit : 41 photos et 27 grilles distinctes, donc chaque photo pèse 2,4 points.
Le détail, y compris les pistes abandonnées, est dans [docs/](docs/README.md).

## Les difficultés qui ont le plus compté

- **Une fuite de données.** Les jeux publics contiennent plusieurs photos de la même page. Notre
  premier découpage par photo mettait les mêmes grilles en entraînement et en test. On a tout
  refait avec un découpage par grille ([docs/02_donnees.md](docs/02_donnees.md)).
- **Les pages courbées.** Une homographie ne peut pas aplatir une surface courbe, d'où le suivi
  des traits ([docs/03_redressement.md](docs/03_redressement.md)).
- **Une solution fausse inventée.** Sur une de nos photos, la réparation a produit une grille
  valide mais fausse. Il a fallu 4 essais pour corriger la géométrie sans faire baisser le score
  du test ([docs/journal.md](docs/journal.md)).

## Lancer le projet

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/streamlit run app/streamlit_app.py     # http://localhost:8501
```

Les poids (`runs/`), le registry (`mlflow.db`) et les données (`data/`) ne sont pas versionnés.
Il faut les récupérer ou réentraîner les modèles : les commandes sont dans
[docs/README.md](docs/README.md).

## Où en est le projet

Le pipeline, l'évaluation, le suivi MLflow et l'application Streamlit fonctionnent.
L'API FastAPI prévue dans la note de cadrage n'est pas encore écrite : pour l'instant,
l'application appelle directement le pipeline Python.
