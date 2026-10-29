# 7. Application Streamlit

## Ce qui a été fait

`app/streamlit_app.py` : on importe une photo (ou on la prend avec la caméra) et l'application
montre chaque étape du pipeline :

1. **Grille trouvée** : bbox YOLO en bleu, les 20 traits suivis par le redressement en orange
   (on voit tout de suite si la grille est mal localisée) ;
2. **Grille redressée** 450×450, avec la rotation appliquée si la grille était tournée ;
3. **Grille lue**, dans un **tableau modifiable** : les cases réparées et les cases peu sûres
   (confiance < 0,8) sont listées. Si une case est mal lue, l'utilisateur la corrige et la
   solution est recalculée tout de suite, sans relancer les modèles ;
4. **Solution** écrite sur la photo, ou un message clair si la grille est insoluble (doublon,
   aucune solution) ou ambiguë (plusieurs solutions).

Réglages dans la barre latérale :
- modèles du **Model Registry MLflow** (alias `champion`) ou fichiers locaux ;
- activer ou désactiver les 4 orientations et la réparation, pour montrer en démo l'apport de
  chaque brique.

## Choix techniques

- `st.cache_resource` : les modèles YOLO sont chargés une seule fois, pas à chaque clic.
- Le résultat du pipeline est gardé dans `st.session_state`, avec comme clé l'empreinte MD5
  de la photo et les réglages. Modifier une case du tableau relance seulement le solveur
  (quelques ms), pas la détection ni la lecture (environ 0,8 s).
- Correction manuelle plutôt qu'échec : sur une photo difficile (grille coupée, w_image17),
  l'application indique les cases douteuses et l'utilisateur peut finir la lecture à la main.

## Problèmes rencontrés

- `cv2.polylines` refusait les colonnes du tableau des intersections (`pts[:, k]`) : une
  colonne numpy n'est pas contiguë en mémoire, et OpenCV exige un tableau contigu.
  Correction : `np.ascontiguousarray`.
- `use_container_width` est obsolète dans Streamlit 1.66 : remplacé par `width="stretch"`.
- Tester une application Streamlit sans navigateur : `streamlit.testing.AppTest` ne sait pas
  simuler l'import d'un fichier. On a remplacé `st.file_uploader` par une fonction qui
  renvoie une photo de test, pour vérifier 3 cas : page courbée réparée (rg_18), grille
  coupée insoluble (w_image17), grille tournée de 90° résolue (w_image1041).

## Lancement

```bash
.venv/Scripts/streamlit run app/streamlit_app.py      # http://localhost:8501
```
