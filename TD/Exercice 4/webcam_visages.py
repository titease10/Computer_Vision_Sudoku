"""Exercice 4.2 : détection de visages en direct sur la webcam (Haar cascade).

Usage (depuis le dossier TD/) :
    python "Exercice 4/webcam_visages.py"              # webcam par défaut
    python "Exercice 4/webcam_visages.py" --camera 1   # autre webcam
    python "Exercice 4/webcam_visages.py" --yeux       # + vérification des yeux (moins de faux positifs)
Touche q ou Échap pour quitter.
"""
import argparse
import time

import cv2

cascade_visage = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cascade_yeux = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def detecter_visages(img_bgr, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), verifier_yeux=False):
    """Même fonction que dans le notebook : renvoie la liste des visages (x, y, w, h)."""
    gris = cv2.equalizeHist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    visages = cascade_visage.detectMultiScale(gris, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    if verifier_yeux:
        visages = [(x, y, w, h) for x, y, w, h in visages
                   if len(cascade_yeux.detectMultiScale(gris[y:y + 2 * h // 3, x:x + w], 1.1, 5)) > 0]
    return visages


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="indice de la webcam (0 par défaut)")
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--yeux", action="store_true", help="ne garder que les visages où un œil est détecté")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Webcam {args.camera} introuvable")
    t_prec = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # effet miroir
            visages = detecter_visages(frame, minNeighbors=args.min_neighbors, verifier_yeux=args.yeux)
            for x, y, w, h in visages:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            t = time.time()
            fps = 1 / max(t - t_prec, 1e-6)
            t_prec = t
            cv2.putText(frame, f"{len(visages)} visage(s) - {fps:.0f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Visages (q pour quitter)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
