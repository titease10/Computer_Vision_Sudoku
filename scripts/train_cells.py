"""Entraîne le classifieur de cases YOLO-cls et trace tout dans MLflow.

Le run MLflow est ouvert ici ; le callback MLflow d'Ultralytics s'y greffe (paramètres,
métriques par époque, poids). On y ajoute ensuite l'évaluation sur les cases RÉELLES
des photos de test (data/cells_cls/test), qui est la vraie mesure qui compte :
précision par case et nombre de grilles lues sans aucune erreur.

Usage :
    python scripts/train_cells.py --epochs 15
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # puis http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO, settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sudoku_cv.digit_classifier import DigitClassifier  # noqa: E402

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT = "digit_classification"


def evaluate_real_cells(weights: Path, test_dir: Path) -> tuple[dict, np.ndarray, list[str]]:
    """Évalue sur les cases réelles. Fichiers nommés <photo>_<index>.png dans test/<classe>/."""
    clf = DigitClassifier(weights)
    paths, labels = [], []
    for c in range(10):
        for p in sorted((test_dir / str(c)).glob("*.png")):
            paths.append(p)
            labels.append(c)
    probs = clf.predict_probs([cv2.imread(str(p)) for p in paths])
    preds = probs.argmax(axis=1)
    labels = np.array(labels)

    confusion = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels, preds):
        confusion[t, p] += 1

    errors_by_photo: dict[str, int] = defaultdict(int)
    errors = []
    for path, t, p in zip(paths, labels, preds):
        photo = path.stem.rsplit("_", 1)[0]
        errors_by_photo[photo] += int(t != p)
        if t != p:
            errors.append(f"{path.name}: vrai={t} prédit={p}")
    metrics = {
        "real_cell_accuracy": float((preds == labels).mean()),
        "real_digit_accuracy": float((preds[labels > 0] == labels[labels > 0]).mean()),
        "real_empty_accuracy": float((preds[labels == 0] == 0).mean()),
        "real_grids_perfect": sum(v == 0 for v in errors_by_photo.values()),
        "real_grids_total": len(errors_by_photo),
        "real_errors": int((preds != labels).sum()),
    }
    return metrics, confusion, errors


def plot_confusion(confusion: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(confusion, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if confusion[i, j]:
                ax.text(j, i, confusion[i, j], ha="center", va="center", color="black", fontsize=8)
    ax.set_xticks(range(10), ["vide"] + [str(d) for d in range(1, 10)])
    ax.set_yticks(range(10), ["vide"] + [str(d) for d in range(1, 10)])
    ax.set_xlabel("prédit")
    ax.set_ylabel("vrai")
    ax.set_title("Cases réelles (photos de test)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/cells_cls")
    parser.add_argument("--model", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--name", default="cells")
    parser.add_argument("--note", default="", help="description libre du run (tag MLflow)")
    args = parser.parse_args()

    settings.update({"mlflow": True})
    os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    stats = json.loads((args.data / "stats.json").read_text())
    with mlflow.start_run(run_name=args.name) as run:
        mlflow.set_tags({"model": args.model, "note": args.note, "dataset": "rg1990 fonts + synth + real empties"})
        mlflow.log_params({f"data_{k}_total": sum(v) for k, v in stats["counts"].items()})
        mlflow.log_params({"n_fonts": stats["n_fonts"], "n_val_fonts": stats["n_val_fonts"]})
        mlflow.log_artifact(str(args.data / "stats.json"))

        model = YOLO(args.model)
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device="cpu",
            workers=4,
            project=str(ROOT / "runs/cells"),
            name=args.name,
            exist_ok=True,
            fliplr=0.0,  # un chiffre retourné n'est plus le même chiffre
            erasing=0.0,  # l'effacement aléatoire peut transformer un chiffre en case « vide »
            plots=True,
        )
        best = Path(model.trainer.best)

        metrics, confusion, errors = evaluate_real_cells(best, args.data / "test")
        mlflow.log_metrics(metrics)
        out = best.parent.parent
        plot_confusion(confusion, out / "real_confusion.png")
        (out / "real_errors.txt").write_text("\n".join(errors) or "aucune erreur", encoding="utf-8")
        mlflow.log_artifact(str(out / "real_confusion.png"))
        mlflow.log_artifact(str(out / "real_errors.txt"))

        print(f"\nRun MLflow : {run.info.run_id}")
        print(json.dumps(metrics, indent=2))
        print("\n".join(errors[:30]))


if __name__ == "__main__":
    main()
