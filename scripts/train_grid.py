"""Entraîne le détecteur de grille YOLO detect et trace tout dans MLflow.

Au-delà du mAP d'Ultralytics, on mesure ce qui compte pour la suite du pipeline : les
4 COINS de la grille sont-ils retrouvés ? Trois méthodes comparées sur le jeu de test :
  A. OpenCV seul       : find_grid_corners(image)               (référence, sans YOLO)
  B. YOLO + OpenCV     : find_grid_corners(image, bbox YOLO)    (méthode visée)
  C. YOLO seul         : les 4 coins de la bbox YOLO            (repli)
Erreur d'un coin = distance au coin annoté / côté moyen de la grille. Une image est
« réussie » si ses 4 coins sont à moins de 3 % (≈ 1/4 de case : le découpage reste bon).

Usage :
    python scripts/train_grid.py --epochs 60
    python scripts/train_grid.py --resume          # reprend runs/grid/grid_v1/weights/last.pt
    python scripts/train_grid.py --eval-only --weights runs/grid/grid_v1/weights/best.pt
    .venv/Scripts/mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import mlflow
import numpy as np
from ultralytics import YOLO, settings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sudoku_cv.warp import find_grid_corners, order_corners  # noqa: E402

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT = "grid_detection"
SUCCESS_THRESHOLD = 0.03


def corner_error(pred: Optional[np.ndarray], truth: np.ndarray) -> float:
    """Erreur max des 4 coins, relative au côté moyen de la grille (inf si pas de prédiction)."""
    if pred is None:
        return float("inf")
    truth = order_corners(truth)
    side = np.mean([np.linalg.norm(truth[i] - truth[(i + 1) % 4]) for i in range(4)])
    return float(np.linalg.norm(order_corners(pred) - truth, axis=1).max() / side)


def evaluate_corners(weights: Optional[Path], data_dir: Path, split: str = "test") -> tuple[dict, list[dict]]:
    """Compare les méthodes A/B/C sur un ensemble. Sans poids, seule A est évaluée."""
    from sudoku_cv.grid_detector import GridDetector

    infos = json.loads((data_dir / "corners.json").read_text(encoding="utf-8"))
    detector = GridDetector(weights) if weights else None
    rows = []
    for name, info in sorted(infos.items()):
        if info["split"] != split:
            continue
        image = cv2.imread(str(data_dir / "images" / split / f"{name}.jpg"))
        truth = np.array(info["corners"], dtype=np.float32)
        row = {"image": name, "source": info["source"], "A_opencv": corner_error(find_grid_corners(image), truth)}
        if detector:
            det = detector.detect(image)
            if det is None:
                row.update(B_yolo_opencv=float("inf"), C_yolo_bbox=float("inf"), yolo_conf=0.0)
            else:
                (x1, y1, x2, y2), conf = det
                bbox_corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                row.update(
                    B_yolo_opencv=corner_error(find_grid_corners(image, bbox=(x1, y1, x2, y2)), truth),
                    C_yolo_bbox=corner_error(bbox_corners, truth),
                    yolo_conf=conf,
                )
        rows.append(row)

    metrics = {}
    for method in ("A_opencv", "B_yolo_opencv", "C_yolo_bbox"):
        errs = np.array([r[method] for r in rows if method in r])
        if len(errs) == 0:
            continue
        metrics[f"{split}_{method}_success"] = float((errs < SUCCESS_THRESHOLD).mean())
        metrics[f"{split}_{method}_median_err"] = float(np.median(errs))
        for src in ("wichtounet", "rg1990"):
            e = np.array([r[method] for r in rows if r["source"] == src and method in r])
            if len(e):
                metrics[f"{split}_{method}_success_{src}"] = float((e < SUCCESS_THRESHOLD).mean())
    metrics[f"{split}_n_images"] = len(rows)
    return metrics, rows


def backfill_epoch_metrics(results_csv: Path) -> None:
    """Après une reprise, Ultralytics refuse de logger dans le run MLflow existant (paramètre
    `model` différent) : on recopie les époques manquantes depuis results.csv, avec les mêmes
    noms (parenthèses retirées) et la même numérotation (step = époque - 1)."""
    import csv

    client = mlflow.MlflowClient()
    run_id = mlflow.active_run().info.run_id
    logged = {m.step for m in client.get_metric_history(run_id, "metrics/mAP50B")}
    with open(results_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = int(row["epoch"]) - 1
            if step in logged:
                continue
            mlflow.log_metrics({k.strip().replace("(", "").replace(")", ""): float(v)
                                for k, v in row.items() if k not in ("epoch", "time")}, step=step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/grid_det")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="grid_v1")
    parser.add_argument("--note", default="")
    parser.add_argument("--eval-only", action="store_true", help="évalue des poids existants sans entraîner")
    parser.add_argument("--weights", type=Path, help="poids à évaluer (--eval-only) ; absent = OpenCV seul")
    parser.add_argument("--resume", action="store_true",
                        help="reprend un entraînement interrompu (last.pt) dans le même run MLflow")
    args = parser.parse_args()

    settings.update({"mlflow": True})
    os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    run_id = None
    if args.resume:  # on rattache la reprise au run MLflow resté « RUNNING »
        found = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{args.name}' and status = 'RUNNING'",
                                   max_results=1, output_format="list")
        run_id = found[0].info.run_id if found else None

    with mlflow.start_run(run_id=run_id, run_name=None if run_id else args.name) as run:
        mlflow.set_tags({"model": args.model if not args.eval_only else str(args.weights), "note": args.note})
        weights = args.weights
        if not args.eval_only:
            if args.resume:
                model = YOLO(str(ROOT / "runs/grid" / args.name / "weights/last.pt"))
                model.train(resume=True)
                backfill_epoch_metrics(ROOT / "runs/grid" / args.name / "results.csv")
            else:
                model = YOLO(args.model)
                model.train(
                    data=str(args.data / "data.yaml"),
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device="cpu",
                    workers=4,
                    project=str(ROOT / "runs/grid"),
                    name=args.name,
                    exist_ok=True,
                    patience=20,  # arrêt si le mAP de val ne progresse plus pendant 20 époques
                    degrees=10.0,  # photos prises un peu de travers
                    perspective=0.0005,  # vues obliques
                    flipud=0.0,  # une page à l'envers est rare ; on garde fliplr (bbox symétrique)
                    plots=True,
                )
            weights = Path(model.trainer.best)
            # mAP sur le jeu de test (jamais utilisé pendant l'entraînement)
            test = YOLO(str(weights)).val(data=str(args.data / "data.yaml"), split="test", device="cpu",
                                         project=str(ROOT / "runs/grid"), name=f"{args.name}_test", exist_ok=True)
            mlflow.log_metrics({"test_mAP50": float(test.box.map50), "test_mAP50_95": float(test.box.map),
                                "test_precision": float(test.box.mp), "test_recall": float(test.box.mr)})

        metrics, rows = evaluate_corners(weights, args.data, "test")
        mlflow.log_metrics(metrics)
        out = ROOT / "runs/grid" / args.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "corner_eval_test.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        mlflow.log_artifact(str(out / "corner_eval_test.json"))

        print(f"\nRun MLflow : {run.info.run_id}")
        print(json.dumps(metrics, indent=2))
        print("\nÉchecs (erreur > 3 %) :")
        for r in rows:
            fails = [m for m in ("A_opencv", "B_yolo_opencv", "C_yolo_bbox") if m in r and r[m] >= SUCCESS_THRESHOLD]
            if fails:
                print(f"  {r['image']:18s} " + "  ".join(f"{m}={r[m]:.3f}" for m in fails))


if __name__ == "__main__":
    main()
