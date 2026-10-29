"""Évaluation de bout en bout : photo -> solution, sur le jeu de test (jamais vu par les modèles).

Jeu de test : les 41 photos de test de data/grid_det (30 wichtounet + 11 rg1990), avec leur
grille vraie (corners.json). Aucune n'a servi à entraîner le détecteur ni le classifieur.

Plusieurs configurations sont comparées (un run MLflow chacune, expérience `pipeline_eval`),
en ajoutant les briques une à une pour mesurer l'apport de chacune :
  opencv_only : coins par OpenCV sur toute l'image, lecture brute
  yolo        : + détecteur YOLO (coins cherchés dans la bbox)
  yolo_rot    : + choix de l'orientation (4 quarts de tour)
  full        : + réparation par les règles du sudoku (2e choix des cases peu sûres)

Métrique principale : `solved_rate`, la part des photos dont la solution est exacte.

Usage :
    python scripts/evaluate.py
    python scripts/evaluate.py --configs full --renders
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mlflow
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sudoku_cv.pipeline import SudokuPipeline, render_solution  # noqa: E402
from sudoku_cv.solver import solve  # noqa: E402

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT = "pipeline_eval"
CONFIGS = {
    "opencv_only": dict(use_detector=False, try_rotations=False, repair=False),
    "yolo": dict(use_detector=True, try_rotations=False, repair=False),
    "yolo_rot": dict(use_detector=True, try_rotations=True, repair=False),
    "full": dict(use_detector=True, try_rotations=True, repair=True),
}


def grid_from_str(s: str) -> list[list[int]]:
    return [[int(s[9 * r + c]) for c in range(9)] for r in range(9)]


def evaluate(pipe: SudokuPipeline, samples: list[dict], render_dir: Path | None) -> tuple[dict, list[dict]]:
    rows = []
    for s in samples:
        image = cv2.imread(str(s["path"]))
        truth = s["truth"]
        res = pipe.run(image)
        raw_ok = int(sum(res.raw_grid[r][c] == truth[r][c] for r in range(9) for c in range(9))) if res.raw_grid else 0
        rows.append({
            "image": s["name"],
            "source": s["source"],
            "status": res.status,
            "bbox_conf": res.bbox_conf,
            "rotation": res.rotation,
            "raw_cells_correct": raw_ok,
            "raw_errors": [(r, c, truth[r][c], res.raw_grid[r][c]) for r in range(9) for c in range(9)
                           if res.raw_grid[r][c] != truth[r][c]] if res.raw_grid else None,
            "grid_correct": res.grid == truth,
            "solved_correct": res.solution is not None and res.solution == s["solution"],
            "repaired": res.repaired,
            "time": res.timings.get("total", 0.0),
        })
        if render_dir is not None:
            render_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(render_dir / f"{s['name']}.jpg"), render_solution(image, res))

    def rate(key, subset):
        return float(np.mean([bool(r[key]) for r in subset])) if subset else 0.0

    metrics = {
        "n_images": len(rows),
        "grid_found_rate": float(np.mean([r["status"] != "no_grid" for r in rows])),
        "raw_cell_accuracy": float(np.mean([r["raw_cells_correct"] / 81 for r in rows])),
        "raw_grid_perfect_rate": float(np.mean([r["raw_cells_correct"] == 81 for r in rows])),
        "grid_correct_rate": rate("grid_correct", rows),
        "solved_rate": rate("solved_correct", rows),
        # une réponse fausse affichée avec assurance est pire qu'un « je n'y arrive pas »
        "wrong_solution_rate": float(np.mean([r["status"] == "solved" and not r["solved_correct"] for r in rows])),
        "mean_time_s": float(np.mean([r["time"] for r in rows])),
    }
    for src in ("wichtounet", "rg1990"):
        sub = [r for r in rows if r["source"] == src]
        metrics[f"solved_rate_{src}"] = rate("solved_correct", sub)
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/grid_det")
    parser.add_argument("--split", default="test")
    parser.add_argument("--grid-weights", type=Path, default=ROOT / "runs/grid/grid_v1/weights/best.pt")
    parser.add_argument("--digit-weights", type=Path, default=ROOT / "runs/cells/cells_v2/weights/best.pt")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--renders", action="store_true", help="enregistre les photos annotées (config full)")
    args = parser.parse_args()

    infos = json.loads((args.data / "corners.json").read_text(encoding="utf-8"))
    samples = []
    for name, info in sorted(infos.items()):
        if info["split"] != args.split:
            continue
        truth = grid_from_str(info["grid"])
        samples.append({"name": name, "source": info["source"], "truth": truth, "solution": solve(truth),
                        "path": args.data / "images" / args.split / f"{name}.jpg"})
    assert all(s["solution"] for s in samples), "une grille vraie n'a pas de solution"

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    out = ROOT / "runs/eval"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.configs:
        cfg = CONFIGS[name]
        pipe = SudokuPipeline(args.grid_weights if cfg["use_detector"] else None, args.digit_weights,
                              try_rotations=cfg["try_rotations"], repair=cfg["repair"])
        render_dir = out / f"renders_{name}" if args.renders and name == "full" else None
        with mlflow.start_run(run_name=name):
            mlflow.log_params({**cfg, "split": args.split, "grid_weights": args.grid_weights.as_posix(),
                               "digit_weights": args.digit_weights.as_posix()})
            metrics, rows = evaluate(pipe, samples, render_dir)
            mlflow.log_metrics(metrics)
            path = out / f"{name}_{args.split}.json"
            path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
            mlflow.log_artifact(str(path))
        summary[name] = metrics
        print(f"\n== {name}")
        print(json.dumps(metrics, indent=2))
        for r in rows:
            if not r["solved_correct"]:
                print(f"  ÉCHEC {r['image']:14s} statut={r['status']:10s} rot={r['rotation']} "
                      f"cases ok={r['raw_cells_correct']}/81 réparé={r['repaired']} erreurs={r['raw_errors']}")

    print("\nRésumé (solved_rate) :", {k: round(v["solved_rate"], 3) for k, v in summary.items()})


if __name__ == "__main__":
    main()
