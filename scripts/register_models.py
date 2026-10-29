"""Enregistre les poids YOLO d'un run dans le Model Registry MLflow et pose l'alias `champion`.

Les poids Ultralytics (.pt) ne sont pas un « flavor » MLflow : on enregistre le dossier
`weights/` du run comme version de modèle. L'application les retrouve ensuite par leur nom
et leur alias (`models:/sudoku-digit-classifier@champion`) sans connaître l'id du run.

Usage :
    python scripts/register_models.py                       # grid_v1 + cells_v2 (défaut)
    python scripts/register_models.py --digit-run cells_v3  # promeut un nouveau classifieur
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
GRID_MODEL = "sudoku-grid-detector"
DIGIT_MODEL = "sudoku-digit-classifier"


def register(client: mlflow.MlflowClient, model: str, experiment: str, run_name: str, description: str) -> None:
    exp = client.get_experiment_by_name(experiment)
    runs = client.search_runs([exp.experiment_id], f"tags.mlflow.runName = '{run_name}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit(f"run {run_name!r} introuvable dans {experiment!r}")
    run = runs[0]
    try:
        client.create_registered_model(model, description=description)
    except mlflow.exceptions.MlflowException:
        pass  # déjà enregistré
    version = client.create_model_version(model, f"{run.info.artifact_uri}/weights", run_id=run.info.run_id,
                                          tags={"run_name": run_name})
    client.set_registered_model_alias(model, "champion", version.version)
    print(f"{model} v{version.version} <- {run_name} ({run.info.run_id}), alias champion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-run", default="grid_v1")
    parser.add_argument("--digit-run", default="cells_v2")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    register(client, GRID_MODEL, "grid_detection", args.grid_run,
             "YOLO detect (1 classe) : bbox de la grille de sudoku dans la photo.")
    register(client, DIGIT_MODEL, "digit_classification", args.digit_run,
             "YOLO-cls 10 classes : contenu d'une case (0 = vide, 1..9).")


if __name__ == "__main__":
    main()
