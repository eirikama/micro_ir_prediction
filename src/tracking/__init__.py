"""Experiment tracking backends.

Select a backend by setting ``tracking: <name>`` in config.yaml
(or overriding on the command line):

    tracking: mlflow   # default — logs to MLflow / SQLite
    tracking: wandb    # logs to Weights & Biases
    tracking: none     # disables all tracking
"""
from __future__ import annotations

from src.tracking.base import Tracker
from src.tracking.noop_tracker import NoOpTracker


def build_tracker(cfg, run_name: str) -> Tracker:
    """Instantiate the correct tracker from the Hydra config.

    Reads ``cfg.tracking`` (default: ``"mlflow"``).
    Wandb entity is read from ``cfg.wandb.entity`` if present.
    """
    backend = cfg.get("tracking", "mlflow")

    if backend == "mlflow":
        from src.tracking.mlflow_tracker import MLflowTracker
        return MLflowTracker(
            experiment_name=cfg.mlflow.experiment_name,
            tracking_uri=cfg.mlflow.tracking_uri,
            run_name=run_name,
        )

    if backend == "wandb":
        from src.tracking.wandb_tracker import WandbTracker
        entity = None
        try:
            entity = cfg.wandb.entity
        except Exception:
            pass
        return WandbTracker(
            project=cfg.mlflow.experiment_name,
            run_name=run_name,
            entity=entity,
        )

    if backend == "none":
        return NoOpTracker()

    raise ValueError(
        f"Unknown tracking backend '{backend}'. "
        "Expected one of: mlflow, wandb, none."
    )


__all__ = ["Tracker", "NoOpTracker", "build_tracker"]
