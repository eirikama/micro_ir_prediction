"""MLflow implementation of the Tracker protocol."""
from __future__ import annotations

import traceback as tb
from typing import Any

import mlflow
from pytorch_lightning.loggers import MLFlowLogger


class MLflowTracker:
    """Wraps MLflow + PL's MLFlowLogger behind the Tracker interface.

    Shares a single MLflow run between the PL logger (used inside the
    Trainer) and direct mlflow API calls, exactly as the original main.py did.
    """

    def __init__(self, experiment_name: str, tracking_uri: str, run_name: str) -> None:
        self._experiment_name = experiment_name
        self._tracking_uri    = tracking_uri
        self._run_name        = run_name
        self._mlf_logger: MLFlowLogger | None = None
        self._run_ctx = None
        self._run     = None

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        if self._run is None:
            raise RuntimeError("MLflowTracker not started — use 'with tracker:' first")
        return self._run.info.run_id

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "MLflowTracker":
        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        # Create the PL logger first — it creates the MLflow run internally.
        self._mlf_logger = MLFlowLogger(
            experiment_name=self._experiment_name,
            tracking_uri=self._tracking_uri,
            run_name=self._run_name,
        )
        # Re-attach the mlflow API to the same run so both share one run ID.
        self._run_ctx = mlflow.start_run(run_id=self._mlf_logger.run_id)
        self._run     = self._run_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if exc_type is not None:
                mlflow.log_text(tb.format_exc(), "errors/main_error.txt")
                mlflow.set_tag("error", str(exc_val))
                mlflow.set_tag("status", "failed")
            else:
                mlflow.set_tag("status", "completed")
        except Exception:
            pass
        self._run_ctx.__exit__(exc_type, exc_val, exc_tb)
        return False   # never suppress exceptions

    # ── logging ───────────────────────────────────────────────────────────────

    def log_param(self, key: str, value) -> None:
        mlflow.log_param(key, value)

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        mlflow.log_metric(key, value, step=step)

    def set_tags(self, tags: dict) -> None:
        mlflow.set_tags({str(k): str(v) for k, v in tags.items()})

    def set_tag(self, key: str, value: Any) -> None:
        mlflow.set_tag(key, str(value))

    def log_text(self, text: str, artifact_name: str) -> None:
        mlflow.log_text(text, artifact_name)

    def log_dict(self, d: dict, artifact_name: str) -> None:
        mlflow.log_dict(d, artifact_name)

    # ── PL integration ────────────────────────────────────────────────────────

    def pl_logger(self) -> MLFlowLogger:
        return self._mlf_logger
