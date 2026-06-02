"""No-op tracker — runs the pipeline without any experiment tracking."""
from __future__ import annotations

from typing import Any


class NoOpTracker:
    """Silently discards all logging calls.

    Use with ``tracking: none`` in config.yaml to run without MLflow or WandB.
    The PL Trainer will use its default CSV logger.
    """

    @property
    def run_id(self) -> str:
        return ""

    def __enter__(self) -> "NoOpTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def log_param(self, key: str, value) -> None:
        pass

    def log_params(self, params: dict) -> None:
        pass

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        pass

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        pass

    def set_tags(self, tags: dict) -> None:
        pass

    def set_tag(self, key: str, value: Any) -> None:
        pass

    def log_text(self, text: str, artifact_name: str) -> None:
        pass

    def log_dict(self, d: dict, artifact_name: str) -> None:
        pass

    def pl_logger(self):
        return None   # PL falls back to its default CSV logger
