"""Weights & Biases implementation of the Tracker protocol.

Usage
-----
Set ``tracking: wandb`` in config.yaml (or pass on the command line).
Optionally add a ``wandb:`` section to config.yaml for project / entity:

    wandb:
      entity: my-team          # optional
"""
from __future__ import annotations

import json
import os
import tempfile
import traceback as tb
from typing import Any


class WandbTracker:
    """Wraps wandb + PL's WandbLogger behind the Tracker interface."""

    def __init__(self, project: str, run_name: str, entity: str | None = None) -> None:
        self._project  = project
        self._run_name = run_name
        self._entity   = entity
        self._run      = None
        self._pl_logger = None

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        if self._run is None:
            raise RuntimeError("WandbTracker not started — use 'with tracker:' first")
        return self._run.id

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "WandbTracker":
        import wandb
        from pytorch_lightning.loggers import WandbLogger

        self._run = wandb.init(
            project=self._project,
            name=self._run_name,
            entity=self._entity,
            reinit=True,
        )
        self._pl_logger = WandbLogger(
            project=self._project,
            name=self._run_name,
            id=self._run.id,
            entity=self._entity,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        import wandb

        try:
            if exc_type is not None:
                wandb.run.summary["status"] = "failed"
                wandb.run.summary["error"]  = str(exc_val)
                # Log traceback as a text artifact
                self.log_text(tb.format_exc(), "errors/main_error.txt")
            else:
                wandb.run.summary["status"] = "completed"
        except Exception:
            pass
        try:
            wandb.finish()
        except Exception:
            pass
        return False

    # ── logging ───────────────────────────────────────────────────────────────

    def log_param(self, key: str, value) -> None:
        import wandb
        wandb.config.update({key: value}, allow_val_change=True)

    def log_params(self, params: dict) -> None:
        import wandb
        wandb.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        import wandb
        wandb.log(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        import wandb
        wandb.log({key: value}, step=step)

    def set_tags(self, tags: dict) -> None:
        import wandb
        wandb.run.summary.update({str(k): str(v) for k, v in tags.items()})

    def set_tag(self, key: str, value: Any) -> None:
        import wandb
        wandb.run.summary[key] = str(value)

    def log_text(self, text: str, artifact_name: str) -> None:
        import wandb
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            tmp = f.name
        try:
            artifact = wandb.Artifact(
                name=artifact_name.replace("/", "_").replace(".", "_"),
                type="text",
            )
            artifact.add_file(tmp, name=artifact_name)
            wandb.log_artifact(artifact)
        finally:
            os.unlink(tmp)

    def log_dict(self, d: dict, artifact_name: str) -> None:
        import wandb
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(d, f, indent=2)
            tmp = f.name
        try:
            artifact = wandb.Artifact(
                name=artifact_name.replace("/", "_").replace(".", "_"),
                type="data",
            )
            artifact.add_file(tmp, name=artifact_name)
            wandb.log_artifact(artifact)
        finally:
            os.unlink(tmp)

    # ── PL integration ────────────────────────────────────────────────────────

    def pl_logger(self):
        return self._pl_logger
