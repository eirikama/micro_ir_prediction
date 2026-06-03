"""Tests for the experiment tracking layer."""
from __future__ import annotations

import pytest


# ── NoOpTracker ───────────────────────────────────────────────────────────────

class TestNoOpTracker:

    @pytest.fixture
    def tracker(self):
        from src.tracking.noop_tracker import NoOpTracker
        return NoOpTracker()

    def test_context_manager(self, tracker):
        with tracker as t:
            assert t is tracker

    def test_run_id_is_empty_string(self, tracker):
        with tracker:
            assert tracker.run_id == ""

    def test_log_params_no_error(self, tracker):
        with tracker:
            tracker.log_params({"lr": 1e-3, "batch_size": 64})

    def test_log_param_no_error(self, tracker):
        with tracker:
            tracker.log_param("lr", 1e-3)

    def test_log_metrics_no_error(self, tracker):
        with tracker:
            tracker.log_metrics({"loss": 0.5, "acc": 0.9})

    def test_log_metric_no_error(self, tracker):
        with tracker:
            tracker.log_metric("loss", 0.5, step=1)

    def test_set_tags_no_error(self, tracker):
        with tracker:
            tracker.set_tags({"status": "running", "user": "eirik"})

    def test_set_tag_no_error(self, tracker):
        with tracker:
            tracker.set_tag("status", "completed")

    def test_log_text_no_error(self, tracker):
        with tracker:
            tracker.log_text("some text content", "notes.txt")

    def test_log_dict_no_error(self, tracker):
        with tracker:
            tracker.log_dict({"class_0": 0, "class_1": 1}, "label_encoding.json")

    def test_pl_logger_returns_none(self, tracker):
        with tracker:
            assert tracker.pl_logger() is None

    def test_does_not_suppress_exceptions(self, tracker):
        with pytest.raises(RuntimeError):
            with tracker:
                raise RuntimeError("should propagate")


# ── build_tracker factory ─────────────────────────────────────────────────────

class TestBuildTracker:

    def _cfg(self, backend: str):
        from omegaconf import OmegaConf
        return OmegaConf.create({
            "tracking": backend,
            "mlflow": {
                "experiment_name": "test_experiment",
                "tracking_uri":    "sqlite:///test.db",
            },
        })

    def test_none_backend_returns_noop(self):
        from src.tracking import build_tracker
        from src.tracking.noop_tracker import NoOpTracker
        t = build_tracker(self._cfg("none"), "run_0")
        assert isinstance(t, NoOpTracker)

    def test_mlflow_backend_returns_mlflow_tracker(self):
        from src.tracking import build_tracker
        from src.tracking.mlflow_tracker import MLflowTracker
        t = build_tracker(self._cfg("mlflow"), "run_0")
        assert isinstance(t, MLflowTracker)

    def test_unknown_backend_raises(self):
        from src.tracking import build_tracker
        with pytest.raises(ValueError, match="Unknown tracking backend"):
            build_tracker(self._cfg("tensorboard"), "run_0")
