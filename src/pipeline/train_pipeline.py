"""Training pipeline — model construction, fitting, metric logging."""
from __future__ import annotations

import gc
import logging
import time

import torch
from omegaconf import DictConfig, open_dict

from src.models.aacnn import AACNN
from src.training.trainer_engine import run_training
from src.tracking.base import Tracker

log = logging.getLogger(__name__)


def run_training_pipeline(
    cfg: DictConfig,
    datamodule,
    tracker: Tracker,
) -> tuple[str, float]:
    """Build, train, and log the model.

    Returns ``(best_checkpoint_path, best_val_acc)`` so callers can both
    continue to inference and report the score to a hyperparameter sweeper.
    """
    model = AACNN(cfg.model)

    tracker.log_params({
        "Model/n_params_trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "Model/n_params_total": sum(p.numel() for p in model.parameters()),
    })

    log.info("Starting training…")
    t0 = time.time()
    best_path, best_score, final_epoch, stopped_early = run_training(
        cfg, model, datamodule, tracker.pl_logger()
    )
    train_seconds = time.time() - t0
    log.info("Training complete. Best checkpoint: %s", best_path)

    tracker.log_metrics({
        "Timer/train_duration_minutes": train_seconds / 60,
        "Timer/epochs_per_second":      final_epoch / train_seconds if train_seconds > 0 else 0.0,
        "Timer/samples_per_second":     final_epoch * cfg.data.batch_size / train_seconds
                                        if train_seconds > 0 else 0.0,
        "Accuracy/best_val_acc":        best_score,
        "Trainer/final_epoch":          final_epoch,
    })
    tracker.set_tag("stopped_early", str(stopped_early))
    tracker.set_tag("ckpt_path", best_path)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return best_path, best_score
