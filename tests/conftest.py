"""Shared fixtures for the test suite.

All fixtures that need a real model checkpoint use scope="session" so the
(slow) Trainer.fit call happens only once per test session.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

# ── constants shared across test modules ──────────────────────────────────────
SPECTRAL_LEN = 400   # L — large enough that adaptive_pool(95) can downsample
N_CLASSES    = 3
N_SPECTRA    = 48    # must be divisible by N_CLASSES for stratified split


# ── tiny model config ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model_cfg():
    from src.config_schema import ModelConfig
    return OmegaConf.structured(ModelConfig(
        conv_channels=8,
        kernel_size=7,
        pred_dropout=0.0,
        num_classes=N_CLASSES,
        gamma=0.0,
        alpha=[1.0] * N_CLASSES,
        lr=1e-4,
        weight_decay=0.0,
    ))


@pytest.fixture(scope="session")
def tiny_model(model_cfg):
    from src.models.aacnn import AACNN
    return AACNN(model_cfg)


# ── synthetic spectra ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def spectra_2d(rng):
    """(N, L) flat spectra array with positive values."""
    s = rng.random((N_SPECTRA, SPECTRAL_LEN), dtype=np.float32) + 0.1
    return s


@pytest.fixture(scope="session")
def spectra_3d(rng):
    """(H, W, L) image cube with positive values."""
    return (rng.random((8, 8, SPECTRAL_LEN), dtype=np.float32) + 0.1)


@pytest.fixture(scope="session")
def labels():
    """Integer class labels balanced across N_CLASSES."""
    return np.repeat(np.arange(N_CLASSES), N_SPECTRA // N_CLASSES).astype(np.int64)


@pytest.fixture(scope="session")
def wavenumbers():
    return np.linspace(400, 4000, SPECTRAL_LEN, dtype=np.float32)


# ── real checkpoint (one Trainer.fit step) ────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory, model_cfg):
    """Save a tiny AACNN checkpoint to a temp directory.

    Uses a real Trainer.fit(max_steps=1) so the checkpoint is in the exact
    format that load_from_checkpoint expects.
    """
    import pytorch_lightning as pl
    from src.models.aacnn import AACNN

    tmp       = tmp_path_factory.mktemp("ckpts")
    ckpt_path = str(tmp / "tiny.ckpt")

    model = AACNN(model_cfg)

    x  = torch.randn(16, 1, SPECTRAL_LEN)
    y  = torch.randint(0, N_CLASSES, (16,))
    dl = DataLoader(TensorDataset(x, y), batch_size=16)

    trainer = pl.Trainer(
        max_steps=1,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloaders=dl)
    trainer.save_checkpoint(ckpt_path)
    return ckpt_path
