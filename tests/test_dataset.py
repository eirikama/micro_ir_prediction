"""Tests for SpectralDataset (the IterableDataset used in training)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from tests.conftest import N_CLASSES, N_SPECTRA, SPECTRAL_LEN


@pytest.fixture
def data_cfg():
    return OmegaConf.create({
        "batch_size":        N_CLASSES * 4,   # divisible by N_CLASSES
        "include_bkg_pixels": False,
        "augmentations":     [],
        "z_normalize":       False,
        "augment_train":     False,
        "augment_val":       False,
    })


@pytest.fixture
def dataset(spectra_2d, labels, data_cfg):
    from src.data.datamodule import SpectralDataset
    wn = np.linspace(400, 4000, SPECTRAL_LEN, dtype=np.float32)
    return SpectralDataset(spectra_2d, labels, wn, data_cfg, augment=False)


# ── basic shape checks ────────────────────────────────────────────────────────

def test_yields_correct_spectra_shape(dataset, data_cfg):
    it = iter(dataset)
    x, y = next(it)
    B = data_cfg.batch_size
    assert x.shape == (B, 1, SPECTRAL_LEN)
    assert x.dtype == torch.float32


def test_yields_correct_label_shape(dataset, data_cfg):
    it = iter(dataset)
    x, y = next(it)
    B = data_cfg.batch_size
    assert y.shape == (B,)
    assert y.dtype == torch.long


def test_labels_in_valid_range(dataset):
    it = iter(dataset)
    _, y = next(it)
    assert (y >= 0).all() and (y < N_CLASSES).all()


# ── balanced sampling ─────────────────────────────────────────────────────────

def test_class_balance(dataset, data_cfg):
    """Each class should appear samples_per_class times in each batch."""
    it = iter(dataset)
    _, y = next(it)
    samples_per_class = data_cfg.batch_size // N_CLASSES
    for cls in range(N_CLASSES):
        count = (y == cls).sum().item()
        # remainder may add a few extra; at minimum samples_per_class each
        assert count >= samples_per_class


# ── z-normalization ───────────────────────────────────────────────────────────

def test_z_normalize_zero_mean(spectra_2d, labels, data_cfg):
    from src.data.datamodule import SpectralDataset
    wn  = np.linspace(400, 4000, SPECTRAL_LEN, dtype=np.float32)
    cfg = OmegaConf.create({**dict(data_cfg), "z_normalize": True})
    ds  = SpectralDataset(spectra_2d, labels, wn, cfg, augment=False)
    x, _ = next(iter(ds))
    means = x.squeeze(1).mean(dim=1)   # mean per spectrum
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-4)


# ── augmentation flag ─────────────────────────────────────────────────────────

def test_augment_false_preserves_dtype(dataset):
    x, _ = next(iter(dataset))
    assert x.dtype == torch.float32


def test_multiple_batches_are_different(dataset):
    it   = iter(dataset)
    x1, _ = next(it)
    x2, _ = next(it)
    # Random sampling should produce different batches
    assert not torch.equal(x1, x2)
