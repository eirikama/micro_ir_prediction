"""Tests for spectral augmentation functions.

Mie-scattering tests are skipped when the Cython extensions are not compiled
(e.g. in CI before building). All other augmentations are pure Python/NumPy/Torch
and always run.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import N_SPECTRA, SPECTRAL_LEN

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def spectra(rng):
    """(N, L) spectra with positive values, float32."""
    return (rng.random((N_SPECTRA, SPECTRAL_LEN), dtype=np.float32) + 0.5)


@pytest.fixture
def mask_all(spectra):
    return np.ones(len(spectra), dtype=bool)


@pytest.fixture
def wn(wavenumbers):
    return wavenumbers


# ── Raman augmentations ───────────────────────────────────────────────────────

def test_cosmic_rays_shape(spectra, mask_all, wn):
    from src.data.augmentation import _apply_cosmic_rays
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"spike_rate": 0.001, "amplitude_range": [2.0, 5.0], "max_width": 3})
    out = _apply_cosmic_rays(spectra.copy(), mask_all, wn, cfg)
    assert out.shape == spectra.shape


def test_cosmic_rays_no_nan(spectra, mask_all, wn):
    from src.data.augmentation import _apply_cosmic_rays
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"spike_rate": 0.001, "amplitude_range": [2.0, 5.0], "max_width": 3})
    out = _apply_cosmic_rays(spectra.copy(), mask_all, wn, cfg)
    assert np.isfinite(out).all()


def test_shot_noise_shape(spectra, mask_all, wn):
    from src.data.augmentation import _apply_shot_noise
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"scale": 0.02})
    out = _apply_shot_noise(spectra.copy(), mask_all, wn, cfg)
    assert out.shape == spectra.shape


def test_shot_noise_no_nan(spectra, mask_all, wn):
    from src.data.augmentation import _apply_shot_noise
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"scale": 0.02})
    out = _apply_shot_noise(spectra.copy(), mask_all, wn, cfg)
    assert np.isfinite(out).all()


def test_fluorescence_fit_and_apply(spectra, wn):
    from src.data.augmentation import _apply_fluorescence
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "enabled": True, "signal_only": True, "ratio": 0.5,
        "n_components": 3, "poly_degree": 2, "amplitude_range": [0.0, 0.1],
    })
    _apply_fluorescence.fit(wn, spectra, cfg)
    mask = np.ones(len(spectra), dtype=bool)
    out  = _apply_fluorescence(spectra.copy(), mask, wn, cfg)
    assert out.shape == spectra.shape
    assert np.isfinite(out).all()


def test_fluorescence_raises_before_fit(spectra, wn):
    from src.data.augmentation import _FluorescenceAugWrapper
    from omegaconf import OmegaConf
    wrapper = _FluorescenceAugWrapper()
    cfg  = OmegaConf.create({"amplitude_range": [0.0, 0.1]})
    mask = np.ones(len(spectra), dtype=bool)
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        wrapper(spectra.copy(), mask, wn, cfg)


# ── IR augmentations ──────────────────────────────────────────────────────────

def test_noise_shape(spectra, mask_all, wn):
    from src.data.augmentation import _apply_noise
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"max_level": 0.01})
    out = _apply_noise(spectra.copy(), mask_all, wn, cfg)
    assert out.shape == spectra.shape


def test_noise_no_nan(spectra, mask_all, wn):
    from src.data.augmentation import _apply_noise
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"max_level": 0.01})
    out = _apply_noise(spectra.copy(), mask_all, wn, cfg)
    assert np.isfinite(out).all()


def test_polynomial_baseline_shape(spectra, mask_all, wn):
    from src.data.augmentation import _apply_polynomial_baseline
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "param_ranges": [[-1.0, 1.0], [0.1, 2.0], [-0.01, 0.01], [-0.001, 0.001]]
    })
    out = _apply_polynomial_baseline(spectra.copy(), mask_all, wn, cfg)
    assert out.shape == spectra.shape


def test_mie_scattering_skipped_without_cython():
    """Importing the Mie function gracefully fails when Cython is absent."""
    from src.data.augmentation import _MIE_AVAILABLE
    if not _MIE_AVAILABLE:
        pytest.skip("Cython extensions not compiled — Mie scattering skipped")


@pytest.mark.skipif(
    not __import__("src.data.augmentation", fromlist=["_MIE_AVAILABLE"])._MIE_AVAILABLE,
    reason="Cython Mie extensions not compiled",
)
def test_mie_scattering_shape(spectra, mask_all, wn):
    from src.data.augmentation import _apply_mie_scattering
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "variant": "spherical",
        "n0_min": 1.3, "n0_max": 1.65,
        "r_min":  3.0, "r_max":  17.0,
        "n_imag_min": 1e-5, "n_imag_max": 1e-1,
        "h_min": 1.0, "h_max": 2.0,
        "scale_min": 2.5, "scale_max": 5.5,
        "theta_min": 0.2, "theta_max": 0.45,
    })
    out = _apply_mie_scattering(spectra[:4].copy(), mask_all[:4], wn, cfg)
    assert out.shape == spectra[:4].shape
