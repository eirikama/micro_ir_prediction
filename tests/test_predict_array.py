"""Tests for predict_array — the standalone numpy-array inference function."""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import N_CLASSES, SPECTRAL_LEN


# ── 3-D image input (H, W, L) ─────────────────────────────────────────────────

def test_3d_output_shape(tiny_ckpt, spectra_3d):
    from src.inference.inference_engine import predict_array
    H, W, _ = spectra_3d.shape
    out = predict_array(spectra_3d, tiny_ckpt, batch_size=32, device="cpu")
    assert out.shape == (H, W, N_CLASSES)


def test_3d_output_dtype(tiny_ckpt, spectra_3d):
    from src.inference.inference_engine import predict_array
    out = predict_array(spectra_3d, tiny_ckpt, batch_size=32, device="cpu")
    assert out.dtype == np.float16


def test_3d_probs_sum_to_one(tiny_ckpt, spectra_3d):
    from src.inference.inference_engine import predict_array
    out = predict_array(spectra_3d, tiny_ckpt, batch_size=32, device="cpu")
    sums = out.reshape(-1, N_CLASSES).sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-2)


# ── 2-D spectra input (N, L) ──────────────────────────────────────────────────

def test_2d_output_shape(tiny_ckpt, spectra_2d):
    from src.inference.inference_engine import predict_array
    N, _ = spectra_2d.shape
    out = predict_array(spectra_2d, tiny_ckpt, batch_size=16, device="cpu")
    assert out.shape == (N, N_CLASSES)


def test_2d_probs_sum_to_one(tiny_ckpt, spectra_2d):
    from src.inference.inference_engine import predict_array
    out = predict_array(spectra_2d, tiny_ckpt, batch_size=16, device="cpu")
    sums = out.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-2)


def test_2d_no_nan(tiny_ckpt, spectra_2d):
    from src.inference.inference_engine import predict_array
    out = predict_array(spectra_2d, tiny_ckpt, device="cpu")
    assert np.isfinite(out).all()


# ── z-normalisation option ────────────────────────────────────────────────────

def test_z_normalize_runs_without_error(tiny_ckpt, spectra_2d):
    from src.inference.inference_engine import predict_array
    out = predict_array(spectra_2d, tiny_ckpt, device="cpu", z_normalize=True)
    assert out.shape == (len(spectra_2d), N_CLASSES)


# ── small batch size (exercises the batch loop) ───────────────────────────────

def test_small_batch_size(tiny_ckpt, spectra_2d):
    from src.inference.inference_engine import predict_array
    out = predict_array(spectra_2d, tiny_ckpt, batch_size=3, device="cpu")
    assert out.shape == (len(spectra_2d), N_CLASSES)


# ── input validation ──────────────────────────────────────────────────────────

def test_wrong_ndim_raises(tiny_ckpt):
    from src.inference.inference_engine import predict_array
    bad = np.random.rand(4).astype(np.float32)
    with pytest.raises(ValueError, match="2-D.*3-D"):
        predict_array(bad, tiny_ckpt, device="cpu")
