"""Tests for zarr-based spectra sampling (get_training_data).

Uses in-memory zarr stores patched over zarr.open so no real data files
are needed and sampling.py itself requires no test-specific changes.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import zarr

from tests.conftest import N_CLASSES, SPECTRAL_LEN


def _make_zarr_root(n_images_per_class: int = 2) -> zarr.Group:
    """Return an in-memory zarr group that mimics the training-store layout."""
    store = zarr.MemoryStore()
    root  = zarr.open_group(store, mode="w")
    root.attrs["wavenumbers"] = np.linspace(400, 4000, SPECTRAL_LEN).tolist()
    root.attrs["classes"]     = [f"class_{i}" for i in range(N_CLASSES)]

    images = root.require_group("images")
    rng    = np.random.default_rng(99)

    for cls_idx in range(N_CLASSES):
        cls = f"class_{cls_idx}"
        for j in range(n_images_per_class):
            H, W  = 16, 16
            data  = rng.random((H, W, SPECTRAL_LEN), dtype=np.float32) + 0.5
            grp   = images.require_group(f"{cls}_img{j}")
            grp.array("data", data)
            grp.attrs["label"] = cls

    return root


# ── get_training_data ─────────────────────────────────────────────────────────

def test_returns_correct_spectra_count():
    from src.data.sampling import get_training_data
    root = _make_zarr_root()
    spectra_per_class = 8

    with patch("src.data.sampling.zarr.open", return_value=root):
        labels, spectra, wn, enc = get_training_data(
            zarr_path="__mem__",
            spectra_per_class=spectra_per_class,
            bkg_per_class=0,
            patch_size=8,
            background_max=0.0,
            sample_min=0.0,
            max_class_attempts=500,
            include_bkg_pixels=False,
        )

    assert len(labels) == N_CLASSES * spectra_per_class
    assert spectra.shape == (N_CLASSES * spectra_per_class, SPECTRAL_LEN)


def test_label_encoding_covers_all_classes():
    from src.data.sampling import get_training_data
    root = _make_zarr_root()

    with patch("src.data.sampling.zarr.open", return_value=root):
        _, _, _, enc = get_training_data(
            zarr_path="__mem__",
            spectra_per_class=4,
            bkg_per_class=0,
            patch_size=8,
            background_max=0.0,
            sample_min=0.0,
            max_class_attempts=500,
            include_bkg_pixels=False,
        )

    for i in range(N_CLASSES):
        assert f"class_{i}" in enc


def test_labels_and_spectra_same_length():
    from src.data.sampling import get_training_data
    root = _make_zarr_root()

    with patch("src.data.sampling.zarr.open", return_value=root):
        labels, spectra, _, _ = get_training_data(
            zarr_path="__mem__",
            spectra_per_class=6,
            bkg_per_class=0,
            patch_size=8,
            background_max=0.0,
            sample_min=0.0,
            max_class_attempts=500,
            include_bkg_pixels=False,
        )

    assert len(labels) == len(spectra)


def test_wavenumbers_returned():
    from src.data.sampling import get_training_data
    root = _make_zarr_root()

    with patch("src.data.sampling.zarr.open", return_value=root):
        _, _, wn, _ = get_training_data(
            zarr_path="__mem__",
            spectra_per_class=4,
            bkg_per_class=0,
            patch_size=8,
            background_max=0.0,
            sample_min=0.0,
            max_class_attempts=500,
            include_bkg_pixels=False,
        )

    assert len(wn) == SPECTRAL_LEN


def test_exceeding_attempts_raises():
    from src.data.sampling import get_training_data
    # all-zero spectra: means <= 0 → sample_min=0.5 will never be satisfied
    store = zarr.MemoryStore()
    root  = zarr.open_group(store, mode="w")
    root.attrs["wavenumbers"] = np.zeros(SPECTRAL_LEN).tolist()
    root.attrs["classes"]     = ["class_0"]
    grp = root.require_group("images/img0")
    grp.array("data", np.zeros((8, 8, SPECTRAL_LEN), dtype=np.float32))
    grp.attrs["label"] = "class_0"

    with patch("src.data.sampling.zarr.open", return_value=root):
        with pytest.raises(RuntimeError, match="Data Sampling Error"):
            get_training_data(
                zarr_path="__mem__",
                spectra_per_class=4,
                bkg_per_class=0,
                patch_size=4,
                background_max=0.0,
                sample_min=0.5,    # impossible — all spectra are zero
                max_class_attempts=5,
                include_bkg_pixels=False,
            )
