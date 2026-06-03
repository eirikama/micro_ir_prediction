"""Tests for inference output saving (LMDB/zarr store)."""
from __future__ import annotations

import numpy as np
import pytest
import zarr

from tests.conftest import N_CLASSES


@pytest.fixture
def prob_map():
    """Tiny (4, 4, N_CLASSES) float16 probability map with valid probabilities."""
    rng  = np.random.default_rng(0)
    raw  = rng.random((4, 4, N_CLASSES), dtype=np.float32)
    norm = raw / raw.sum(axis=-1, keepdims=True)
    return norm.astype(np.float16)


@pytest.fixture
def lmdb_store(tmp_path):
    store = zarr.LMDBStore(str(tmp_path / "pred.lmdb"), map_size=int(1e9))
    grp   = zarr.open_group(store, mode="a")
    try:
        yield grp
    finally:
        store.close()


# ── save_inference_outputs_zarr ───────────────────────────────────────────────

def test_saves_argmax_map(prob_map, lmdb_store):
    from src.inference.export_inference import save_inference_outputs_zarr
    save_inference_outputs_zarr(
        prob_map=prob_map, image_name="img_001", store=lmdb_store,
        N=64, seed=0, true_idx=1, background_idx=0, top_k_save=2,
    )
    trial_key = [k for k in lmdb_store.keys()][0]
    argmax = lmdb_store[f"{trial_key}/img_001/argmax_map"][:]
    assert argmax.shape == (4, 4)
    assert argmax.dtype == np.uint8


def test_saves_bg_prob(prob_map, lmdb_store):
    from src.inference.export_inference import save_inference_outputs_zarr
    save_inference_outputs_zarr(
        prob_map=prob_map, image_name="img_002", store=lmdb_store,
        N=64, seed=0, true_idx=1, background_idx=0, top_k_save=2,
    )
    trial_key = [k for k in lmdb_store.keys()][0]
    bg = lmdb_store[f"{trial_key}/img_002/bg_prob"][:]
    assert bg.shape == (4, 4)
    assert bg.dtype == np.float16


def test_saves_top_k_arrays(prob_map, lmdb_store):
    from src.inference.export_inference import save_inference_outputs_zarr
    k = 2
    save_inference_outputs_zarr(
        prob_map=prob_map, image_name="img_003", store=lmdb_store,
        N=64, seed=0, true_idx=1, background_idx=0, top_k_save=k,
    )
    trial_key = [k for k in lmdb_store.keys()][0]
    grp = lmdb_store[f"{trial_key}/img_003"]
    assert grp["top_k_classes"].shape == (4, 4, 2)
    assert grp["top_k_probs"].shape   == (4, 4, 2)


def test_argmax_matches_prob_map(prob_map, lmdb_store):
    from src.inference.export_inference import save_inference_outputs_zarr
    save_inference_outputs_zarr(
        prob_map=prob_map, image_name="img_004", store=lmdb_store,
        N=64, seed=0, true_idx=1, background_idx=0, top_k_save=2,
    )
    trial_key = [k for k in lmdb_store.keys()][0]
    argmax    = lmdb_store[f"{trial_key}/img_004/argmax_map"][:].astype(int)
    expected  = prob_map.argmax(axis=-1).astype(int)
    np.testing.assert_array_equal(argmax, expected)


def test_attrs_stored(prob_map, lmdb_store):
    from src.inference.export_inference import save_inference_outputs_zarr
    save_inference_outputs_zarr(
        prob_map=prob_map, image_name="img_005", store=lmdb_store,
        N=32, seed=7, true_idx=0, background_idx=0, top_k_save=2,
    )
    trial_key = [k for k in lmdb_store.keys()][0]
    attrs = dict(lmdb_store[f"{trial_key}/img_005"].attrs)
    assert attrs["N"]    == 32
    assert attrs["seed"] == 7


# ── _chunk_for helper ─────────────────────────────────────────────────────────

def test_chunk_for_fits_within_limit():
    from src.inference.export_inference import _chunk_for
    H, W = 512, 512
    ch, cw = _chunk_for(H, W, max_mb=8.0)
    assert ch * cw * 2 / 1e6 <= 8.0   # float16 = 2 bytes


def test_chunk_for_small_image():
    from src.inference.export_inference import _chunk_for
    ch, cw = _chunk_for(16, 16, max_mb=8.0)
    assert ch == 16 and cw == 16      # fits immediately — no splitting needed
