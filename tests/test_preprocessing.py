"""Tests for config-driven spectral preprocessing.

Pure numpy/scipy — no zarr or Lightning, so these always run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.data.preprocessing import (
    _BIOSPECTOOLS_AVAILABLE,
    apply_preprocessing,
    fit_preprocessing,
    load_state,
    resolve_steps,
    save_state,
)

L = 400


@pytest.fixture
def wn():
    return np.linspace(600.0, 3200.0, L)


@pytest.fixture
def spectra(wn):
    """Peaks + per-spectrum multiplicative scale + additive polynomial baseline
    — i.e. exactly the nuisance EMSC is meant to remove."""
    rng = np.random.default_rng(0)
    peaks = (
        np.exp(-((wn - 1650.0) ** 2) / (2 * 25.0 ** 2))
        + 0.7 * np.exp(-((wn - 2900.0) ** 2) / (2 * 40.0 ** 2))
    )
    return np.stack([
        (0.5 + 2 * rng.random()) * peaks
        + rng.uniform(-1, 1)
        + rng.uniform(-0.5, 0.5) * (wn - wn.mean()) / 1000.0
        for _ in range(48)
    ])


def _steps(*d):
    return OmegaConf.create(list(d))


# ── switches ─────────────────────────────────────────────────────────────────

def test_missing_block_is_identity(spectra, wn):
    for block in (None, [], {}):
        X, w = apply_preprocessing(spectra.copy(), wn, block)
        np.testing.assert_allclose(X, spectra)
        np.testing.assert_allclose(w, wn)


def test_master_switch_off_disables_everything(spectra, wn):
    block = OmegaConf.create({
        "enabled": False,
        "steps": [{"type": "crop", "keep": [[1000, 1800]]},
                  {"type": "savgol", "window": 11, "polyorder": 3, "deriv": 1}],
    })
    assert resolve_steps(block) == []
    X, w = apply_preprocessing(spectra.copy(), wn, block)
    np.testing.assert_allclose(X, spectra)
    np.testing.assert_allclose(w, wn)          # crop did not run


def test_master_switch_on_runs_steps(spectra, wn):
    block = OmegaConf.create({
        "enabled": True,
        "steps": [{"type": "crop", "keep": [[1000, 1800]]}],
    })
    X, w = apply_preprocessing(spectra.copy(), wn, block)
    assert X.shape[1] < spectra.shape[1]
    assert w.min() >= 1000.0 and w.max() <= 1800.0


def test_per_step_switch(spectra, wn):
    on  = _steps({"type": "crop", "keep": [[1000, 1800]]})
    off = _steps({"type": "crop", "keep": [[1000, 1800]], "enabled": False})

    X_on, w_on = apply_preprocessing(spectra.copy(), wn, on)
    X_off, w_off = apply_preprocessing(spectra.copy(), wn, off)

    assert X_on.shape[1] < spectra.shape[1]
    np.testing.assert_allclose(X_off, spectra)
    np.testing.assert_allclose(w_off, wn)


def test_bare_list_and_dict_forms_agree(spectra, wn):
    as_list = _steps({"type": "snv"})
    as_dict = OmegaConf.create({"enabled": True, "steps": [{"type": "snv"}]})
    a, _ = apply_preprocessing(spectra.copy(), wn, as_list)
    b, _ = apply_preprocessing(spectra.copy(), wn, as_dict)
    np.testing.assert_allclose(a, b)


def test_unknown_step_raises_even_when_disabled():
    """A typo must surface immediately, not the one time you re-enable it."""
    for block in (
        _steps({"type": "savgol_typo", "enabled": False}),
        OmegaConf.create({"enabled": False, "steps": [{"type": "nope"}]}),
    ):
        with pytest.raises(KeyError, match="Unknown preprocessing step"):
            resolve_steps(block)


# ── individual steps ─────────────────────────────────────────────────────────

def test_crop_keeps_only_requested_windows(spectra, wn):
    block = _steps({"type": "crop", "keep": [[1000, 1800], [2800, 3000]]})
    X, w = apply_preprocessing(spectra.copy(), wn, block)
    assert X.shape[0] == spectra.shape[0]
    assert X.shape[1] == len(w)
    assert ((w >= 1000) & (w <= 1800) | (w >= 2800) & (w <= 3000)).all()


def test_crop_rejects_empty_selection(spectra, wn):
    with pytest.raises(ValueError, match="kept 0 channels"):
        apply_preprocessing(spectra.copy(), wn, _steps(
            {"type": "crop", "keep": [[10.0, 20.0]]}))


def test_savgol_derivative_removes_constant_offset(spectra, wn):
    block = _steps({"type": "savgol", "window": 11, "polyorder": 3, "deriv": 1})
    a, _ = apply_preprocessing(spectra.copy(), wn, block)
    b, _ = apply_preprocessing(spectra.copy() + 5.0, wn, block)
    np.testing.assert_allclose(a, b, atol=1e-8)


def test_snv_gives_zero_mean_unit_std(spectra, wn):
    X, _ = apply_preprocessing(spectra.copy(), wn, _steps({"type": "snv"}))
    np.testing.assert_allclose(X.mean(axis=1), 0.0, atol=1e-9)
    np.testing.assert_allclose(X.std(axis=1), 1.0, atol=1e-6)


def test_emsc_removes_scale_and_baseline_nuisance(spectra, wn):
    block = _steps({"type": "emsc", "poly_order": 2})
    state = fit_preprocessing(spectra, wn, block)
    X, _ = apply_preprocessing(spectra.copy(), wn, block, state)
    # spectra differed only by scale + polynomial baseline, so after EMSC the
    # spread of their peak-to-peak ranges should collapse
    before = np.std([np.ptp(s) for s in spectra])
    after = np.std([np.ptp(s) for s in X])
    assert after < 0.05 * before


def test_emsc_without_fitted_state_raises(spectra, wn):
    with pytest.raises(RuntimeError, match="fitted reference"):
        apply_preprocessing(spectra.copy(), wn, _steps({"type": "emsc"}))


def test_emsc_reference_is_fitted_on_the_post_crop_axis(spectra, wn):
    """Stateful steps must be fitted on the representation they will see."""
    block = _steps(
        {"type": "crop", "keep": [[1000, 1800]]},
        {"type": "emsc", "poly_order": 2},
    )
    state = fit_preprocessing(spectra, wn, block)
    X, w = apply_preprocessing(spectra.copy(), wn, block, state)
    assert len(state["emsc_reference"]) == len(w) == X.shape[1]


def test_emsc_channel_mismatch_is_reported(spectra, wn):
    block = _steps({"type": "emsc", "poly_order": 2})
    with pytest.raises(ValueError, match="reference has"):
        apply_preprocessing(spectra.copy(), wn, block,
                            {"emsc_reference": np.zeros(L // 2)})


def test_als_subtracts_baseline(spectra, wn):
    block = _steps({"type": "als", "lam": 1e5, "p": 0.01, "n_iter": 5})
    X, _ = apply_preprocessing(spectra[:6].copy(), wn, block)
    assert np.isfinite(X).all()
    assert abs(np.median(X)) < abs(np.median(spectra[:6]))


def test_vector_norm_gives_unit_norm(spectra, wn):
    X, _ = apply_preprocessing(spectra.copy(), wn, _steps({"type": "vector_norm"}))
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-9)


# ── state persistence ────────────────────────────────────────────────────────

def test_state_round_trips_through_disk(spectra, wn, tmp_path):
    block = _steps(
        {"type": "crop", "keep": [[1000, 1800]]},
        {"type": "emsc", "poly_order": 2},
    )
    state = fit_preprocessing(spectra, wn, block)
    p = tmp_path / "preproc_state.npz"
    save_state(state, str(p))

    a, _ = apply_preprocessing(spectra.copy(), wn, block, state)
    b, _ = apply_preprocessing(spectra.copy(), wn, block, load_state(str(p)))
    np.testing.assert_allclose(a, b)


def test_load_state_of_missing_path_is_empty():
    assert load_state("") == {}


def test_full_pipeline_is_deterministic(spectra, wn):
    block = _steps(
        {"type": "crop", "keep": [[1000, 1800], [2800, 3000]]},
        {"type": "emsc", "poly_order": 2},
        {"type": "savgol", "window": 11, "polyorder": 3, "deriv": 1},
    )
    state = fit_preprocessing(spectra, wn, block)
    a, wa = apply_preprocessing(spectra.copy(), wn, block, state)
    b, wb = apply_preprocessing(spectra.copy(), wn, block, state)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(wa, wb)
    assert np.isfinite(a).all()


# ── the real domain configs ──────────────────────────────────────────────────

_DOMAIN_CFGS = sorted(
    (Path(__file__).parent.parent / "configs" / "domain").glob("*/data/default.yaml")
)


@pytest.mark.parametrize("cfg_path", _DOMAIN_CFGS, ids=lambda p: p.parts[-3])
def test_domain_config_preprocessing_is_valid(cfg_path):
    """Every step `type` in every shipped config must exist in the registry.

    resolve_steps validates even disabled steps, so this catches a typo in a
    block that is currently switched off — which is exactly the block nobody
    notices is broken until they enable it.
    """
    block = OmegaConf.load(cfg_path).get("preprocessing")
    resolve_steps(block)                                   # raises on unknown type
    resolve_steps(OmegaConf.create({**block, "enabled": True}))


@pytest.mark.parametrize("cfg_path", _DOMAIN_CFGS, ids=lambda p: p.parts[-3])
def test_domain_config_preprocessing_ships_disabled(cfg_path):
    """Shipped configs must be off by default, so adding the block does not
    silently change anyone's existing results."""
    block = OmegaConf.load(cfg_path).get("preprocessing")
    assert resolve_steps(block) == []


@pytest.mark.parametrize("cfg_path", _DOMAIN_CFGS, ids=lambda p: p.parts[-3])
def test_domain_config_merges_against_schema(cfg_path):
    """DataConfig is a structured config: a key it does not declare is
    rejected at Hydra compose time, so the block must be in the schema."""
    from src.config_schema import DataConfig

    merged = OmegaConf.merge(OmegaConf.structured(DataConfig), OmegaConf.load(cfg_path))
    assert "preprocessing" in merged


# ── biospectools-backed steps ────────────────────────────────────────────────
#
# EMSC and friends are NOT reimplemented here — they delegate to biospectools.
# These tests execute each one for real: the return contract of the underlying
# call is the part most likely to be wrong (interp2wns returns
# (spectra, wavenumbers), not the other way round).

needs_bs = pytest.mark.skipif(
    not _BIOSPECTOOLS_AVAILABLE, reason="biospectools not installed"
)


@needs_bs
@pytest.mark.parametrize("step", [
    {"type": "emsc", "poly_order": 2},
    {"type": "me_emsc", "n_components": 7, "max_iter": 5},
    {"type": "fringe_emsc", "fringe_wn_location": [2400, 2700]},
])
def test_emsc_family_preserves_shape(step, spectra, wn):
    block = _steps(step)
    X = spectra[:4]
    state = fit_preprocessing(X, wn, block)
    out, w = apply_preprocessing(X.copy(), wn, block, state)
    assert out.shape == X.shape
    np.testing.assert_allclose(w, wn)
    assert np.isfinite(out).all()


@needs_bs
def test_interpolate_returns_spectra_not_wavenumbers(spectra, wn):
    """Regression: interp2wns returns (spectra, wavenumbers). Unpacking it the
    other way silently replaced every batch with the axis array."""
    block = _steps({"type": "interpolate", "start": 1200, "stop": 2800, "num": 128})
    out, w = apply_preprocessing(spectra.copy(), wn, block)

    assert out.ndim == 2
    assert out.shape == (spectra.shape[0], 128)   # not (128,)
    assert w.shape == (128,)
    assert np.isfinite(out).all()


@needs_bs
def test_interpolate_recovers_original_on_identity_grid(spectra, wn):
    block = _steps({"type": "interpolate", "start": float(wn[0]),
                    "stop": float(wn[-1]), "num": len(wn)})
    out, w = apply_preprocessing(spectra.copy(), wn, block)
    np.testing.assert_allclose(w, wn)
    np.testing.assert_allclose(out, spectra, atol=1e-8)


def test_emsc_family_raises_without_biospectools(spectra, wn, monkeypatch):
    """With biospectools absent these must fail loudly, not fall back to a
    numerically different in-house version."""
    import src.data.preprocessing as pp

    monkeypatch.setattr(pp, "_BIOSPECTOOLS_AVAILABLE", False)
    for t in ("emsc", "me_emsc", "fringe_emsc", "interpolate"):
        with pytest.raises(ImportError, match="requires biospectools"):
            apply_preprocessing(
                spectra[:2].copy(), wn, _steps({"type": t, "start": 1, "stop": 2,
                                                "num": 3, "fringe_wn_location": [2400, 2700]}),
                {"emsc_reference": spectra.mean(0)},
            )
