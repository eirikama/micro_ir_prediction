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


# ── wavenumber shift: axis-order regression ───────────────────────────────────
#
# np.interp and np.searchsorted (used by _interp_batch) both require an
# ASCENDING x-axis and neither raises on a descending one — they silently
# return garbage. FTIR stores are commonly written 4000->400 cm-1, and on such
# an axis this augmentation used to annihilate the spectrum entirely (a
# unit-height Gaussian came back with max 0.0). The `wavenumbers` fixture is
# ascending, which is exactly why that went unnoticed, so these tests build
# their own axes in both orders.

def _peak_spectrum(w: np.ndarray) -> np.ndarray:
    return (
        np.exp(-((w - 1650.0) ** 2) / (2 * 30.0 ** 2))
        + 0.6 * np.exp(-((w - 2900.0) ** 2) / (2 * 40.0 ** 2))
    )


@pytest.mark.parametrize("descending", [False, True], ids=["ascending", "descending"])
@pytest.mark.parametrize("per_batch", [False, True])
def test_wavenumber_shift_preserves_peaks_on_either_axis_order(descending, per_batch):
    from src.data.augmentation import _apply_wavenumber_shift
    from omegaconf import OmegaConf

    axis = np.linspace(400.0, 4000.0, 400)
    if descending:
        axis = axis[::-1].copy()

    s = _peak_spectrum(axis)[None, :].copy()
    cfg = OmegaConf.create({"shift_max": 3.0, "per_batch": per_batch})

    np.random.seed(0)
    out = _apply_wavenumber_shift(s.copy(), np.ones(1, dtype=bool), axis, cfg)

    assert np.isfinite(out).all()
    # A few-cm-1 shift must leave the band structure essentially intact.
    assert out[0].max() == pytest.approx(s[0].max(), rel=0.02)
    assert out[0].argmax() == s[0].argmax()
    assert np.corrcoef(out[0], s[0])[0, 1] > 0.99


@pytest.mark.parametrize("per_batch", [False, True])
def test_wavenumber_shift_is_axis_order_invariant(per_batch):
    """Same seed + mirrored axis must give the mirrored result exactly."""
    from src.data.augmentation import _apply_wavenumber_shift
    from omegaconf import OmegaConf

    wn_asc = np.linspace(400.0, 4000.0, 400)
    wn_desc = wn_asc[::-1].copy()
    cfg = OmegaConf.create({"shift_max": 5.0, "per_batch": per_batch})
    mask = np.ones(1, dtype=bool)

    np.random.seed(7)
    out_asc = _apply_wavenumber_shift(
        _peak_spectrum(wn_asc)[None, :].copy(), mask, wn_asc, cfg
    )
    np.random.seed(7)
    out_desc = _apply_wavenumber_shift(
        _peak_spectrum(wn_desc)[None, :].copy(), mask, wn_desc, cfg
    )

    np.testing.assert_allclose(out_asc[0], out_desc[0][::-1], atol=1e-12)


# ── shot noise ────────────────────────────────────────────────────────────────

def test_shot_noise_level_varies_between_spectra():
    """The noise level must be drawn per spectrum, not once per call.

    It was previously a single scalar shared by the whole batch, which
    collapsed augmentation diversity (realised spread was only ~1.2x).
    """
    from src.data.augmentation import augment_shot_noise

    torch.manual_seed(0)
    X = torch.ones(64, 300)
    realised = (augment_shot_noise(X, scale=0.05) - X).std(dim=1)
    assert realised.max() / realised.min() > 3.0


def test_shot_noise_read_noise_reaches_zero_signal_regions():
    """Without a floor, near-zero baseline is left perfectly smooth, which is
    unphysical and an easy 'smooth => background' shortcut for the model."""
    from src.data.augmentation import augment_shot_noise

    torch.manual_seed(0)
    w = torch.linspace(600, 1800, 400)
    sig = torch.exp(-((w - 1000) ** 2) / (2 * 15.0 ** 2))
    X = sig.repeat(128, 1)
    baseline = sig < 0.01

    without = (augment_shot_noise(X.clone(), scale=0.05, read_noise=0.0) - X)[:, baseline].std()
    with_floor = (augment_shot_noise(X.clone(), scale=0.05, read_noise=0.02) - X)[:, baseline].std()

    assert without < 1e-3                 # proportional-only: baseline is silent
    assert with_floor > 20 * without      # floor actually reaches the baseline


def test_shot_noise_read_noise_defaults_to_previous_behaviour():
    from src.data.augmentation import augment_shot_noise

    X = torch.rand(16, 200) + 0.5
    torch.manual_seed(3)
    default = augment_shot_noise(X.clone(), scale=0.05)
    torch.manual_seed(3)
    explicit = augment_shot_noise(X.clone(), scale=0.05, read_noise=0.0)
    torch.testing.assert_close(default, explicit)


# ── fluorescence background: ALS baseline + uncontaminated PCA basis ──────────
#
# The basis used to be fitted to a 5%-boxcar "baseline", which smears Raman
# peaks rather than removing them. Any peak whose height varied between
# spectra therefore leaked into the components (measured ~7x energy
# enrichment in peak regions for PC2+), so the augmentation added scaled
# copies of the analytical signal back onto the spectra.

_PEAK_CENTERS = (1003.0, 1450.0, 1650.0)


def _raman_like(n: int, p: int = 400, seed: int = 0):
    """Synthetic Raman: multi-mode fluorescence background + peaks whose
    heights vary between spectra (i.e. the class-discriminative part)."""
    rng = np.random.default_rng(seed)
    wn = np.linspace(600.0, 1800.0, p)
    x = (wn - wn.min()) / (wn.max() - wn.min())
    out = []
    for _ in range(n):
        bg = (1.0 + rng.random()) * np.exp(
            -((x - rng.uniform(0.3, 0.7)) ** 2) / (2 * 0.25 ** 2)
        )
        bg = bg + rng.uniform(-0.5, 0.5) * x + rng.uniform(0.0, 1.0) * np.exp(-3 * x)
        pk = sum(
            rng.uniform(0.3, 1.5) * np.exp(-((wn - c) ** 2) / (2 * 8.0 ** 2))
            for c in _PEAK_CENTERS
        )
        out.append(bg + pk)
    return wn, np.asarray(out)


def _peak_region_mask(wn: np.ndarray, half_width: float = 25.0) -> np.ndarray:
    m = np.zeros(len(wn), dtype=bool)
    for c in _PEAK_CENTERS:
        m |= np.abs(wn - c) < half_width
    return m


def test_als_baseline_sits_under_peaks():
    """A baseline must pass *below* peaks, leaving a large residual there,
    while tracking the background closely elsewhere.

    The assertion is on the ratio rather than an absolute off-peak residual:
    ALS with p << 0.5 deliberately fits slightly *under* the data, so a small
    positive residual everywhere is correct behaviour, not leakage.
    """
    from src.data.augmentation import _als_baseline

    wn, Y = _raman_like(8, seed=1)
    base = _als_baseline(Y)
    resid = Y - base
    peaks = _peak_region_mask(wn)

    assert base.shape == Y.shape
    assert np.isfinite(base).all()
    assert resid[:, peaks].mean() > 0.05        # peaks survive above baseline
    # background is tracked far more closely than peaks are
    assert resid[:, peaks].mean() > 3.0 * abs(resid[:, ~peaks].mean())


def test_als_baseline_leaks_less_peak_shape_than_boxcar():
    """The boxcar this replaced smears peaks into its own output, which is
    what contaminated the fitted PCA basis.

    Leakage is measured as correlation between the baseline estimate and the
    true peak profile. (Residual *means* are not usable here: a boxcar is an
    average, so its residuals cancel to ~0 by construction regardless of how
    much peak shape it absorbed.)
    """
    from src.data.augmentation import _als_baseline

    wn, Y = _raman_like(8, seed=1)
    true_peaks = np.zeros(len(wn))
    for c in _PEAK_CENTERS:
        true_peaks += np.exp(-((wn - c) ** 2) / (2 * 8.0 ** 2))

    def peak_leakage(baselines: np.ndarray) -> float:
        centred = baselines - baselines.mean(axis=1, keepdims=True)
        return abs(np.mean([np.corrcoef(r, true_peaks)[0, 1] for r in centred]))

    als = _als_baseline(Y)

    win = Y.shape[1] // 20
    k = np.ones(win) / win
    boxcar = np.stack([np.convolve(y, k, mode="same") for y in Y])

    assert peak_leakage(als) < 0.05
    assert peak_leakage(als) < peak_leakage(boxcar)


def test_als_baseline_is_smoother_than_input():
    from src.data.augmentation import _als_baseline

    _, Y = _raman_like(4, seed=2)
    base = _als_baseline(Y)
    rough_in = np.abs(np.diff(Y, n=2, axis=1)).mean()
    rough_out = np.abs(np.diff(base, n=2, axis=1)).mean()
    assert rough_out < 0.1 * rough_in


def test_als_baseline_accepts_1d():
    from src.data.augmentation import _als_baseline

    _, Y = _raman_like(1, seed=3)
    assert _als_baseline(Y[0]).shape == Y[0].shape


def test_fluorescence_basis_is_not_peak_contaminated():
    """Regression: components must describe background, not Raman peaks.

    With the old boxcar fit this measured ~7x enrichment for PC2+.
    """
    from src.data.augmentation import FluorescenceBackgroundAugmentor

    wn, X = _raman_like(256, seed=4)
    aug = FluorescenceBackgroundAugmentor(torch.from_numpy(wn).float(), n_components=5)
    aug.fit(torch.from_numpy(X).float())

    peaks = _peak_region_mask(wn)
    width = peaks.mean()
    basis = aug._pca_basis.numpy()

    for k, b in enumerate(basis):
        enrichment = ((b[peaks] ** 2).sum() / (b ** 2).sum()) / width
        assert enrichment < 3.0, (
            f"PC{k + 1} carries {enrichment:.1f}x its share of energy in peak "
            "regions — the basis is picking up analytical signal, not background"
        )


def test_fluorescence_retains_component_scales():
    """Singular values must be kept, and ordered, so sampled coefficients
    reproduce the fitted distribution instead of weighting all components
    equally."""
    from src.data.augmentation import FluorescenceBackgroundAugmentor

    wn, X = _raman_like(128, seed=5)
    aug = FluorescenceBackgroundAugmentor(torch.from_numpy(wn).float(), n_components=4)
    aug.fit(torch.from_numpy(X).float())

    assert aug._pca_scale is not None
    scales = aug._pca_scale.numpy()
    assert len(scales) == 4
    assert (scales > 0).all()
    assert np.all(np.diff(scales) <= 0), "component scales must be descending"


def test_fluorescence_call_batch_is_finite_and_additive():
    from src.data.augmentation import FluorescenceBackgroundAugmentor

    wn, X = _raman_like(64, seed=6)
    aug = FluorescenceBackgroundAugmentor(torch.from_numpy(wn).float(), n_components=5)
    aug.fit(torch.from_numpy(X).float())

    Xt = torch.from_numpy(X).float()
    out = aug(Xt, amplitude_range=(0.0, 0.25))

    assert out.shape == Xt.shape
    assert torch.isfinite(out).all()
    # fluorescence only ever adds signal
    assert (out - Xt).min() >= -1e-5


def test_fluorescence_call_accepts_1d():
    from src.data.augmentation import FluorescenceBackgroundAugmentor

    wn, X = _raman_like(8, seed=7)
    aug = FluorescenceBackgroundAugmentor(torch.from_numpy(wn).float(), n_components=3)
    aug.fit(torch.from_numpy(X).float())
    assert aug(torch.from_numpy(X[0]).float()).shape == (X.shape[1],)


def test_fluorescence_works_before_fit_via_polynomial_fallback():
    """Un-fitted augmentor must still run (polynomial fallback basis)."""
    from src.data.augmentation import FluorescenceBackgroundAugmentor

    wn, X = _raman_like(8, seed=8)
    aug = FluorescenceBackgroundAugmentor(torch.from_numpy(wn).float())
    out = aug(torch.from_numpy(X).float(), amplitude_range=(0.0, 0.1))
    assert out.shape == X.shape
    assert torch.isfinite(out).all()


def test_fluorescence_fit_subsamples_large_training_sets():
    """fit() must cap how many spectra go through ALS — it is ~2.5 ms each."""
    from src.data.augmentation import FluorescenceBackgroundAugmentor

    wn, X = _raman_like(120, seed=9)
    X = np.repeat(X, 6, axis=0)          # 720 spectra
    aug = FluorescenceBackgroundAugmentor(
        torch.from_numpy(wn).float(), n_components=3, max_fit_spectra=50
    )
    aug.fit(torch.from_numpy(X).float())
    assert aug._pca_basis.shape == (3, X.shape[1])


def test_fluorescence_fit_pool_size_overrides_internal_cap():
    """When the caller has drawn a fixed pool of K spectra, K must govern —
    otherwise the augmentor's internal performance cap (max_fit_spectra,
    default 2000) would silently truncate the pool it was handed, and the
    basis would stop being independent of spectra_per_class."""
    from omegaconf import OmegaConf
    from src.data.augmentation import _FluorescenceAugWrapper

    wn, X = _raman_like(40, seed=11)
    wrapper = _FluorescenceAugWrapper()

    cfg = OmegaConf.create({"n_components": 2, "fit_pool_size": 4000})
    wrapper.fit(wn.astype(np.float64), X.astype(np.float32), cfg)
    assert wrapper._cache[len(wn)].max_fit_spectra == 4000

    # falls back to the plain cap when no pool was drawn
    cfg2 = OmegaConf.create({"n_components": 2, "max_fit_spectra": 750})
    wrapper.fit(wn.astype(np.float64), X.astype(np.float32), cfg2)
    assert wrapper._cache[len(wn)].max_fit_spectra == 750
