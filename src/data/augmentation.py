from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from scipy.signal import hilbert

# Restore removed aliases for legacy Cython extensions (removed in NumPy 1.24)
for _alias, _target in [
    ("complex", np.complex128),
    ("float",   np.float64),
    ("int",     np.int_),
    ("bool",    np.bool_),
]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

try:
    from src.physics.sphere.sphere_mie import q_ext_sca_na
    from src.physics.cylinder.cylinder_mie import cyl_q_ext_sca_na
    _MIE_AVAILABLE = True
except ImportError:
    q_ext_sca_na = None          # type: ignore[assignment]
    cyl_q_ext_sca_na = None      # type: ignore[assignment]
    _MIE_AVAILABLE = False


# ===========================================================================
# IR augmentations (numpy)
# ===========================================================================

def get_imagpart(pure_absorbance, wavelength, radius, factor=1):
    deff = np.pi / 2 * radius * factor
    imagpart = (pure_absorbance * np.log(10)) / (4 * np.pi * deff / wavelength)
    return imagpart


def get_nkk(imag_part, wavelengths: np.ndarray, pad_size=200):
    pad_last_axis = [(0, 0)] * imag_part.ndim
    pad_last_axis[-1] = (pad_size, pad_size)
    nkk = np.imag(hilbert(np.pad(imag_part, pad_last_axis, mode="edge")))
    nkk = nkk[..., pad_size:-pad_size]

    wls_increase = wavelengths[..., 0] < wavelengths[..., -1]
    if wls_increase:
        return nkk.copy()
    else:
        return -nkk


def add_cylindrical_scattering(spec, wns, r, n0, n_im, theta_na, h, scatt_coeff, theta_res=25):
    wls = 1e4 / wns[None]

    n_const = n0 + n_im * 1j
    n_i = get_imagpart(spec, wls, r, factor=h)
    n_r = get_nkk(n_i, wls.squeeze())
    ms = n_const + n_r + 1j * n_i

    QextI, QabsI, QscaI, QextII, QabsII, QscaII, qscaNA = cyl_q_ext_sca_na(
        wns,
        ms.conj(),
        r,
        theta_na=theta_na,
        theta_res=theta_res,
    )

    qabs       = (QabsI.real + QabsII.real) / 2
    qsca_total = (QscaI.real + QscaII.real) / 2
    qsca_lost  = qsca_total - qscaNA

    q_effective = scatt_coeff * qabs + qsca_lost
    q_norm = q_effective / np.abs(q_effective).max(axis=1, keepdims=True)
    return -np.log10(1 - 0.6 * q_norm)


def add_scattering(spec, wn, r, n0, n_im, theta_max, h, scatt_coeff, theta_res=15):
    wls = 10e3 / wn[None]

    n_const = n0 + n_im * 1j
    n_i = get_imagpart(spec, wls, r, factor=h)
    n_r = get_nkk(n_i, wls.squeeze())
    ms = n_const + n_r + 1j * n_i

    Qext, Qsca, QscaNA = q_ext_sca_na(
        ms,
        wls,
        r,
        theta_na=theta_max,
        theta_resolution=theta_res,
    )

    A = Qsca - QscaNA + scatt_coeff * (Qext - Qsca)
    return -np.log10(1 - 0.6 * A / np.abs(A).max(axis=1, keepdims=True))


def add_whitenoise(spectra, max_noise):
    spectra += np.random.normal(
        np.zeros(spectra.shape),
        np.random.uniform(0, max_noise, spectra.shape[0])[:, None],
        spectra.shape,
    )
    return spectra


def add_polynomial(spectra, wn, params):
    half_rng = np.abs(wn[0] - wn[-1]) / 2
    norm_wn = (wn - np.mean(wn)) / half_rng

    p0, p1, p2, p3 = (
        params[0][:, None],
        params[1][:, None],
        params[2][:, None],
        params[3][:, None],
    )
    return p1 * spectra + p0 + p2 * norm_wn + p3 * (norm_wn**2)


def add_co2(spectra, wn, co2_params):
    B, L = spectra.shape
    loc1, loc2, d_loc, height, d_height, width, d_width, N = co2_params
    N = int(N)

    centers = np.random.normal(np.random.uniform(loc1, loc2, N), d_loc, (B, N))[:, :, None]
    amps    = np.random.normal(height, d_height, (B, N))[:, :, None]
    widths  = np.abs(np.random.normal(width, d_width, (B, N)))[:, :, None]

    diff = (wn[None, None, :] - centers) ** 2
    w2 = widths**2
    lorentzian_peaks = (amps * w2) / (w2 + 4 * diff)
    return spectra + lorentzian_peaks.sum(axis=1)


# ===========================================================================
# Raman augmentations (torch)
# ===========================================================================

def augment_cosmic_rays(
    X: Tensor,
    spike_rate: float = 0.002,
    amplitude_range: tuple[float, float] = (3.0, 15.0),
    max_width: int = 3,
) -> Tensor:
    """
    Inject synthetic cosmic-ray spikes into spectra.

    Physics
    -------
    Cosmic rays hitting a CCD produce rare, very narrow,
    extremely high-intensity spikes affecting 1-3 channels.

    Fix
    ---
    Spike shape is now a *symmetric* triangle centred on `center`,
    matching real cosmic-ray PSFs (previously was an asymmetric ramp).

    Parameters
    ----------
    X               : (p,) or (n, p) tensor of spectra
    spike_rate      : expected fraction of channels hit (≈ 1 spike / 500 ch)
    amplitude_range : spike amplitude as multiples of mean spectrum intensity
    max_width       : spike width in channels (1-3 typical)
    """
    is_1d = X.ndim == 1
    if is_1d:
        X = X.unsqueeze(0)

    n, p   = X.shape
    X_aug  = X.clone()
    device = X.device

    # Expected spikes per spectrum — sample counts for whole batch at once
    expected      = p * spike_rate
    n_spikes_each = torch.poisson(
        torch.full((n,), expected)
    ).long()                                      # (n,)

    total_spikes = n_spikes_each.sum().item()
    if total_spikes == 0:
        return X_aug.squeeze(0) if is_1d else X_aug

    # Which spectrum does each spike belong to?
    spec_idx = torch.repeat_interleave(
        torch.arange(n), n_spikes_each
    )                                             # (total_spikes,)

    centers   = torch.randint(0, p, (total_spikes,))
    widths    = torch.randint(1, max_width + 1, (total_spikes,))
    mean_int  = torch.abs(X).mean(dim=1)         # (n,)
    amp_scale = torch.empty(total_spikes).uniform_(*amplitude_range)
    amplitudes = amp_scale * mean_int[spec_idx]  # (total_spikes,)

    # Build spike vectors and scatter-add into X_aug
    for k in range(total_spikes):
        si     = spec_idx[k].item()
        c      = centers[k].item()
        w      = widths[k].item()
        lo     = max(0, c - w // 2)
        hi     = min(p, c + w // 2 + 1)
        length = hi - lo

        if length <= 2:
            # A window this narrow (only reachable when a spike center
            # near either edge of the spectrum gets clipped asymmetrically
            # by lo=max(0,...)/hi=min(p,...), producing an even length as
            # small as 2) has no room to taper — a flat top is the closest
            # sensible shape. length==2 previously fell into the general
            # branch below, where linspace(0.0, 1.0, steps=1) degenerates
            # to a single-point [0.0], making both halves all-zero and
            # `shape / shape.max()` a 0/0 -> NaN.
            shape = torch.ones(length, device=device)
        else:
            half  = torch.linspace(0.0, 1.0, (length + 1) // 2, device=device)
            shape = torch.cat([half, half[:-1].flip(0)]) if length % 2 != 0 \
                    else torch.cat([half, half.flip(0)])
            shape = shape[:length]
            shape = shape / shape.max()

        X_aug[si, lo:hi] += amplitudes[k] * shape

    if is_1d:
        X_aug = X_aug.squeeze(0)
    return X_aug

def augment_shot_noise(
    X: Tensor,           # FIX: was annotated np.ndarray but used torch ops
    scale: float = 0.05,
) -> Tensor:
    """
    Poisson (shot) noise augmentation.

    Physics
    -------
    Each CCD channel counts photons; variance = mean count (Poisson).
    Approximated as signal-proportional Gaussian noise for counts > ~20.

    Fix
    ---
    Type annotation and all internal ops are now consistently *torch*.
    The original had an np.ndarray annotation but called torch.abs /
    torch.randn_like, causing silent conversion issues.

    Parameters
    ----------
    X     : (p,) or (n, p) *torch.Tensor* of spectra
    scale : noise amplitude as a fraction of local signal magnitude
    """
    is_1d = X.ndim == 1
    if is_1d:
        X = X.unsqueeze(0)

    noise_level = torch.rand(1, device=X.device).item() * scale
    noise_std   = noise_level * torch.abs(X)          # heteroscedastic std
    noise       = torch.randn_like(X) * noise_std

    X_aug = X + noise

    if is_1d:
        X_aug = X_aug.squeeze(0)
    return X_aug

# ── Paraffin peak parameters ──────────────────────────────────────────────────
# Each entry: (center_cm1, width_cm1, relative_amplitude)
# Derived from paraffin reference spectra in the fingerprint region.
# The 1462 band is dominant; others are scaled relative to it.
# Widths are typical for FFPE paraffin at 5 cm-1 resolution.
_PARAFFIN_BANDS = np.array([
    # Fingerprint region
    [1462.0, 12.0, 0.35],   # CH2 scissoring
    [1377.0,  8.0, 0.15],   # CH3 sym bending
    [1170.0,  9.0, 0.06],   # C-C stretch
    [1062.0,  8.0, 0.04],   # C-C stretch
    # CH stretch region — dominant bands
    [2920.0, 14.0, 1.00],   # CH2 asymmetric stretch  — strongest paraffin band
    [2850.0, 12.0, 0.85],   # CH2 symmetric stretch   — nearly as strong
    [2955.0,  9.0, 0.30],   # CH3 asymmetric stretch
    [2870.0,  8.0, 0.20],   # CH3 symmetric stretch
    # Combination bands (weak, but visible at good SNR)
    [1735.0,  8.0, 0.03],   # ester C=O overtone — very weak in pure paraffin
])  # shape (9, 3)

def _paraffin_spectrum(wn: np.ndarray, amplitude: float) -> np.ndarray:
    """
    Generate a paraffin absorption spectrum on wavenumber axis wn.
    Returns array of shape (len(wn),) scaled to peak amplitude.
    """
    spec = np.zeros(len(wn))
    for center, width, rel_amp in _PARAFFIN_BANDS:
        w2   = width ** 2
        diff = (wn - center) ** 2
        spec += rel_amp * (w2 / (w2 + 4 * diff))   # Lorentzian
    # Normalise so the dominant peak (1462) has unit height, then scale
    spec /= spec.max() + 1e-9
    return spec * amplitude


def _apply_paraffin(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    """
    Residual paraffin augmentation for FFPE FTIR tissue spectra.

    Adds realistic paraffin Lorentzian absorption bands to simulate
    incomplete deparaffinisation. The contamination level is drawn
    uniformly per spectrum from [amplitude_min, amplitude_max],
    expressed as absolute absorbance units (same scale as the spectrum).

    The 1360-1490 cm-1 paraffin removal window is deliberately NOT
    masked here — the augmentation simulates pre-removal contamination,
    consistent with applying augmentation before the preprocessing step.
    If augmentation is applied to already-preprocessed spectra, set
    amplitude_max low (~0.02) so only bleedthrough into neighbouring
    bands is simulated.

    Config keys
    -----------
    ratio         : float  — fraction of spectra to augment
    amplitude_min : float  — min paraffin peak amplitude (default 0.0)
    amplitude_max : float  — max paraffin peak amplitude (default 0.05)
    randomise_bands : bool — if True, jitter band centers ±3 cm-1 per
                             spectrum to simulate batch-to-batch variation
    """
    n_samples    = mask.sum()
    amp_min      = float(cfg.get("amplitude_min", 0.0))
    amp_max      = float(cfg.get("amplitude_max", 0.05))
    randomise    = bool(cfg.get("randomise_bands", True))

    amplitudes = np.random.uniform(amp_min, amp_max, n_samples)

    if not randomise:
        # One paraffin template, scaled per spectrum — fast path
        template = _paraffin_spectrum(wn, 1.0)
        s[mask] += amplitudes[:, None] * template[None, :]
    else:
        # Per-spectrum band jitter — more realistic across tissue batches
        for j, idx in enumerate(np.where(mask)[0]):
            jittered_bands = _PARAFFIN_BANDS.copy()
            jittered_bands[:, 0] += np.random.uniform(-3.0, 3.0, len(_PARAFFIN_BANDS))
            spec = np.zeros(len(wn))
            for center, width, rel_amp in jittered_bands:
                w2   = width ** 2
                diff = (wn - center) ** 2
                spec += rel_amp * (w2 / (w2 + 4 * diff))
            spec /= spec.max() + 1e-9
            s[idx] += amplitudes[j] * spec

    return s


class FluorescenceBackgroundAugmentor:
    """
    Simulates fluorescence baselines using a PCA basis fitted to *real*
    training spectra rather than random polynomial coefficients.

    Fix
    ---
    The original drew random polynomial coefficients, producing baselines
    that do not match the true fluorescence distribution of the dataset.
    At large N this mis-calibration hurts more than it helps.

    Usage
    -----
    # Once, before training:
    augmentor = FluorescenceBackgroundAugmentor(wavenumbers, n_components=5)
    augmentor.fit(X_train)          # fits PCA on baseline-only part of signal

    # Inside Dataset.__getitem__ or apply_augmentation:
    X_aug = augmentor(X_batch, amplitude_range=(0.0, 0.25))
    """

    def __init__(
        self,
        wavenumbers: Tensor,
        n_components: int = 5,
        poly_degree: int = 3,
    ) -> None:
        self.wavenumbers   = wavenumbers
        self.n_components  = n_components
        self.poly_degree   = poly_degree
        self._pca_basis: Tensor | None = None   # (n_components, p)
        self._fallback_basis: Tensor            # polynomial, used before fit()

        w_norm = (wavenumbers - wavenumbers.mean()) / wavenumbers.std()
        self._fallback_basis = torch.stack(
            [w_norm ** k for k in range(poly_degree + 1)], dim=1
        )  # (p, poly_degree+1)

    # ------------------------------------------------------------------
    def fit(self, X_train: Tensor) -> "FluorescenceBackgroundAugmentor":
        """
        Estimate a PCA basis from the *low-frequency* content of X_train.

        We approximate baseline by smoothing each spectrum with a wide
        median filter (or just keep the very low wavenumber trend),
        then run PCA.  No sklearn dependency: uses torch.linalg.svd.
        """
        # Rough baseline estimate: per-spectrum minimum + smoothed trend
        # Use a simple boxcar average as a cheap low-pass filter
        X_np = X_train.detach().cpu().float()
        window = max(1, X_np.shape[1] // 20)          # 5 % of spectrum width
        kernel  = torch.ones(1, 1, window) / window
        smooth  = torch.nn.functional.conv1d(
            X_np.unsqueeze(1),
            kernel,
            padding=window // 2,
        ).squeeze(1)
        # Trim to original length (conv1d padding can add 1 extra)
        smooth = smooth[:, : X_np.shape[1]]

        # Centre
        mean_baseline = smooth.mean(dim=0, keepdim=True)
        centred        = smooth - mean_baseline

        # SVD → keep top n_components
        _, _, Vt = torch.linalg.svd(centred, full_matrices=False)
        self._pca_basis = Vt[: self.n_components].to(X_train.device)  # (k, p)
        self._mean_baseline = mean_baseline.to(X_train.device)
        return self

    # ------------------------------------------------------------------
    def __call__(self, X, amplitude_range=(0.0, 0.25)):
        is_1d = X.ndim == 1
        if is_1d:
            X = X.unsqueeze(0)

        n, p = X.shape
        X_aug = X.clone()
        robust_intensity = torch.median(torch.abs(X), dim=1).values

        for i in range(n):
            amp = (
                torch.FloatTensor(1).uniform_(*amplitude_range).item()
                * robust_intensity[i].item()
            )

            if self._pca_basis is not None:
                coeffs     = torch.randn(self.n_components, device=X.device)
                background = coeffs @ self._pca_basis          # (p,)
            else:
                coeffs     = torch.randn(self.poly_degree + 1)
                coeffs[0]  = coeffs[0].abs() + 0.5
                background = (self._fallback_basis @ coeffs).to(X.device)

            background = background - background.min()
            if background.max() > 0:
                background = background / background.max()     # [0,1] — keep this
            # ← second `background = coeffs @ self._pca_basis` deleted

            if torch.isnan(background).any():
                continue                                       # skip this spectrum only
            X_aug[i] = X_aug[i] + amp * background

        if is_1d:
            X_aug = X_aug.squeeze(0)
        return X_aug



# ===========================================================================
# Wet-milk MIR augmentations (numpy) — Bentley instrument, physics-informed
#
# These augmentations target spectral variation that is specific to wet
# (liquid) milk MIR spectra acquired on Bentley-style FTIR milk analysers:
# fat-globule (homogenisation) scattering, sample-temperature drift,
# preservative chemistry, and dilution / low-solids effects.
# ===========================================================================

def _fat_globule_profile(
    wn: np.ndarray,
    slope_strength: float,
    bump_center: float,
    bump_width: float,
    bump_strength: float,
) -> np.ndarray:
    """
    Broad fat-globule scattering profile.

    Combines a smooth, monotonic wavenumber-dependent slope (a coarse
    stand-in for the general Mie/Rayleigh-type scattering trend) with a
    Gaussian elevation centred on the C-H stretch region, where fat-globule
    scattering and lipid absorption overlap most strongly.

    Returns a profile normalised to [0, 1] so callers can scale it to a
    desired absorbance elevation.
    """
    wn_range = wn.max() - wn.min()
    wn_norm = (wn - wn.min()) / (wn_range + 1e-9)
    bump = np.exp(-(wn - bump_center) ** 2 / (2 * bump_width ** 2))

    profile = slope_strength * wn_norm + bump_strength * bump
    profile -= profile.min()
    profile /= profile.max() + 1e-9
    return profile


def _apply_homogenizer_degradation(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    """
    Simulated homogenizer degradation for wet milk MIR spectra.

    Physics
    -------
    A worn or under-pressured homogenizer leaves larger, less uniform fat
    globules in the milk. Larger particles increase Mie-type scattering,
    which manifests as a broad, sloping elevation of the baseline that is
    most pronounced in the C-H stretch region (~2800-3000 cm-1), where the
    fat-globule scattering cross-section overlaps the lipid absorption
    bands most strongly.

    This is modelled as an additive profile combining a broad
    wavenumber-dependent slope with a Gaussian "bump" centred near the C-H
    stretch region. The bump centre is jittered per spectrum to mimic
    globule-size variation across samples/instruments.

    Config keys
    -----------
    ratio              : float — fraction of spectra to augment
    amplitude_min      : float — minimum added baseline elevation
                         (default 0.0)
    amplitude_max      : float — maximum added baseline elevation
                         (default 0.10)
    relative_amplitude : bool  — if True (default), amplitude_min/max are
                         fractions of each spectrum's peak absolute
                         intensity. If False, they are absolute absorbance
                         units.
    bump_center        : float — nominal centre of the C-H stretch
                         elevation, cm-1 (default 2900.0)
    bump_width         : float — Gaussian width (sigma) of the elevation,
                         cm-1 (default 100.0)
    bump_center_jitter : float — per-spectrum random jitter of bump_center,
                         cm-1 (default 50.0), simulating globule-size
                         variation across samples
    slope_strength     : float — relative weight of the broad sloping
                         component vs. the localised bump (default 0.3)
    bump_strength      : float — relative weight of the C-H stretch bump
                         (default 1.0)
    """
    n_samples = mask.sum()
    amp_min   = float(cfg.get("amplitude_min", 0.0))
    amp_max   = float(cfg.get("amplitude_max", 0.10))
    relative  = bool(cfg.get("relative_amplitude", True))

    bump_center   = float(cfg.get("bump_center", 2900.0))
    bump_width    = float(cfg.get("bump_width", 100.0))
    center_jitter = float(cfg.get("bump_center_jitter", 50.0))
    slope_strength = float(cfg.get("slope_strength", 0.3))
    bump_strength  = float(cfg.get("bump_strength", 1.0))

    amplitudes = np.random.uniform(amp_min, amp_max, n_samples)
    centers    = bump_center + np.random.uniform(-center_jitter, center_jitter, n_samples)

    for j, i in enumerate(np.where(mask)[0]):
        profile = _fat_globule_profile(wn, slope_strength, centers[j], bump_width, bump_strength)
        amp = amplitudes[j]
        if relative:
            amp = amp * np.abs(s[i]).max()
        s[i] += amp * profile

    return s


def _band_shift_profile(wn: np.ndarray, center: float, width: float, shift: float) -> np.ndarray:
    """
    First-derivative-of-Gaussian profile approximating the change in a
    spectrum caused by a small rigid shift of a Gaussian-shaped absorption
    band.

    For a band shape g(wn) = exp(-(wn - center)^2 / (2 * width^2)), shifting
    the band by `shift` cm-1 changes the spectrum to first order by
    `shift * dg/dwn` (sign chosen so that a positive `shift` corresponds to
    the band moving toward higher wavenumber). The returned profile is
    *not* unit-normalised: its peak-to-peak amplitude scales with
    `shift / width`, so small shifts on narrow bands produce small,
    band-localised perturbations.
    """
    gaussian   = np.exp(-(wn - center) ** 2 / (2 * width ** 2))
    derivative = (wn - center) / (width ** 2) * gaussian
    return shift * derivative


def _apply_temperature_perturbation(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    """
    Temperature perturbation augmentation for wet milk MIR spectra.

    Physics
    -------
    Small (+-1-2 C) sample-temperature variations change hydrogen bonding
    in water and shift the position of temperature-sensitive water
    absorption bands — most notably the H-O-H bending / amide II overlap
    region near 1640 cm-1, and (if within the instrument's range) the broad
    O-H stretch band near 3300-3400 cm-1.

    A small rigid shift of a band is, to first order, equivalent to adding
    a scaled derivative-of-Gaussian profile centred on that band (see
    `_band_shift_profile`). Each spectrum is assigned an independent random
    simulated temperature offset (degrees C), converted to a wavenumber
    shift via `shift_per_degree`, and the corresponding perturbation is
    added at each configured band.

    Config keys
    -----------
    ratio            : float — fraction of spectra to augment
    temp_delta_max   : float — max simulated temperature offset, +-deg C
                       (default 2.0)
    shift_per_degree : float — band shift in cm-1 per degree C
                       (default 0.3)
    band_center      : float — primary affected band centre, cm-1
                       (default 1640.0 — H-O-H bend / amide II overlap)
    band_width       : float — Gaussian width (sigma) of the primary band,
                       cm-1 (default 40.0)
    extra_bands      : list of [center, width] pairs, optional — additional
                       temperature-sensitive bands (e.g. the O-H stretch
                       near 3300 cm-1) to perturb in the same way
    amplitude_scale  : float — overall scaling of the perturbation relative
                       to each spectrum's peak intensity (default 1.0)
    """
    n_samples = mask.sum()
    temp_delta_max   = float(cfg.get("temp_delta_max", 2.0))
    shift_per_degree = float(cfg.get("shift_per_degree", 0.3))
    band_center      = float(cfg.get("band_center", 1640.0))
    band_width       = float(cfg.get("band_width", 40.0))
    extra_bands      = cfg.get("extra_bands", None)
    amplitude_scale  = float(cfg.get("amplitude_scale", 1.0))

    bands = [(band_center, band_width)]
    if extra_bands:
        bands += [(float(c), float(w)) for c, w in extra_bands]

    temp_deltas = np.random.uniform(-temp_delta_max, temp_delta_max, n_samples)

    for j, i in enumerate(np.where(mask)[0]):
        shift = temp_deltas[j] * shift_per_degree
        local_amp = np.abs(s[i]).max()
        for center, width in bands:
            s[i] += amplitude_scale * local_amp * _band_shift_profile(wn, center, width, shift)

    return s


# Characteristic IR-active bands for common milk-sample preservatives.
# Each entry: (center_cm1, width_cm1, relative_amplitude)
_PRESERVATIVE_BANDS: dict = {
    # Azidiol (sodium azide-based): strong, narrow azide (N=N=N) asymmetric
    # stretch, well clear of milk's major C=O / C-H / fingerprint bands.
    "azidiol": [
        (2050.0, 12.0, 1.00),   # N3- asymmetric stretch
    ],
    # Bronopol (2-bromo-2-nitropropane-1,3-diol): nitro-group stretches plus
    # a C-O stretch in the fingerprint region.
    "bronopol": [
        (1545.0, 14.0, 0.60),   # -NO2 asymmetric stretch
        (1370.0, 10.0, 0.40),   # -NO2 symmetric stretch
        (1040.0, 10.0, 0.25),   # C-O stretch
    ],
}


def _apply_preservative_effect(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    """
    Preservative-effect augmentation for wet milk MIR spectra.

    Physics
    -------
    Preservatives used during sample collection (azidiol / sodium azide,
    bronopol) are themselves IR-active and add characteristic absorption
    features on top of the milk spectrum, with intensity that varies with
    preservative concentration. Their bands also show small, protocol- and
    matrix-dependent positional shifts across collection sites/batches.

    This augmentation (1) adds small Lorentzian peaks at the preservative's
    characteristic band positions, with per-spectrum band-centre jitter and
    a random amplitude, and (2) applies a small derivative-shaped shift to
    those same bands (see `_band_shift_profile`) to simulate the
    protocol-dependent positional variation.

    Config keys
    -----------
    ratio          : float — fraction of spectra to augment
    preservative   : str   — "azidiol" (default) or "bronopol"
    amplitude_min  : float — min added peak amplitude, as a fraction of
                     each spectrum's peak intensity (default 0.0)
    amplitude_max  : float — max added peak amplitude, as a fraction of
                     each spectrum's peak intensity (default 0.03)
    band_jitter    : float — per-spectrum random jitter applied to each
                     band centre before adding the peak, cm-1 (default 3.0)
    shift_max      : float — max simulated band shift, +-cm-1
                     (default 1.0). Set to 0 to disable the shift component.
    shift_scale    : float — overall scaling of the shift perturbation
                     relative to each spectrum's peak intensity
                     (default 1.0)
    """
    n_samples    = mask.sum()
    preservative = cfg.get("preservative", "azidiol")
    amp_min      = float(cfg.get("amplitude_min", 0.0))
    amp_max      = float(cfg.get("amplitude_max", 0.03))
    jitter       = float(cfg.get("band_jitter", 3.0))
    shift_max    = float(cfg.get("shift_max", 1.0))
    shift_scale  = float(cfg.get("shift_scale", 1.0))

    bands = _PRESERVATIVE_BANDS.get(preservative)
    if bands is None:
        raise ValueError(
            f"Unknown preservative '{preservative}'. "
            f"Available options: {list(_PRESERVATIVE_BANDS)}"
        )

    amplitudes = np.random.uniform(amp_min, amp_max, n_samples)
    shifts     = np.random.uniform(-shift_max, shift_max, n_samples)

    for j, i in enumerate(np.where(mask)[0]):
        local_amp = np.abs(s[i]).max()

        # 1. add the preservative's characteristic absorption peak(s)
        spec = np.zeros(len(wn))
        for center, width, rel_amp in bands:
            jittered_center = center + np.random.uniform(-jitter, jitter)
            w2   = width ** 2
            diff = (wn - jittered_center) ** 2
            spec += rel_amp * (w2 / (w2 + 4 * diff))
        spec /= spec.max() + 1e-9
        s[i] += amplitudes[j] * local_amp * spec

        # 2. slight protocol-dependent shift of those same bands
        if shift_max > 0:
            for center, width, rel_amp in bands:
                s[i] += (
                    shift_scale * rel_amp * local_amp
                    * _band_shift_profile(wn, center, width, shifts[j])
                )

    return s


# Characteristic water absorption bands for the synthetic water-reference
# spectrum used by `_apply_dilution` when no measured reference is supplied.
# Each entry: (center_cm1, width_cm1, relative_amplitude)
_WATER_BANDS = np.array([
    [3340.0, 250.0, 1.00],  # O-H stretch — broad, dominant
    [2130.0, 120.0, 0.10],  # libration + bend combination band — weak
    [1640.0,  70.0, 0.55],  # H-O-H bending mode
])


def _water_reference_spectrum(wn: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    """
    Generate a synthetic pure-water MIR absorption spectrum on wavenumber
    axis `wn`, normalised to peak `amplitude`. Used as a fallback for
    `_apply_dilution` when no measured water reference is supplied.
    """
    spec = np.zeros(len(wn))
    for center, width, rel_amp in _WATER_BANDS:
        w2   = width ** 2
        diff = (wn - center) ** 2
        spec += rel_amp * (w2 / (w2 + 4 * diff))
    spec /= spec.max() + 1e-9
    return spec * amplitude


def _apply_dilution(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    """
    Dilution augmentation for wet milk MIR spectra.

    Physics
    -------
    Slight sample contamination with water/cleaning fluid, or measurement
    of abnormally low-solids milk, moves the spectrum linearly toward that
    of pure water:

        s_aug = (1 - alpha) * s + alpha * water_ref

    where `alpha` is the (small) effective water fraction. A real,
    instrument-matched pure-water spectrum can be supplied via
    `cfg.water_reference` (an array on the same `wn` axis as the milk
    spectra); otherwise a synthetic water spectrum (broad O-H stretch +
    H-O-H bend, see `_water_reference_spectrum`) is used.

    Config keys
    -----------
    ratio           : float — fraction of spectra to augment
    dilution_min    : float — min interpolation fraction toward water
                      (default 0.0)
    dilution_max    : float — max interpolation fraction toward water
                      (default 0.15)
    water_reference : array-like, optional — measured pure-water spectrum
                      on the same wavenumber axis as `wn`. If omitted, a
                      synthetic water spectrum is generated.
    scale_reference : bool — if True (default), the water reference is
                      rescaled to match each spectrum's peak intensity
                      before interpolation, so `alpha` remains meaningful
                      across datasets with different absorbance scales. Set
                      False to use the supplied/synthetic reference as-is
                      (appropriate if `water_reference` is already on the
                      same absorbance scale as the milk spectra).
    """
    dil_min = float(cfg.get("dilution_min", 0.0))
    dil_max = float(cfg.get("dilution_max", 0.15))
    scale_reference = bool(cfg.get("scale_reference", True))

    water_ref = cfg.get("water_reference", None)
    if water_ref is not None:
        water_ref = np.asarray(water_ref, dtype=float)
        if water_ref.shape != wn.shape:
            raise ValueError(
                f"water_reference shape {water_ref.shape} does not match "
                f"wn shape {wn.shape}"
            )
    else:
        water_ref = _water_reference_spectrum(wn, amplitude=1.0)

    s_masked = s[mask]
    n_samples = s_masked.shape[0]
    alphas = np.random.uniform(dil_min, dil_max, n_samples)[:, None]

    if scale_reference:
        scale = np.abs(s_masked).max(axis=1, keepdims=True)
        water_scaled = water_ref[None, :] * scale / (np.abs(water_ref).max() + 1e-9)
    else:
        water_scaled = np.broadcast_to(water_ref[None, :], s_masked.shape)

    s[mask] = (1 - alphas) * s_masked + alphas * water_scaled
    return s




class AugmentationPipeline:
    """
    Composes all three augmentations and applies them stochastically
    each time it is called — suitable for Dataset.__getitem__.

    Fix
    ---
    The original `apply_augmentation` returned a *static* tensor of
    n_copies augmented spectra computed once at dataset-construction time.
    That means the model sees the same augmented samples every epoch,
    giving no regularisation benefit at large N and only distorting the
    training distribution.

    By wrapping augmentation in a callable object used inside __getitem__,
    each epoch draws fresh noise, fresh baselines, and fresh cosmic rays.

    Parameters
    ----------
    fluorescence_augmentor : a FluorescenceBackgroundAugmentor instance.
                             Call .fit(X_train) before training starts.
    shot_noise_scale       : passed to augment_shot_noise
    fluor_amplitude        : passed to FluorescenceBackgroundAugmentor
    cosmic_rate            : passed to augment_cosmic_rays
    amplitude_range        : cosmic-ray amplitude range
    max_width              : cosmic-ray max width
    p_apply                : probability of applying augmentation at all
                             (set < 1.0 to randomly skip, useful at large N)
    """

    def __init__(
        self,
        wavenumbers: Tensor,
        fluorescence_augmentor: FluorescenceBackgroundAugmentor | None = None,
        shot_noise_scale: float = 0.075,
        fluor_amplitude: tuple[float, float] = (0.0, 0.25),
        cosmic_rate: float = 0.002,
        amplitude_range: tuple[float, float] = (3.0, 15.0),
        max_width: int = 3,
        p_apply: float = 1.0,
    ) -> None:
        self.wavenumbers   = wavenumbers
        self.fluor_aug     = fluorescence_augmentor or FluorescenceBackgroundAugmentor(wavenumbers)
        self.shot_scale    = shot_noise_scale
        self.fluor_amp     = fluor_amplitude
        self.cosmic_rate   = cosmic_rate
        self.amp_range     = amplitude_range
        self.max_width     = max_width
        self.p_apply       = p_apply

    def fit(self, X_train: Tensor) -> "AugmentationPipeline":
        """Fit fluorescence PCA basis. Call once before training."""
        self.fluor_aug.fit(X_train)
        return self

    def __call__(self, x: Tensor) -> Tensor:
        if self.p_apply < 1.0 and torch.rand(1).item() > self.p_apply:
            return x

        x_orig = x.clone()

        x = augment_cosmic_rays(x, spike_rate=self.cosmic_rate,
                                amplitude_range=self.amp_range,
                                max_width=self.max_width)
        x = self.fluor_aug(x, amplitude_range=self.fluor_amp)
        x = augment_shot_noise(x, scale=self.shot_scale)

        # If augmentation produced NaNs/Infs, fall back to original spectrum
        if torch.isnan(x).any() or torch.isinf(x).any():
            return x_orig

        return x


def make_schedule(
    n_train: int,
    base_shot_scale: float = 0.075,
    base_fluor_max: float = 0.25,
    base_cosmic_rate: float = 0.002,
    reference_n: int = 16,
) -> dict:
    """
    Return augmentation kwargs scaled down as training-set size grows.

    Rationale: at small N augmentation compensates for lack of real variance;
    at large N real variance already covers the distribution and aggressive
    augmentation adds domain-mismatched noise.

    Scaling: strength ∝ 1 / log2(n_train / reference_n + 1)

    Parameters
    ----------
    n_train     : number of training spectra
    reference_n : the N at which base strengths are 100 %

    Returns
    -------
    dict with keys: shot_noise_scale, fluor_amplitude, cosmic_rate
    """
    factor = 1.0 / np.log2(n_train / reference_n + 1)
    factor = float(np.clip(factor, 0.1, 1.0))
    return dict(
        shot_noise_scale=base_shot_scale   * factor,
        fluor_amplitude=(0.0, base_fluor_max * factor),
        cosmic_rate=base_cosmic_rate       * factor,
    )


# ===========================================================================
# Registry-compatible wrappers
# Signature: (s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray
# IR wrappers operate on numpy throughout.
# Raman wrappers convert to torch, apply, convert back.
# ===========================================================================

def _apply_mie_scattering(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    if not _MIE_AVAILABLE:
        raise ImportError(
            "Mie scattering requires compiled Cython extensions.\n"
            "Run:  cd src/physics/sphere  && python setup_sphere_mie.py build_ext --inplace\n"
            "      cd src/physics/cylinder && python setup_bessel.py    build_ext --inplace"
        )
    s_subset = s[mask].copy()
    s_subset -= s_subset.min(axis=1, keepdims=True)
    s_subset /= s_subset.max(axis=1, keepdims=True) + 1e-9

    n0s, rs, n_ims, hs, scs = (
        np.random.uniform(low, high, (s_subset.shape[0], 1))
        for low, high in [
            (cfg.n0_min, cfg.n0_max),
            (cfg.r_min, cfg.r_max),
            (cfg.n_imag_min, cfg.n_imag_max),
            (cfg.h_min, cfg.h_max),
            (cfg.scale_min, cfg.scale_max),
        ]
    )
    theta = np.random.uniform(cfg.theta_min, cfg.theta_max)

    if cfg.get("variant", "spherical") == "cylindrical":
        s[mask] = add_cylindrical_scattering(s_subset, wn, rs, n0s, n_ims, theta, hs, scs)
    else:
        s[mask] = add_scattering(s_subset, wn, rs, n0s, n_ims, theta, hs, scs)
    return s


def _apply_polynomial_baseline(s, mask, wn, cfg) -> np.ndarray:
    n_samples = mask.sum()

    # support both flat keys (new) and param_ranges list (legacy)
    if hasattr(cfg, "param_ranges"):
        ranges = cfg.param_ranges
        lows  = np.array([r[0] for r in ranges])
        highs = np.array([r[1] for r in ranges])
    else:
        lows  = np.array([cfg.p0_min, cfg.p1_min, cfg.p2_min, cfg.p3_min])
        highs = np.array([cfg.p0_max, cfg.p1_max, cfg.p2_max, cfg.p3_max])

    params = (np.random.rand(n_samples, 4) * (highs - lows) + lows).T
    s[mask] = add_polynomial(s[mask], wn, params)
    return s


def _apply_noise(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    n_noisy = mask.sum()
    noise_scales = np.random.rand(n_noisy, 1) * cfg.max_level
    s[mask] += np.random.normal(0, 1, s[mask].shape) * noise_scales
    return s


def _apply_co2_peaks(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    if hasattr(cfg, "params"):
        # legacy list format — other domains
        co2_params = cfg.params
    else:
        # flat keys — PCUK / sweepable
        co2_params = [
            cfg.loc1, cfg.loc2, cfg.d_loc,
            cfg.height, cfg.d_height,
            cfg.width, cfg.d_width,
            cfg.n_peaks,
        ]
    s[mask] = add_co2(s[mask], wn, co2_params)
    return s


def _apply_cosmic_rays(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    x = torch.from_numpy(s[mask]).float()
    amp_min = float(cfg.get("amplitude_min", cfg.get("amplitude_range", [3.0, 15.0])[0]))
    amp_max = float(cfg.get("amplitude_max", cfg.get("amplitude_range", [3.0, 15.0])[1]))
    x_aug = augment_cosmic_rays(
        x,
        spike_rate=cfg.get("spike_rate", 0.002),
        amplitude_range=(amp_min, amp_max),
        max_width=int(cfg.get("max_width", 3)),
    )
    s[mask] = x_aug.numpy()
    return s


def _apply_shot_noise(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    x = torch.from_numpy(s[mask]).float()
    x_aug = augment_shot_noise(x, scale=cfg.get("scale", 0.05))
    s[mask] = x_aug.numpy()
    return s

def _apply_wavenumber_shift(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    """
    Wavenumber axis shift augmentation for FTIR tissue spectra.

    Simulates instrument calibration drift between measurement sessions,
    slides, and the multi-year acquisition window of the PCUK dataset.
    Each spectrum receives an independent random shift drawn uniformly
    from [-shift_max, +shift_max] cm-1, applied via linear interpolation.

    Pixels shifted outside the original wavenumber range are filled by
    edge extrapolation (np.interp clamps to boundary values by default),
    which is physically reasonable since spectra change slowly near edges.

    Config keys
    -----------
    ratio     : float — fraction of spectra to augment
    shift_max : float — maximum shift in cm-1 (default 3.0)
                        realistic range for FTIR: 1-3 cm-1
                        keep below half the channel spacing to avoid
                        aliasing (channel spacing here is ~0.58 cm-1
                        for 1479 channels over 850 cm-1 range)
    per_batch : bool  — if False (default), each spectrum gets an
                        independent shift. If True, one shift is drawn
                        for the whole masked batch, simulating a
                        session-level calibration offset rather than
                        per-spectrum noise.
    """
    shift_max = float(cfg.get("shift_max", 3.0))
    per_batch = bool(cfg.get("per_batch", False))

    if per_batch:
        # one shift for entire batch — simulates session-level drift
        shift = np.random.uniform(-shift_max, shift_max)
        shifted_wn = wn + shift
        # vectorised interpolation for whole batch at once
        # np.interp is 1D only so we use a loop-free approach via searchsorted
        s[mask] = _interp_batch(s[mask], wn, shifted_wn)
    else:
        # independent shift per spectrum
        shifts = np.random.uniform(-shift_max, shift_max, mask.sum())
        s_masked = s[mask]
        for j in range(len(shifts)):
            shifted_wn = wn + shifts[j]
            s_masked[j] = np.interp(wn, shifted_wn, s_masked[j])
        s[mask] = s_masked

    return s


def _interp_batch(spectra: np.ndarray, wn: np.ndarray, shifted_wn: np.ndarray) -> np.ndarray:
    """
    Vectorised linear interpolation of a batch of spectra onto a shifted
    wavenumber axis. Equivalent to applying np.interp row-wise but faster
    for large batches via searchsorted.

    spectra    : (N, L) float32
    wn         : (L,)   original wavenumber axis
    shifted_wn : (L,)   shifted wavenumber axis (wn + shift)

    Returns (N, L) interpolated spectra.
    """
    # find indices in shifted_wn that bracket each point in wn
    idx = np.searchsorted(shifted_wn, wn, side="right") - 1
    idx = np.clip(idx, 0, len(wn) - 2)

    # linear interpolation weights
    wn_lo = shifted_wn[idx]
    wn_hi = shifted_wn[idx + 1]
    denom = wn_hi - wn_lo
    denom = np.where(np.abs(denom) < 1e-10, 1.0, denom)   # avoid div by zero
    t = np.clip((wn - wn_lo) / denom, 0.0, 1.0)           # (L,)

    # gather low and high values for each spectrum — (N, L)
    s_lo = spectra[:, idx]
    s_hi = spectra[:, idx + 1]

    return s_lo + t[None, :] * (s_hi - s_lo)


class _FluorescenceAugWrapper:
    def __init__(self) -> None:
        self._cache: dict[int, FluorescenceBackgroundAugmentor] = {}
        self._n_nan_fallbacks: dict[int, int] = {}

    def fit(self, wn: np.ndarray, X_train: np.ndarray, cfg) -> None:
        """Call once before training with the full training set."""
        key  = len(wn)
        wn_t = torch.from_numpy(wn).float()
        aug  = FluorescenceBackgroundAugmentor(
            wn_t,
            n_components=int(cfg.get("n_components", 5)),
            poly_degree=int(cfg.get("poly_degree", 3)),
        )
        aug.fit(torch.from_numpy(X_train).float())
        self._cache[key] = aug
        self._n_nan_fallbacks[key] = 0

    def __call__(self, s, mask, wn, cfg):
        key = len(wn)
        if key not in self._cache:
            raise RuntimeError(
                "FluorescenceAugWrapper.fit() must be called with the full "
                "training set before augmentation can run."
            )
        aug     = self._cache[key]
        amp_min = float(cfg.get("amplitude_min", cfg.get("amplitude_range", [0.0, 0.25])[0]))
        amp_max = float(cfg.get("amplitude_max", cfg.get("amplitude_range", [0.0, 0.25])[1]))
        amplitude_range = (amp_min, amp_max)
        s_t             = torch.from_numpy(s[mask]).float()
        x_aug           = aug(s_t, amplitude_range=amplitude_range)

        # Track NaN fallbacks rather than silently ignoring
        nan_mask = torch.isnan(x_aug).any(dim=-1)
        if nan_mask.any():
            self._n_nan_fallbacks[key] += nan_mask.sum().item()
            x_aug[nan_mask] = s_t[nan_mask]   # fall back per-spectrum, not whole batch

        s[mask] = x_aug.numpy()
        return s

    def nan_fallback_count(self, wn_len: int) -> int:
        return self._n_nan_fallbacks.get(wn_len, 0)


_apply_fluorescence = _FluorescenceAugWrapper()


AUG_REGISTRY: dict = {
    # IR
    "mie_scattering":      _apply_mie_scattering,
    "polynomial_baseline": _apply_polynomial_baseline,
    "noise":               _apply_noise,
    "co2_peaks":           _apply_co2_peaks,
    "paraffin":            _apply_paraffin,
    "wavenumber_shift":    _apply_wavenumber_shift,
    # Wet milk MIR
    "homogenizer_degradation":  _apply_homogenizer_degradation,
    "temperature_perturbation": _apply_temperature_perturbation,
    "preservative_effect":      _apply_preservative_effect,
    "dilution":                 _apply_dilution,
    # Raman
    "cosmic_rays":         _apply_cosmic_rays,
    "shot_noise":          _apply_shot_noise,
    "fluorescence":        _apply_fluorescence,
}
