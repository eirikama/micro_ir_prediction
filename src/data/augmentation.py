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

from src.physics.sphere.sphere_mie import q_ext_sca_na
from src.physics.cylinder.cylinder_mie import cyl_q_ext_sca_na


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

        if length == 1:
            shape = torch.ones(1, device=device)
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


def _apply_polynomial_baseline(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    n_samples = mask.sum()
    ranges = cfg.param_ranges
    lows   = np.array([r[0] for r in ranges])
    highs  = np.array([r[1] for r in ranges])
    params = (np.random.rand(n_samples, 4) * (highs - lows) + lows).T
    s[mask] = add_polynomial(s[mask], wn, params)
    return s


def _apply_noise(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    n_noisy = mask.sum()
    noise_scales = np.random.rand(n_noisy, 1) * cfg.max_level
    s[mask] += np.random.normal(0, 1, s[mask].shape) * noise_scales
    return s


def _apply_co2_peaks(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    s[mask] = add_co2(s[mask], wn, cfg.params)
    return s


def _apply_cosmic_rays(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    x = torch.from_numpy(s[mask]).float()
    x_aug = augment_cosmic_rays(
        x,
        spike_rate=cfg.get("spike_rate", 0.002),
        amplitude_range=tuple(cfg.get("amplitude_range", [3.0, 15.0])),
        max_width=int(cfg.get("max_width", 3)),
    )
    s[mask] = x_aug.numpy()
    return s


def _apply_shot_noise(s: np.ndarray, mask: np.ndarray, wn: np.ndarray, cfg) -> np.ndarray:
    x = torch.from_numpy(s[mask]).float()
    x_aug = augment_shot_noise(x, scale=cfg.get("scale", 0.05))
    s[mask] = x_aug.numpy()
    return s


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
        aug             = self._cache[key]
        amplitude_range = tuple(cfg.get("amplitude_range", [0.0, 0.25]))
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
    # Raman
    "cosmic_rays":         _apply_cosmic_rays,
    "shot_noise":          _apply_shot_noise,
    "fluorescence":        _apply_fluorescence,
}
