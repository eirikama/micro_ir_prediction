import numpy as np
from scipy.signal import hilbert

from src.physics.mie import q_ext_sca_na


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


def add_scattering(spec, wn, r, n0, n_im, theta_max, h, scatt_coeff, theta_res=15):
    n_const = n0 + n_im * 1j
    wls = 10e3 / wn[None]

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
    A = -np.log10(1 - 0.6 * A / np.abs(A).max(axis=1, keepdims=True))

    return A


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
    amps = np.random.normal(height, d_height, (B, N))[:, :, None]
    widths = np.abs(np.random.normal(width, d_width, (B, N)))[:, :, None]

    diff = (wn[None, None, :] - centers) ** 2
    w2 = widths**2
    lorentzian_peaks = (amps * w2) / (w2 + 4 * diff)

    return spectra + lorentzian_peaks.sum(axis=1)
