import numpy as np

from biospectools_private.physics import cmie
from biospectools_private.physics import misc


def add_scattering(
    spec, wn, r, n0, n_im, theta_max, h, scatt_coeff, theta_res=15
):
    
    n_const = n0 + n_im * 1j
    wls = 10e+3 / wn[None]
    
    n_i = misc.get_imagpart(spec, wls, r, factor=h)
    n_r = misc.get_nkk(n_i, wls.squeeze())   

    ms = n_const + n_r + 1j * n_i

    Qext, Qsca, QscaNA = cmie.q_ext_sca_na(
        ms,
        wls,
        r,
        theta_na=theta_max,
        theta_resolution=theta_res,
    )

    A = Qsca - QscaNA + scatt_coeff * (Qext - Qsca)
    A = -np.log10((1 - 0.6 * A / np.abs(A).max(axis=1, keepdims=True)))
        
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
    
    p0, p1, p2, p3 = params[0][:, None], params[1][:, None], params[2][:, None], params[3][:, None]
    
    return p1 * spectra + p0 + p2 * norm_wn + p3 * (norm_wn ** 2)

    
def add_co2(spectra, wn, co2_params):
    
    B, L = spectra.shape
    loc1, loc2, d_loc, height, d_height, width, d_width, N = co2_params
    N = int(N)

    centers = np.random.normal(np.random.uniform(loc1, loc2, N), d_loc, (B, N))[:, :, None]
    amps = np.random.normal(height, d_height, (B, N))[:, :, None]
    widths = np.abs(np.random.normal(width, d_width, (B, N)))[:, :, None]

    diff = (wn[None, None, :] - centers) ** 2
    w2 = widths ** 2
    lorentzian_peaks = (amps * w2) / (w2 + 4 * diff)
    
    return spectra + lorentzian_peaks.sum(axis=1)








