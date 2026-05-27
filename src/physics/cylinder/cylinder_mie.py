import math
import os
import sys
from scipy.io import loadmat
from scipy.signal import hilbert
import scipy.integrate

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import numexpr as ne
from src.physics.cylinder.bessel import jv_range, yv_range


def cyl_q_ext_sca_na(wn_cm, ms, R, theta_na=0.3, theta_res=40):
    wn_cm = np.asarray(wn_cm)
    Ns, Nw = ms.shape

    ks = 2 * np.pi * wn_cm[None, :] * 1e-4
    xs = ks * R
    ys = ms * xs

    x_max = xs.real.max()
    nu_max = math.ceil(x_max + 5 * x_max**(1/3) + 2)

    xs_flat = xs.ravel()
    ys_flat = ys.ravel()

    Jxs_full = jv_range(nu_max + 1, xs_flat).T.reshape(nu_max + 2, Ns, Nw)
    Jys_full = jv_range(nu_max + 1, ys_flat).T.reshape(nu_max + 2, Ns, Nw)
    Nxs_full = yv_range(nu_max + 1, xs_flat).T.reshape(nu_max + 2, Ns, Nw)

    dJxs = np.empty((nu_max + 1, Ns, Nw), dtype=np.complex128)
    dJxs[1:] = 0.5 * (Jxs_full[:-2] - Jxs_full[2:])
    Jxs = Jxs_full[:-1]
    dJxs[0] = -Jxs[1]

    dJys = np.empty((nu_max + 1, Ns, Nw), dtype=np.complex128)
    dJys[1:] = 0.5 * (Jys_full[:-2] - Jys_full[2:])
    Jys = Jys_full[:-1]
    dJys[0] = -Jys[1]

    dNxs = np.empty((nu_max + 1, Ns, Nw), dtype=np.complex128)
    dNxs[1:] = 0.5 * (Nxs_full[:-2] - Nxs_full[2:])
    Nxs = Nxs_full[:-1]
    dNxs[0] = -Nxs[1]

    ms_b = ms[None, :, :]

    tan_betas = (ms_b * dJys * Jxs - Jys * dJxs) / (ms_b * dJys * Nxs - Jys * dNxs)
    tan_alfas = (dJys * Jxs - ms_b * Jys * dJxs) / (dJys * Nxs - ms_b * Jys * dNxs)


    b_coefs = tan_betas / (tan_betas - 1j)
    a_coefs = tan_alfas / (tan_alfas - 1j)

    b0 = b_coefs[0]; b_coefs = b_coefs[1:]
    a0 = a_coefs[0]; a_coefs = a_coefs[1:]

    inv_xs = 2 / xs
    QextI  = inv_xs * (b0.real + 2 * b_coefs.real.sum(axis=0))
    QscaI  = inv_xs * (b0 * b0.conj() + 2 * (b_coefs * b_coefs.conj()).sum(axis=0))
    QextII = inv_xs * (a0.real + 2 * a_coefs.real.sum(axis=0))
    QscaII = inv_xs * (a0 * a0.conj() + 2 * (a_coefs * a_coefs.conj()).sum(axis=0))

    thetas = np.linspace(0, theta_na, theta_res)
    Nn = b_coefs.shape[0]
    orders = np.arange(1, Nn + 1)[:, None, None, None]
    cos_term = np.cos(orders * thetas[None, None, None, :])

    T1 = b0[..., None] + 2 * (b_coefs[..., None] * cos_term).sum(axis=0)
    T2 = a0[..., None] + 2 * (a_coefs[..., None] * cos_term).sum(axis=0)

    dQdtheta = (2 / (np.pi * xs[..., None])) * (np.abs(T1)**2 + np.abs(T2)**2)
    qsca_NA = 0.5 * np.trapezoid(dQdtheta, thetas, axis=-1)

    QabsI  = QextI  - QscaI
    QabsII = QextII - QscaII
    return QextI, QabsI, QscaI, QextII, QabsII, QscaII, qsca_NA
