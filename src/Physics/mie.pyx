# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from typing import Union as U
import numpy as np
cimport numpy as np
np.import_array()
import cython
from cython.parallel import prange
from libc.math cimport sin, cos, round, pow, M_PI
from libc.stdlib cimport malloc, free

cdef extern from "<complex>" namespace "std" nogil:
    double complex csin "sin"(double complex z)
    double complex ccos "cos"(double complex z)


# ── dependency 1: Fast recurrence for angular functions ───────────────────────
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fast_pi_tao(thetas, int nmax):
    """
    Computes pi and tao using the standard Legendre recurrence relation.
    Returns arrays of shape (nmax, n_thetas) for contiguous memory access
    in the inner loops later on.
    """
    cdef int n_thetas = thetas.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] pi_out = np.empty((nmax, n_thetas), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] tao_out = np.empty((nmax, n_thetas), dtype=np.float64)

    cdef double[::1] cos_t = np.cos(thetas)
    cdef double[:, ::1] pi = pi_out
    cdef double[:, ::1] tao = tao_out

    cdef int i, n
    cdef double mu, p_n_minus_2, p_n_minus_1, p_n

    for i in range(n_thetas):
        mu = cos_t[i]

        # Base case: n = 1
        p_n_minus_1 = 1.0     # P_0
        p_n = mu              # P_1
        pi[0, i] = p_n
        tao[0, i] = mu * p_n - 2.0 * p_n_minus_1

        # Recurrence: n >= 2
        for n in range(2, nmax + 1):
            p_n_minus_2 = p_n_minus_1
            p_n_minus_1 = p_n
            # P_n(x) = [(2n-1)x P_{n-1}(x) - (n-1)P_{n-2}(x)] / n
            p_n = ((2.0 * n - 1.0) * mu * p_n_minus_1 - (n - 1.0) * p_n_minus_2) / n

            pi[n - 1, i] = p_n
            tao[n - 1, i] = n * mu * p_n - (n + 1.0) * p_n_minus_1

    return pi_out, tao_out


# ── dependency 2: scalar kernel ───────────────────────────────────────────────
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef (double, double, double) q_ext_sca_na_scalar(
        double complex m, double lambd, double r,
        double[:, ::1] pi, double[:, ::1] tao,
        double[::1] thetas, double[::1] sin_thetas) nogil:

    cdef double dtheta     = thetas[1] - thetas[0]
    cdef double x          = 2 * M_PI * r / lambd
    cdef double complex z  = m * x
    cdef double SIN_X      = sin(x)
    cdef double COS_X      = cos(x)
    cdef double ONE_OVER_X = 1 / x
    cdef double complex ONE_OVER_Z = 1 / z

    cdef int nmax  = int(round(2 + x + 4 * pow(x, 1. / 3)) + 1)
    cdef int nsca  = pi.shape[1]  # Transposed shape now!
    nmax = min(nmax, pi.shape[0])

    # 1 Single Malloc block instead of 4 (Massive OpenMP scaling boost)
    # Sizes: double(nmax+1), d_complex(nmax+1), d_complex(nsca), d_complex(nsca)
    # double complex equals 2 doubles in memory size.
    cdef int num_doubles = (nmax + 1) + 2 * (nmax + 1) + 4 * nsca
    cdef double* buffer = <double*>malloc(sizeof(double) * num_doubles)

    cdef double* bx         = buffer
    cdef double complex* bz = <double complex*>(&buffer[(nmax + 1)])
    cdef double complex* s1 = <double complex*>(&buffer[(nmax + 1) + 2*(nmax + 1)])
    cdef double complex* s2 = <double complex*>(&buffer[(nmax + 1) + 2*(nmax + 1) + 2*nsca])

    cdef double[3] yx
    cdef double complex[2] hx
    cdef int i


    # spherical jn — reverse recurrence
    bx[nmax] = 0;   bz[nmax] = 0
    bx[nmax - 1] = 1e-100;   bz[nmax - 1] = 1e-100
    cdef int mult = 0
    for mult in range(2*(nmax + 50 - 2) + 3, 2*(nmax - 2) + 3, -2):
        bx[nmax-1], bx[nmax] = mult * bx[nmax-1] * ONE_OVER_X - bx[nmax], bx[nmax-1]
        bz[nmax-1], bz[nmax] = mult * bz[nmax-1] * ONE_OVER_Z - bz[nmax], bz[nmax-1]
    for i in range(nmax - 2, 1, -1):
        mult -= 2
        bx[i] = mult * bx[i+1] * ONE_OVER_X - bx[i+2]
        bz[i] = mult * bz[i+1] * ONE_OVER_Z - bz[i+2]

    cdef double bx1_approx         = 5 * bx[2] * ONE_OVER_X - bx[3]
    cdef double complex bz1_approx = 5 * bz[2] * ONE_OVER_Z - bz[3]

    bx[0] = SIN_X * ONE_OVER_X
    bz[0] = csin(z) * ONE_OVER_Z
    bx[1] = (bx[0] - COS_X) * ONE_OVER_X
    bz[1] = (bz[0] - ccos(z)) * ONE_OVER_Z

    cdef double alphax         = bx[1] / bx1_approx
    cdef double complex alphaz = bz[1] / bz1_approx
    for i in range(2, nmax):
        bx[i] *= alphax
        bz[i] *= alphaz

    yx[0] = -COS_X * ONE_OVER_X
    hx[0] = bx[0] + yx[0] * 1j

    cdef double complex m2 = m * m  # Using m*m instead of m**2
    cdef double ax
    cdef double complex az, ahx, an, bn, an_factor, bn_factor
    cdef int cn, offset
    cdef double qext = 0
    cdef double qsca_total = 0

    # Pointer arithmetic for contiguous inner loop access
    cdef double* pi_ptr = &pi[0, 0]
    cdef double* tao_ptr = &tao[0, 0]

    # first iteration (n=1)
    yx[1]  = (yx[0] - SIN_X) * ONE_OVER_X
    hx[1]  = bx[1] + 1j * yx[1]
    ax     = x * bx[0] - bx[1]
    az     = z * bz[0] - bz[1]
    ahx    = x * hx[0] - hx[1]
    an     = (m2 * bz[1] * ax - bx[1] * az) / (m2 * bz[1] * ahx - hx[1] * az)
    bn     = (bz[1] * ax - bx[1] * az) / (bz[1] * ahx - hx[1] * az)
    cn     = 3
    qext       += cn * (an.real + bn.real)

    # Replaced an.conjugate() overhead
    qsca_total += cn * ((an.real*an.real + an.imag*an.imag) + (bn.real*bn.real + bn.imag*bn.imag))

    an_factor = cn * 0.5 * an
    bn_factor = cn * 0.5 * bn

    for i in range(nsca):
        s1[i] = an_factor * pi_ptr[i] + bn_factor * tao_ptr[i]
        s2[i] = bn_factor * pi_ptr[i] + an_factor * tao_ptr[i]

    hx[0] = hx[1]

    cdef int n
    for n in range(2, nmax):
        yx[2]  = (2*n - 1) * yx[1] * ONE_OVER_X - yx[0]
        hx[1]  = bx[n] + 1j * yx[2]
        ax     = x * bx[n-1] - n * bx[n]
        az     = z * bz[n-1] - n * bz[n]
        ahx    = x * hx[0] - n * hx[1]
        an     = (m2 * bz[n] * ax - bx[n] * az) / (m2 * bz[n] * ahx - hx[1] * az)
        bn     = (bz[n] * ax - bx[n] * az) / (bz[n] * ahx - hx[1] * az)
        cn    += 2

        qext       += cn * (an.real + bn.real)
        qsca_total += cn * ((an.real*an.real + an.imag*an.imag) + (bn.real*bn.real + bn.imag*bn.imag))
        yx[0] = yx[1];   yx[1] = yx[2];   hx[0] = hx[1]

        # Hoist math out of inner loop & compute pointer offset
        an_factor = (cn / (n * (n + 1.))) * an
        bn_factor = (cn / (n * (n + 1.))) * bn
        offset = (n - 1) * nsca

        for i in range(nsca):
            s1[i] += an_factor * pi_ptr[offset + i] + bn_factor * tao_ptr[offset + i]
            s2[i] += bn_factor * pi_ptr[offset + i] + an_factor * tao_ptr[offset + i]

    # trapezoid integration over NA
    cdef double qsca_na = 0
    cdef double q1, q2
    for i in range(1, nsca - 1):
        q1 = (s1[i].real*s1[i].real + s1[i].imag*s1[i].imag) + \
             (s2[i].real*s2[i].real + s2[i].imag*s2[i].imag)
        qsca_na += q1 * sin_thetas[i]

    q1 = (s1[0].real*s1[0].real + s1[0].imag*s1[0].imag) + \
         (s2[0].real*s2[0].real + s2[0].imag*s2[0].imag)
    q2 = (s1[nsca-1].real*s1[nsca-1].real + s1[nsca-1].imag*s1[nsca-1].imag) + \
         (s2[nsca-1].real*s2[nsca-1].real + s2[nsca-1].imag*s2[nsca-1].imag)

    qsca_na += 0.5 * (q1 * sin_thetas[0] + q2 * sin_thetas[nsca-1])
    qsca_na *= dtheta

    qext       *= 2 * ONE_OVER_X**2
    qsca_total *= 2 * ONE_OVER_X**2
    qsca_na    *= ONE_OVER_X**2

    free(buffer) # Only 1 free call required now

    return qext, qsca_total, qsca_na


# ── public function ───────────────────────────────────────────────────────────
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def q_ext_sca_na(
        ms, lambds, rs,
        theta_na:         U[float, np.ndarray, list] = 0.4,
        theta_resolution: int   = 12,
        theta_min:        float = 0.0,
):
    """
    Extinction and scattering efficiencies with numerical aperture integration.
    """
    out_shape = np.broadcast(ms, lambds, rs).shape
    ms     = np.broadcast_to(ms,     out_shape)
    lambds = np.broadcast_to(lambds, out_shape)
    rs     = np.broadcast_to(rs,     out_shape)

    qext    = np.empty(out_shape, np.float64)
    qsca    = np.empty(out_shape, np.float64)
    qsca_na = np.empty(out_shape, np.float64)

    x     = 2 * np.pi * rs / lambds
    x_max = np.max(x)
    nmax  = int(round(2 + x_max + 4 * x_max ** (1 / 3)) + 1)

    if isinstance(theta_na, (int, float)):
        thetas = np.linspace(theta_min, theta_na, theta_resolution)
    elif isinstance(theta_na, (list, np.ndarray)):
        thetas = np.asarray(theta_na)
    else:
        raise ValueError(f"Unexpected type for theta_na: {type(theta_na)}")

    sin_thetas = np.sin(thetas)

    # We no longer rely on SciPy!
    pi, tao    = fast_pi_tao(thetas, nmax)

    cdef double complex[::1] ms_view      = ms.ravel().astype(np.complex128)
    cdef double[::1]         lambds_view  = lambds.ravel().astype(np.float64)
    cdef double[::1]         rs_view      = rs.ravel().astype(np.float64)
    cdef double[::1]         qext_view    = qext.ravel()
    cdef double[::1]         qsca_view    = qsca.ravel()
    cdef double[::1]         qsca_na_view = qsca_na.ravel()

    cdef double[:, ::1]      pi_view      = pi
    cdef double[:, ::1]      tao_view     = tao
    cdef double[::1]         thetas_view  = thetas.astype(np.float64)
    cdef double[::1]         sin_t_view   = sin_thetas

    cdef int i
    cdef int N = int(np.prod(out_shape)) if out_shape else 1

    for i in prange(N, nogil=True):
        qext_view[i], qsca_view[i], qsca_na_view[i] = q_ext_sca_na_scalar(
            ms_view[i], lambds_view[i], rs_view[i],
            pi_view, tao_view, thetas_view, sin_t_view,
        )

    return qext, qsca, qsca_na
