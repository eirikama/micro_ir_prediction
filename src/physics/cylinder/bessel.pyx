# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, round, pow, M_PI, exp
from cython.parallel import prange
import cython
from typing import Union as U
import numpy as np
cimport numpy as np
np.import_array()

from scipy.special.cython_special cimport jv, yv, y0, y1
from numpy.math cimport INFINITY

cdef extern from "<complex>" namespace "std" nogil:
    double complex csin "sin"(double complex z)
    double complex ccos "cos"(double complex z)
    double complex csqrt "sqrt"(double complex z)
    double complex cexp "exp"(double complex z)
    double complex cabs "abs"(double complex z)

cpdef np.ndarray jv_range(v: float, z):
    """
    Returns range of jv values from v + 0 to v + n.

    :param v: non-negative real order
    :param z: number or array-like arguments
    :return: array of shape (z.shape, n + 1)
    """
    assert v >= 0, 'v must be non-negative'
    z = np.asarray(z, dtype=np.complex128)

    cdef int v_int_part = int(v)
    cdef double v_fract_part = v - v_int_part

    cdef np.npy_intp array_size = v_int_part + 1
    out_shape = z.shape + (array_size,)
    cdef out = np.empty(out_shape, np.complex128)

    # cython views to use with nogil
    cdef double complex[:] zs = z.ravel()
    cdef double complex[:, :] out_view = out.reshape(-1, array_size)

    cdef int N = out_view.shape[0]
    cdef int i
    for i in prange(N, nogil=True):
        if array_size == 1:
            out_view[i, 0] = jv(v_fract_part, zs[i])
        else:
            jv_range_impl(v_fract_part, v_int_part, zs[i],
                          out_view[i])

    return out


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.infer_types(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void jv_range_impl(
        double v_fract_part, int v_int_part, double complex z,
        double complex[:] out) nogil:
    """
    Calculates all jv values from v+0 to v+n in reverse order with regard to

    Recurrence Techniques for the Calculation of Bessel Functions. 1959

    :param v_fract_part: real number in range [0, 1)
    :param v_int_part: integer part of order to calculate
    :param z: argument
    :param out: preallocated array with v_int_part + 1 elements
    """
    if z == 0:
        out[0] = 1
        out[1:] = 0
        return

    cdef double complex ONE_OVER_Z = 1 / z

    cdef double complex prev = 0
    cdef double complex cur = 1e-300
    cdef int NMAX = v_int_part + 50

    cdef int i
    for i in range(NMAX - 2, v_int_part - 2, -1):
        cur, prev = 2 * (i + 1 + v_fract_part) * cur * ONE_OVER_Z - prev, cur

    out[v_int_part] = prev
    out[v_int_part - 1] = cur

    for i in range(v_int_part - 2, -1, -1):
        out[i] = 2*(i + 1 + v_fract_part) * out[i + 1] * ONE_OVER_Z - out[i + 2]

    cdef double complex alphaz = jv(1 + v_fract_part, z) / out[1]

    for i in range(v_int_part, -1, -1):
        out[i] = out[i] * alphaz


cpdef np.ndarray yv_range(v: float, x):
    """
    Returns range of yv values from v + 0 to v + n (including).

    :param v: non-negative real order
    :param x: number or array-like arguments
    :return: array of shape (z.shape, n + 1)
    """
    assert v >= 0, 'v must be non-negative'
    x = np.asarray(x)
    # it quickly diverges if complex part is non-zero
    assert x.dtype != np.complex, 'Function doesn\'t support complex arguments'
    x = x.astype(np.float)

    cdef int v_int_part = int(v)
    cdef double v_fract_part = v - v_int_part

    cdef np.npy_intp array_size = v_int_part + 1
    out_shape = x.shape + (array_size,)
    cdef out = np.empty(out_shape, np.float)

    # cython views to use with nogil
    cdef double [:] xs = x.ravel()
    cdef double [:, :] out_view = out.reshape(-1, array_size)

    cdef int N = out_view.shape[0]
    cdef int i
    for i in prange(N, nogil=True):
        if array_size == 1:
            out_view[i, 0] = yv(v_fract_part, xs[i])
        else:
            yv_range_float_impl(v_fract_part, v_int_part, xs[i], out_view[i])

    return out

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.infer_types(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void yv_range_float_impl(
        double v_fract_part, int v_int_part, double x, double[:] out) nogil:
    """
    Calculates all yv values from v+0 to v+n (including) in forward order
    with regard to

    Recurrence Techniques for the Calculation of Bessel Functions. 1959

    :param v_fract_part: real number in range [0, 1)
    :param v_int_part: integer part of order to calculate
    :param x: real argument
    :param out: preallocated array with v_int_part + 1 elements
    """
    if x == 0:
        out[:] = -INFINITY
        return

    cdef double ONE_OVER_X = 1 / x
    cdef int N = v_int_part + 1
    cdef int i
    if v_fract_part == 0:
        out[0] = y0(x)
        out[1] = y1(x)
    else:
        out[0] = yv(v_fract_part, x)
        out[1] = yv(1 + v_fract_part, x)
    for i in range(2, N):
        out[i] = ONE_OVER_X * 2 * (i - 1 + v_fract_part) * out[i - 1] - out[i - 2]
