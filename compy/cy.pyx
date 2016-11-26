cimport cython
import numpy as np
cimport numpy as np

import scipy.interpolate as interp
import scipy.sparse as sparse

complex = np.complex128
ctypedef np.complex128_t complex_t

float = np.float64
ctypedef np.float64_t float_t

ctypedef np.int_t int_t

@cython.boundscheck(False)
@cython.nonecheck(False)
# @cython.cdivision(True)
def tdma(matrix, np.ndarray[complex_t, ndim = 1] d):
    """
    Return the result of multiplying d by the inverse of matrix.

    :param matrix: A scipy diagonal sparse matrix.
    :param d: A numpy array.
    :return: x = m^-1 d
    """
    cdef np.ndarray[complex_t, ndim = 1] subdiagonal = np.concatenate((np.zeros(1), matrix.data[0]))
    cdef np.ndarray[complex_t, ndim = 1] diagonal = matrix.data[1]
    cdef np.ndarray[complex_t, ndim = 1] superdiagonal = matrix.data[2, 1:]

    cdef int n = diagonal.shape[0]  # dimension of the matrix, also the number of equations we have to solve

    cdef np.ndarray[complex_t, ndim = 1] new_superdiagonal = np.zeros(n - 1, dtype=complex)  # allocate in advance so we don't have to keep appending
    cdef np.ndarray[complex_t, ndim = 1] new_d = np.zeros(n, dtype=complex)
    cdef np.ndarray[complex_t, ndim = 1] x = np.zeros(n, dtype=complex)

    cdef unsigned int i
    cdef complex_t subi

    # compute the primed superdiagonal
    new_superdiagonal[0] = superdiagonal[0] / diagonal[0]
    for i in range(1, n - 1):
        new_superdiagonal[i] = superdiagonal[i] / (diagonal[i] - (subdiagonal[i] * new_superdiagonal[i - 1]))

    # compute the primed d
    new_d[0] = d[0] / diagonal[0]
    for i in range(1, n):
        new_d[i] = (d[i] - (subdiagonal[i] * new_d[i - 1])) / (diagonal[i] - (subdiagonal[i] * new_superdiagonal[i - 1]))

    # compute the answer
    x[n - 1] = new_d[n - 1]
    for i in reversed(range(0, n - 1)):  # iterate in reversed order, since we need to construct x from back to front
        x[i] = new_d[i] - (new_superdiagonal[i] * x[i + 1])

    return x


@cython.boundscheck(False)
@cython.nonecheck(False)
# @cython.cdivision(True)
def chebyshev_fit(np.ndarray[float_t, ndim = 1] rescaled_z, np.ndarray[float_t, ndim = 1] rescaled_rho, np.ndarray[float_t, ndim = 2] g_mesh, int terms):
    cdef unsigned int n, m, i ,j

    cdef np.ndarray[float_t, ndim = 2] c_nm = np.zeros((terms, terms), dtype = float)

    cdef np.ndarray[float_t, ndim = 1] arg = (np.arange(0, terms, dtype = float) + 0.5) * np.pi / terms
    cdef int arg_shape = arg.shape[0]
    cdef np.ndarray[float_t, ndim = 1] eval_points = np.zeros(arg_shape, dtype = float)
    for n in range(arg_shape):
        eval_points[n] = np.cos(arg[n])
    cdef np.ndarray[float_t, ndim = 1] chebyshev_n = np.zeros(arg_shape, dtype = float)
    cdef np.ndarray[float_t, ndim = 1] chebyshev_m = np.zeros(arg_shape, dtype = float)

    cdef float_t prefactor
    cdef float_t n_check = 0
    cdef float_t m_check = 0

    spline = interp.RectBivariateSpline(rescaled_z, rescaled_rho, g_mesh)
    cdef np.ndarray[float_t, ndim = 2] spline_mesh = np.zeros((terms, terms), dtype = float)
    for i in range(terms):
        for j in range(terms):
            spline_mesh[i, j] = spline(eval_points[i], eval_points[j])[0, 0]

    for n in range(terms):
            for m in range(terms):
                if n == 0:
                    n_check = 1
                if m == 0:
                    m_check = 1
                prefactor = (2 - n_check) * (2 - m_check) / (terms ** 2)
                for i in range(terms):
                    for j in range(terms):
                        c_nm[n, m] += spline_mesh[i, j] * np.cos(n * arg[i]) * np.cos(m * arg[j])
                c_nm[n, m] *= prefactor

    return c_nm
