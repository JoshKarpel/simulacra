import logging

import numpy as np
import numpy.fft as nfft
import scipy.sparse as sparse
import scipy.special as spc

import math
from compy import utils
from compy.units import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def sinc(x):
    """A wrapper over np.sinc, which is really sinc(pi * x). This version is sinc(x)."""
    return np.sinc(x / pi)


def gaussian(x, center, sigma, prefactor):
    return prefactor * np.exp(-0.5 * (((x - center) / sigma) ** 2))
    # return (prefactor / (sigma * np.sqrt(twopi))) * np.exp(-0.5 * (((x - center) / sigma) ** 2))


def gaussian_fwhm_from_sigma(sigma):
    return np.sqrt(8 * np.log(2)) * sigma


class SphericalHarmonic:
    """A class that represents a spherical harmonic."""

    __slots__ = ('_l', '_m')

    def __init__(self, l = 0, m = 0):
        self._l = l
        self._m = m

    @property
    def l(self):
        return self._l

    @property
    def m(self):
        return self._m

    def __str__(self):
        return 'Y_({},{})'.format(self.l, self.m)

    def __repr__(self):
        return '{}(l={}, m={})'.format(self.__class__.__name__, self.l, self.m)

    @property
    def tex_str(self):
        """Gets a LaTeX-formatted string for the BoundState."""
        return r'Y_{{{}}}^{{{}}}'.format(self.m, self.l)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.l == other.l and self.m == other.m

    def __hash__(self):
        return hash((self.l, self.m))

    def __call__(self, theta, phi = 0):
        """
        Evaluate the spherical harmonic at a point, or vectorized over an array of points.

        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the spherical harmonic at (theta, phi)
        """
        return spc.sph_harm(self.m, self.l, phi, theta)


@utils.memoize()
def fibonacci(n):
    """Return the n-th Fibonacci number, with Fibonacci(0) = Fibonacci(1) = 1."""
    if 0 <= n == int(n):
        if n == 0 or n == 1:
            return 1
        else:
            return fibonacci(n - 1) + fibonacci(n - 2)
    else:
        raise ValueError('{} is not a valid index for the Fibonacci sequence')


def is_prime(n):
    """Check whether n is prime."""
    if n != int(n) or n < 2:
        return False
    elif n == 2:
        return True
    elif n % 2 == 0:
        return False
    for divisor in range(3, math.ceil(math.sqrt(n)) + 1, 2):
        if n % divisor == 0:
            return False
    return True


def prime_generator():
    """Yield primes forever."""
    yield 2

    primes = [2]

    test = 1
    while True:
        test += 2
        sqrt_test = np.sqrt(test)
        for divisor in primes:
            if test % divisor == 0:
                break
            elif divisor > sqrt_test:
                primes.append(test)
                yield test
                break


def prime_sieve(limit):
    """A generator that yields all of the primes below the limit."""
    yield 2

    primes = {n: True for n in range(3, limit + 1, 2)}

    for n, check in primes.items():
        if check:
            n_squared = n * n
            if n_squared > limit + 1:
                yield n
            else:
                primes.update({m: False for m in range(n_squared, limit + 1, n) if m % 2 != 0})
                yield n


def prime_factorization(n):
    """Return the prime factorization of the input."""
    if n != int(n) and n > 0:
        raise ValueError('n ({}) must be a positive integer'.format(n))

    factors = []
    divisor = 2
    sqrt_n = math.sqrt(n)

    while True:
        if n % divisor == 0:
            factors.append(divisor)
            n = n / divisor
        elif n == 1:
            break
        elif divisor > sqrt_n:
            factors.append(int(n))
            break
        else:
            divisor += 1

    return factors


def centered_first_derivative(y, dx):
    """

    :param y: vector of data to take a derivative of
    :param dx: spacing between y points
    :return:
    """
    dx = np.abs(dx)
    offdiagonal = np.ones(len(y) - 1) / (2 * dx)
    operator = sparse.diags([-offdiagonal, np.zeros(len(y)), offdiagonal], offsets = (-1, 0, 1))
    operator.data[0][-2] = -1 / dx
    operator.data[1][0] = -1 / dx
    operator.data[1][-1] = 1 / dx
    operator.data[2][1] = 1 / dx

    return operator.dot(y)


def centered_second_derivative(y, dx):
    """

    :param y: vector of data to take a derivative of
    :param dx: spacing between y points
    :return:
    """
    # offdiagonal = np.ones(len(y) - 1) / (dx ** 2)
    # diagonal = -2 * np.ones(len(y)) / (dx ** 2)
    # operator = sparse.diags([-offdiagonal, diagonal, offdiagonal], offsets = (-1, 0, 1))



    # print(operator.toarray())

    return centered_first_derivative(centered_first_derivative(y, dx), dx)
