"""
Simulacra mathematics sub-package.


Copyright 2017 Josh Karpel

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging

import numpy as np
import scipy.sparse as sparse
import scipy.special as special
import scipy.integrate as integ
from typing import Callable, Generator, Iterable

from . import utils
from .units import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def sinc(x):
    """A wrapper over np.sinc, which is really sinc(pi * x). This version is sinc(x)."""
    return np.sinc(x / pi)


def gaussian(x, center, sigma, prefactor):
    """
    Return an unnormalized Gaussian centered at :code:`center` with standard deviation :code:`sigma` and prefactor :code:`prefactor`.
    """
    return prefactor * np.exp(-0.5 * (((x - center) / sigma) ** 2))
    # return (prefactor / (sigma * np.sqrt(twopi))) * np.exp(-0.5 * (((x - center) / sigma) ** 2))


def gaussian_fwhm_from_sigma(sigma):
    """Return the full-width-at-half-max of a Gaussian with standard deviation :code:`sigma`"""
    return np.sqrt(8 * np.log(2)) * sigma


def stirling_approximation_exp(n):
    r"""
    Return the Stirling approximation of :math:`n!`, :math:`n! \approx \sqrt{2\pi} \left( \frac{n}{e} \right)^n`.

    Parameters
    ----------
    n : :class:`float`
        The number to approximate the factorial of.

    Returns
    -------
    :class:`float`
        The Stirling approximation to :math:`n!`.
    """
    return np.sqrt(twopi * n) * ((n / e) ** n)


def stirling_approximation_ln(n):
    """
    Return the Stirling approximation of :math:`\log n!`, :math:`\log n! \approx n \, \log n - n`.

    Parameters
    ----------
    n : :class:`float`
        The number to approximate the logarithm of the factorial of.

    Returns
    -------
    :class:`float`
        The Stirling approximation of :math:`\log n!`.
    """
    return n * np.log(n) - n


class SphericalHarmonic:
    """A class that represents a spherical harmonic."""

    __slots__ = ('_l', '_m')

    def __init__(self, l: int = 0, m: int = 0):
        """
        Initialize a SphericalHarmonic from its angular momentum numbers.

        Parameters
        ----------
        l
            Orbital angular momentum "quantum number". Must be >= 0.
        m
            Azimuthal angular momentum "quantum number". Must have ``abs(m) < l``.
        """
        self._l = l
        self._m = m

    @property
    def l(self) -> int:
        return self._l

    @property
    def m(self) -> int:
        return self._m

    def __str__(self):
        return f'Y_({self.l},{self.m})'

    def __repr__(self):
        return f'{self.__class__.__name__}(l={self.l}, m={self.m})'

    @property
    def latex(self) -> str:
        """Returns a LaTeX-formatted string for the SphericalHarmonic."""
        return fr'Y_{{self.m}}^{{self.l}}'

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
        return special.sph_harm(self.m, self.l, phi, theta)


def complex_quad(integrand: Callable, a, b, **kwargs):
    def real_func(x):
        return np.real(integrand(x))

    def imag_func(x):
        return np.imag(integrand(x))

    real_integral = integ.quad(real_func, a, b, **kwargs)
    imag_integral = integ.quad(imag_func, a, b, **kwargs)

    return (real_integral[0] + (1j * imag_integral[0]), real_integral[1:], imag_integral[1:])


def complex_dblquad(integrand, a, b, gfun, hfun, **kwargs):
    def real_func(y, x):
        return np.real(integrand(y, x))

    def imag_func(y, x):
        return np.imag(integrand(y, x))

    real_integral = integ.dblquad(real_func, a, b, gfun, hfun, **kwargs)
    imag_integral = integ.dblquad(imag_func, a, b, gfun, hfun, **kwargs)

    return (real_integral[0] + (1j * imag_integral[0]), real_integral[1:], imag_integral[1:])


def complex_nquad(integrand, ranges, **kwargs):
    def real_func(y, x):
        return np.real(integrand(y, x))

    def imag_func(y, x):
        return np.imag(integrand(y, x))

    real_integral = integ.nquad(real_func, ranges, **kwargs)
    imag_integral = integ.nquad(imag_func, ranges, **kwargs)

    return (real_integral[0] + (1j * imag_integral[0]), real_integral[1:], imag_integral[1:])


@utils.memoize
def fibonacci(n: int) -> int:
    """
    Return the n-th Fibonacci number, with Fibonacci(0) = 0, Fibonacci(1) = 1.

    The Fibonacci numbers are calculated via memoized recursion.
    """
    if 0 <= n == int(n):
        if n == 0 or n == 1:
            return n
        else:
            return fibonacci(n - 1) + fibonacci(n - 2)
    else:
        raise ValueError('{} is not a valid index for the Fibonacci sequence'.format(n))


def is_prime(n: int) -> bool:
    """Check whether n is prime by trial division."""
    if n != int(n) or n < 2:
        return False
    elif n == 2:
        return True
    elif n % 2 == 0:
        return False
    for divisor in range(3, np.ceil(np.sqrt(n)) + 1, 2):
        if n % divisor == 0:
            return False
    return True


def prime_generator() -> Generator:
    """Yield primes forever by trial division."""
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


def prime_sieve(limit: int) -> Generator:
    """A generator that yields all of the primes below the limit, via trial division."""
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


def prime_factorization(n: int) -> Iterable[int]:
    """Return the prime factorization of the input."""
    if n != int(n) and n > 0:
        raise ValueError('n ({}) must be a positive integer'.format(n))

    factors = []
    divisor = 2
    sqrt_n = np.sqrt(n)

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


def centered_first_derivative(y: np.ndarray, dx = 1) -> np.ndarray:
    """
    Return the centered first derivative of the array y, using spacing dx.

    :param y: vector of data to take a derivative of
    :param dx: spacing between y points
    :type dx: float
    :return: the centered first derivative of y
    """
    dx = np.abs(dx)

    offdiagonal = np.ones(len(y) - 1) / (2 * dx)
    operator = sparse.diags([-offdiagonal, np.zeros(len(y)), offdiagonal], offsets = (-1, 0, 1))

    operator.data[0][-2] = -1 / dx
    operator.data[1][0] = -1 / dx
    operator.data[1][-1] = 1 / dx
    operator.data[2][1] = 1 / dx

    return operator.dot(y)


def centered_second_derivative(y: np.ndarray, dx = 1) -> np.ndarray:
    """
    Return the centered second derivative of the array y, using spacing dx.

    :param y: vector of data to take a derivative of
    :param dx: spacing between y points
    :type dx: float
    :return: the centered second derivative of y
    """
    return centered_first_derivative(centered_first_derivative(y, dx), dx)
