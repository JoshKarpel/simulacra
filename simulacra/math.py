import logging
from typing import Callable, Tuple, Union

import numpy as np
import numpy.random as rand
import scipy.special as special
import scipy.integrate as integ

from . import exceptions
from . import units as u

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def rand_phases(shape_tuple: Tuple[int]) -> np.ndarray:
    """Return random phases (0 to 2pi) in the specified shape."""
    return rand.random_sample(shape_tuple) * u.twopi


def rand_phases_like(example: np.ndarray) -> np.ndarray:
    """Return random phases (0 to 2pi) in the same shape as the ``example``."""
    return rand.random_sample(example.shape) * u.twopi


class SphericalHarmonic:
    """A class that represents a spherical harmonic."""

    __slots__ = ("_l", "_m")

    def __init__(self, l: int = 0, m: int = 0):
        """
        Parameters
        ----------
        l
            Orbital angular momentum "quantum number". Must be >= 0.
        m
            Azimuthal angular momentum "quantum number". Must have
            ``abs(m) <= l``.
        """
        if not l >= 0:
            raise exceptions.IllegalSphericalHarmonic(
                f"invalid spherical harmonic: l = {l} must be greater than or equal to 0"
            )
        if not abs(m) <= l:
            raise exceptions.IllegalSphericalHarmonic(
                f"invalid spherical harmonic: |m| = {abs(m)} must be less than l = {l}"
            )

        self._l = l
        self._m = m

    @property
    def l(self) -> int:
        return self._l

    @property
    def m(self) -> int:
        return self._m

    def __repr__(self):
        return f"{self.__class__.__name__}(l={self.l}, m={self.m})"

    def __str__(self):
        return f"Y_({self.l},{self.m})"

    @property
    def latex(self) -> str:
        """Returns a LaTeX-formatted string for the SphericalHarmonic."""
        return fr"Y_{{{self.m}}}^{{{self.l}}}"

    @property
    def lm(self):
        """The tuple (l, m) for this spherical harmonic."""
        return self.l, self.m

    def __hash__(self):
        return hash((self.__class__, self.lm))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.lm == other.lm

    def __le__(self, other):
        return self.lm <= other.lm

    def __ge__(self, other):
        return self.lm >= other.lm

    def __lt__(self, other):
        return self.lm < other.lm

    def __gt__(self, other):
        return self.lm > other.lm

    def __call__(
        self, theta: Union[int, float, np.array], phi: Union[int, float, np.array] = 0
    ):
        """
        Evaluate the spherical harmonic at a point, or vectorized over an array
        of points.

        Parameters
        ----------
        theta
            The polar coordinate.
        phi
            The azimuthal coordinate.

        Returns
        -------
        val
            The value of the spherical harmonic evaluated at (``theta``, ``phi``).
        """
        return special.sph_harm(self.m, self.l, phi, theta)


def complex_quad(
    integrand: Callable, a: float, b: float, **kwargs
) -> Tuple[complex, tuple, tuple]:
    """
    As :func:`scipy.integrate.quad`, but works with complex integrands.

    Parameters
    ----------
    integrand
        The function to integrate.
    a
        The lower limit of integration.
    b
        The upper limit of integration.
    kwargs
        Additional keyword arguments are passed to
        :func:`scipy.integrate.quad`.

    Returns
    -------
    (result, real_extras, imag_extras)
        A tuple containing the result, as well as the other things returned
        by :func:`scipy.integrate.quad` for the real and imaginary parts
        separately.
    """

    def real_func(*args, **kwargs):
        return np.real(integrand(*args, **kwargs))

    def imag_func(*args, **kwargs):
        return np.imag(integrand(*args, **kwargs))

    real_integral = integ.quad(real_func, a, b, **kwargs)
    imag_integral = integ.quad(imag_func, a, b, **kwargs)

    return (
        real_integral[0] + (1j * imag_integral[0]),
        real_integral[1:],
        imag_integral[1:],
    )


def complex_quadrature(
    integrand: Callable, a: float, b: float, **kwargs
) -> Tuple[complex, tuple, tuple]:
    """
    As :func:`scipy.integrate.quadrature`, but works with complex integrands.

    Parameters
    ----------
    integrand
        The function to integrate.
    a
        The lower limit of integration.
    b
        The upper limit of integration.
    kwargs
        Additional keyword arguments are passed to
        :func:`scipy.integrate.quadrature`.

    Returns
    -------
    (result, real_extras, imag_extras)
        A tuple containing the result, as well as the other things returned
        by :func:`scipy.integrate.quadrature` for the real and imaginary parts
        separately.
    """

    def real_func(*args, **kwargs):
        return np.real(integrand(*args, **kwargs))

    def imag_func(*args, **kwargs):
        return np.imag(integrand(*args, **kwargs))

    real_integral = integ.quadrature(real_func, a, b, **kwargs)
    imag_integral = integ.quadrature(imag_func, a, b, **kwargs)

    return (
        real_integral[0] + (1j * imag_integral[0]),
        real_integral[1:],
        imag_integral[1:],
    )


def complex_dblquad(
    integrand: Callable, a: float, b: float, gfun: Callable, hfun: Callable, **kwargs
) -> Tuple[complex, tuple, tuple]:
    """
    As :func:`scipy.integrate.dblquad`, but works with complex integrands.

    Parameters
    ----------
    integrand
        The function to integrate.
    a
        The lower limit of integration.
    b
        The upper limit of integration.
    kwargs
        Additional keyword arguments are passed to
        :func:`scipy.integrate.dblquad`.

    Returns
    -------
    (result, real_extras, imag_extras)
        A tuple containing the result, as well as the other things returned
        by :func:`scipy.integrate.dblquad` for the real and imaginary parts
        separately.
    """

    def real_func(y, x):
        return np.real(integrand(y, x))

    def imag_func(y, x):
        return np.imag(integrand(y, x))

    real_integral = integ.dblquad(real_func, a, b, gfun, hfun, **kwargs)
    imag_integral = integ.dblquad(imag_func, a, b, gfun, hfun, **kwargs)

    return (
        real_integral[0] + (1j * imag_integral[0]),
        real_integral[1:],
        imag_integral[1:],
    )


def complex_nquad(integrand, ranges, **kwargs) -> Tuple[complex, tuple, tuple]:
    """
    As :func:`scipy.integrate.nquad`, but works with complex integrands.

    Parameters
    ----------
    integrand
        The function to integrate.
    a
        The lower limit of integration.
    b
        The upper limit of integration.
    kwargs
        Additional keyword arguments are passed to
        :func:`scipy.integrate.nquad`.

    Returns
    -------
    (result, real_extras, imag_extras)
        A tuple containing the result, as well as the other things returned
        by :func:`scipy.integrate.nquad` for the real and imaginary parts
        separately.
    """

    def real_func(y, x):
        return np.real(integrand(y, x))

    def imag_func(y, x):
        return np.imag(integrand(y, x))

    real_integral = integ.nquad(real_func, ranges, **kwargs)
    imag_integral = integ.nquad(imag_func, ranges, **kwargs)

    return (
        real_integral[0] + (1j * imag_integral[0]),
        real_integral[1:],
        imag_integral[1:],
    )
