import datetime as dt
import functools
import logging
import os
import itertools as it
import functools as ft
from copy import deepcopy

import numpy as np
import scipy as sp
import scipy.special as special

import compy as cp
from compy.units import *
from . import core

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class IllegalQuantumState(Exception):
    pass


class QuantumState:
    def __init__(self, amplitude = 1):
        self.amplitude = amplitude

    @property
    def norm(self):
        return np.abs(self.amplitude) ** 2

    def __repr__(self):
        return self.__class__.__name__

    def __iter__(self):
        yield self

    def __add__(self, other):
        return Superposition(*self, *other)

    def __mul__(self, other):
        new = deepcopy(self)
        new.amplitude *= other
        return new

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1 / other)


class Superposition(QuantumState):
    """A class that represents a superposition of bound states."""

    def __init__(self, *states):
        """
        Construct a discrete superposition of states.

        If normalize is True the initial amplitudes are rescaled so that the state is normalized.

        :param state: a dict of HydrogenBoundState:state amplitude (complex number) pairs.
        :param normalize: if True, renormalize the state amplitudes.
        """
        norm = np.sqrt(sum(s.norm for s in states))
        self.states = list(s / norm for s in states)

        super(Superposition, self).__init__(amplitude = 1)

    def __str__(self):
        return '(' + ' + '.join([str(s) for s in self.states]) + ')'

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join([repr(p) for p in self.states]))

    def __getitem__(self, item):
        return self.states[item]

    def __iter__(self):
        yield from self.states

    def __call__(self, *args, **kwargs):
        return sum(s(*args, **kwargs) for s in self.states)


class HydrogenBoundState(QuantumState):
    """A class that represents a hydrogenic bound state."""

    __slots__ = ('_n', '_l', '_m')

    def __init__(self, n = 1, l = 0, m = 0, amplitude = 1):
        """
        Construct a HydrogenBoundState from its three quantum numbers (n, l, m).

        :param n: principal quantum number
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
        """
        super(HydrogenBoundState, self).__init__(amplitude = amplitude)

        self.n = n
        self.l = l
        self.m = m

    @property
    def n(self):
        """Gets _n."""
        return self._n

    @n.setter
    def n(self, n):
        if 0 < n == int(n):
            self._n = n
        else:
            raise IllegalQuantumState('n ({}) must be an integer greater than zero'.format(n))

    @property
    def l(self):
        """Gets _l."""
        return self._l

    @l.setter
    def l(self, l):
        if int(l) == l and 0 <= l < self.n:
            self._l = l
        else:
            raise IllegalQuantumState('l ({}) must be greater than or equal to zero and less than n ({})'.format(l, self.n))

    @property
    def m(self):
        """Gets _m."""
        return self._m

    @m.setter
    def m(self, m):
        if int(m) == m and -self.l <= m <= self.l:
            self._m = m
        else:
            IllegalQuantumState('|m| (|{}|) must be less than l ({})'.format(m, self.l))

    @property
    def spherical_harmonic(self):
        """Gets the SphericalHarmonic associated with the HydrogenBoundState's l and m."""
        return cp.math.SphericalHarmonic(l = self.l, m = self.m)

    def __str__(self):
        """Returns the external string representation of the HydrogenBoundState."""
        return self.ket

    def __repr__(self):
        """Returns the internal string representation of the HydrogenBoundState."""
        return '{}(n = {}, l = {}, m = {}, amplitude = {})'.format(self.__class__.__name__, self.n, self.l, self.m, self.amplitude)

    @property
    def ket(self):
        """Gets the ket representation of the HydrogenBoundState."""
        return '{}|{},{},{}>'.format(np.around(self.amplitude, 3), self.n, self.l, self.m)

    @property
    def bra(self):
        """Gets the bra representation of the HydrogenBoundState"""
        return '{}<{},{},{}|'.format(np.around(self.amplitude, 3), self.n, self.l, self.m)

    @property
    def tex_str(self):
        """Gets a LaTeX-formatted string for the HydrogenBoundState."""
        return r'\psi_{{{},{},{}}}'.format(self.n, self.l, self.m)

    # TODO: switch to simple checking of (n, l, m) tuple

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.n == other.n and self.l == other.l and self.m == other.m

    def __lt__(self, other):
        return isinstance(other, self.__class__) and self.n < other.n and self.l < other.l and self.m < other.m

    def __gt__(self, other):
        return isinstance(other, self.__class__) and self.n > other.n and self.l > other.l and self.m > other.m

    def __le__(self, other):
        return isinstance(other, self.__class__) and self.n <= other.n and self.l <= other.l and self.m <= other.m

    def __ge__(self, other):
        return isinstance(other, self.__class__) and self.n >= other.n and self.l >= other.l and self.m >= other.m

    def __hash__(self):
        return hash((self.n, self.l, self.m))

    @staticmethod
    def sort_key(state):
        return state.n, state.l, state.m

    def radial_part(self, r):
        normalization = np.sqrt(((2 / (self.n * bohr_radius)) ** 3) * (sp.math.factorial(self.n - self.l - 1) / (2 * self.n * sp.math.factorial(self.n + self.l))))  # Griffith's normalization
        r_dep = np.exp(-r / (self.n * bohr_radius)) * ((2 * r / (self.n * bohr_radius)) ** self.l)
        lag_poly = special.eval_genlaguerre(self.n - self.l - 1, (2 * self.l) + 1, 2 * r / (self.n * bohr_radius))

        return normalization * r_dep * lag_poly

    def __call__(self, r, theta, phi):
        """
        Evaluate the hydrogenic bound state wavefunction at a point, or vectorized over an array of points.

        :param r: radial coordinate
        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the wavefunction at (r, theta, phi)
        """
        radial_part = self.radial_part(r)
        sph_harm = self.spherical_harmonic(theta, phi)

        return self.amplitude * radial_part * sph_harm


class HydrogenFreeState(QuantumState):
    """A class that represents a hydrogenic free state."""

    __slots__ = ('_energy', '_l', '_m')

    def __init__(self, energy = 1 * eV, l = 0, m = 0, amplitude = 1):
        """
        Construct a FreeState from its energy and angular momentum quantum numbers.

        :param energy: energy of the free state
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
        """
        super(HydrogenFreeState, self).__init__(amplitude = amplitude)

        if any(int(x) != x for x in (l, m)):
            raise IllegalQuantumState('l and m must be integers')

        if energy > 0:
            self._energy = energy
        else:
            raise IllegalQuantumState('energy must be greater than zero')

        if l >= 0:
            self._l = l
        else:
            raise IllegalQuantumState('l ({}) must be greater than or equal to zero'.format(l))

        if -l <= m <= l:
            self._m = m
        else:
            raise IllegalQuantumState('m ({}) must be between -l and l ({} to {})'.format(m, -l, l))

    @classmethod
    def from_wavenumber(cls, k, l = 0, m = 0):
        """Construct a FreeState from its wavenumber and angular momentum quantum numbers."""
        energy = core.electron_energy_from_wavenumber(k)

        return cls(energy, l, m)

    @property
    def energy(self):
        return self._energy

    @property
    def k(self):
        return core.electron_wavenumber_from_energy(self.energy)

    @property
    def l(self):
        return self._l

    @property
    def m(self):
        return self._m

    @property
    def spherical_harmonic(self):
        return cp.math.SphericalHarmonic(l = self.l, m = self.m)

    def __str__(self):
        return self.ket

    def __repr__(self):
        return '{}(T = {} eV, k = {} 1/nm, l = {}, m = {}, amplitude = {})'.format(self.__class__.__name__, uround(self.energy, eV, 3), uround(self.k, 1 / nm, 3), self.l, self.m, self.amplitude)

    @property
    def ket(self):
        return '{}|{} eV,{} 1/nm, {}, {}>'.format(np.around(self.amplitude, 3), uround(self.energy, eV, 3), uround(self.k, 1 / nm, 3), self.l, self.m)

    @property
    def bra(self):
        return '{}<{} eV,{} 1/nm, {}, {}|'.format(np.around(self.amplitude, 3), uround(self.energy, eV, 3), uround(self.k, 1 / nm, 3), self.l, self.m)

    @property
    def tex_str(self):
        """Return a LaTeX-formatted string for the HydrogenFreeState."""
        return r'\phi_{{{},{},{}}}'.format(uround(self.energy, eV, 3), self.l, self.m)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and (self.energy, self.l, self.m) == (other.energy, other.l, other.m)

    def __hash__(self):
        return hash((self.energy, self.l, self.m))

    def __call__(self, r, theta, phi):
        """
        Evaluate the hydrogenic bound state wavefunction at a point, or vectorized over an array of points.

        :param r: radial coordinate
        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the wavefunction at (r, theta, phi)
        """
        raise NotImplementedError


class QHOState(QuantumState):
    def __init__(self, omega, mass, n = 0, amplitude = 1, dimension_label = 'x'):
        self.n = n
        self.omega = omega
        self.mass = mass
        self.dimension_label = dimension_label

        super(QHOState, self).__init__(amplitude = amplitude)

    @property
    def energy(self):
        return hbar * self.omega * (self.n + 0.5)

    @property
    def frequency(self):
        return self.omega / twopi

    @property
    def period(self):
        return 1 / self.frequency

    def __str__(self):
        return self.ket

    def __repr__(self):
        return '{}(n = {}, mass = {}, omega = {}, energy = {}, amplitude = {})'.format(self.__class__.__name__,
                                                                                       self.n,
                                                                                       self.mass,
                                                                                       self.omega,
                                                                                       self.energy,
                                                                                       self.amplitude)

    @property
    def ket(self):
        return '|{}>'.format(self.n)

    @property
    def bra(self):
        return '<{}|'.format(self.n)

    @property
    def tex_str(self):
        """Return a LaTeX-formatted string for the QHOState."""
        return r'{}'.format(self.n)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and (self.omega, self.mass, self.n) == (other.omega, other.mass, other.n)

    def __hash__(self):
        return hash((self.omega, self.mass, self.n))

    def __call__(self, x):
        norm = ((self.mass * self.omega / (pi * hbar)) ** (1 / 4)) / (np.float64(2 ** (self.n / 2)) * np.sqrt(np.float64(sp.math.factorial(self.n))))
        exp = np.exp(-self.mass * self.omega * (x ** 2) / (2 * hbar))
        herm = special.hermite(self.n)(np.sqrt(self.mass * self.omega / hbar) * x)

        return self.amplitude * (norm * exp * herm).astype(np.complex128)
