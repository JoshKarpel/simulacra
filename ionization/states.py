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
    pass


class Superposition(QuantumState):
    """A class that represents a superposition of bound states."""

    __slots__ = ('state',)

    def __init__(self, state, normalize = True):
        """
        Construct a discrete superposition of states.

        If normalize is True the initial amplitudes are rescaled so that the state is normalized.

        :param state: a dict of BoundState:state amplitude (complex number) pairs.
        :param normalize: if True, renormalize the state amplitudes.
        """
        super(Superposition, self).__init__()

        state = dict(state)  # consume input iterators because we may need to reuse the dict several times

        if normalize:
            unnormalized_amplitude = np.sqrt(sum([np.abs(amp) ** 2 for amp in state.values()]))
            state = {state: amp / unnormalized_amplitude for state, amp in state.items()}

        self.state = state

    def __str__(self):
        pairs = ['{}: {}'.format(str(s), a) for s, a in self.state.items()]
        out = ', '.join(pairs)
        return out

    def __repr__(self):
        return repr(self.state)

    def __getitem__(self, item):
        return self.state[item]

    def __iter__(self):
        yield from self.state.items()

    @property
    def states(self):
        yield from self.state.keys()

    @property
    def amplitudes(self):
        return np.array(self.state.values())

    @property
    def norm(self):
        return np.sum(np.abs(self.amplitudes) ** 2)

    def __abs__(self):
        return self.norm

    def __call__(self, r, theta, phi):
        return sum(state(r, theta, phi) for state in self.states)


class HydrogenBoundState(QuantumState):
    """A class that represents a hydrogenic bound state."""

    __slots__ = ('_n', '_l', '_m')

    def __init__(self, n = 1, l = 0, m = 0):
        """
        Construct a BoundState from its three quantum numbers (n, l, m).

        :param n: principal quantum number
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
        """
        super(HydrogenBoundState, self).__init__()

        self.n = n
        self.l = l
        self.m = m

    @property
    def n(self):
        """Gets _n."""
        return self._n

    @n.setter
    def n(self, n):
        if int(n) == n and n > 0:
            self._n = n

    @property
    def l(self):
        """Gets _l."""
        return self._l

    @l.setter
    def l(self, l):
        if int(l) == l and 0 <= l < self.n:
            self._l = l

    @property
    def m(self):
        """Gets _m."""
        return self._m

    @m.setter
    def m(self, m):
        if int(m) == m and -self.l <= m <= self.l:
            self._m = m

    @property
    def spherical_harmonic(self):
        """Gets the SphericalHarmonic associated with the BoundState's l and m."""
        return cp.math.SphericalHarmonic(l = self.l, m = self.m)

    def __str__(self):
        """Returns the external string representation of the BoundState."""
        return self.ket

    def __repr__(self):
        """Returns the internal string representation of the BoundState."""
        return '{}(n={}, l={}, m={})'.format(self.__class__.__name__, self.n, self.l, self.m)

    @property
    def ket(self):
        """Gets the ket representation of the BoundState."""
        return '|{},{},{}>'.format(self.n, self.l, self.m)

    @property
    def bra(self):
        """Gets the bra representation of the BoundState"""
        return '<{},{},{}|'.format(self.n, self.l, self.m)

    @property
    def tex_str(self):
        """Gets a LaTeX-formatted string for the BoundState."""
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

        return radial_part * sph_harm


class HydrogenFreeState(QuantumState):
    """A class that represents a hydrogenic free state."""

    __slots__ = ('_energy', '_l', '_m')

    def __init__(self, energy = 1 * eV, l = 0, m = 0):
        """
        Construct a FreeState from its energy and angular momentum quantum numbers.

        :param energy: energy of the free state
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
        """
        super(HydrogenFreeState, self).__init__()

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
        return '{}(T={} eV, k={} 1/nm, l={}, m={})'.format(self.__class__.__name__, uround(self.energy, eV, 3), uround(self.k, 1 / nm, 3), self.l, self.m)

    @property
    def ket(self):
        return '|{} eV,{} 1/nm, {}, {}>'.format(uround(self.energy, eV, 3), uround(self.k, 1 / nm, 3), self.l, self.m)

    @property
    def bra(self):
        return '<{} eV,{} 1/nm, {}, {}|'.format(uround(self.energy, eV, 3), uround(self.k, 1 / nm, 3), self.l, self.m)

    @property
    def tex_str(self):
        """Return a LaTeX-formatted string for the BoundState."""
        return r'\phi_{{{},{},{}}}'.format(uround(self.energy, eV, 3), self.l, self.m)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.energy == other.energy and self.l == other.l and self.m == other.m

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
    def __init__(self, n = 0, omega = twopi * THz, dimension_label = 'x'):
        self.n = n
        self.omega = omega
        self.dimension_label = dimension_label
