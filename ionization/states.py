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
import scipy.optimize as optimize

import compy as cp
from compy.units import *
from . import core

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class IllegalQuantumState(cp.CompyException):
    pass


class QuantumState(cp.Summand):
    def __init__(self, amplitude = 1):
        super(QuantumState, self).__init__()
        self.amplitude = amplitude
        self.summation_class = Superposition

    @property
    def norm(self):
        return np.abs(self.amplitude) ** 2

    def __mul__(self, other):
        new = deepcopy(self)
        new.amplitude *= other
        return new

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1 / other)

    @property
    def tuple(self):
        return 0

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.tuple == other.tuple

    def __lt__(self, other):
        return isinstance(other, self.__class__) and self.tuple < other.tuple

    def __gt__(self, other):
        return isinstance(other, self.__class__) and self.tuple > other.tuple

    def __le__(self, other):
        return isinstance(other, self.__class__) and self.tuple <= other.tuple

    def __ge__(self, other):
        return isinstance(other, self.__class__) and self.tuple >= other.tuple


class Superposition(cp.Sum, QuantumState):
    """A class that represents a superposition of bound states."""

    container_name = 'states'

    def __init__(self, *states):
        """
        Construct a discrete superposition of states.

        If normalize is True the initial amplitudes are rescaled so that the state is normalized.

        :param state: a dict of HydrogenBoundState:state amplitude (complex number) pairs.
        :param normalize: if True, renormalize the state amplitudes.
        """
        super(Superposition, self).__init__(amplitude = 1)
        norm = np.sqrt(sum(s.norm for s in states))
        self.states = list(s / norm for s in states)


class FreeSphericalWave(QuantumState):
    """A class that represents a free spherical wave."""

    energy = cp.utils.Checked('energy', check = lambda x: x > 0)

    def __init__(self, energy = 1 * eV, l = 0, m = 0, amplitude = 1):
        """
        Construct a FreeState from its energy and angular momentum quantum numbers.

        :param energy: energy of the free state
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
        """
        super(FreeSphericalWave, self).__init__(amplitude = amplitude)

        if any(int(x) != x for x in (l, m)):
            raise IllegalQuantumState('l and m must be integers')

        self.energy = energy

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
    def k(self):
        return core.electron_wavenumber_from_energy(self.energy)

    @property
    def l(self):
        return self._l

    @property
    def m(self):
        return self._m

    @property
    def tuple(self):
        return self.k, self.l, self.m

    @property
    def spherical_harmonic(self):
        return cp.math.SphericalHarmonic(l = self.l, m = self.m)

    def __str__(self):
        return self.ket

    def __repr__(self):
        return '{}(energy = {} eV, k = {} 1/nm, l = {}, m = {}, amplitude = {})'.format(self.__class__.__name__, uround(self.energy, eV, 3), uround(self.k, 1 / nm, 3), self.l, self.m, self.amplitude)

    @property
    def ket(self):
        return '{}|{} eV, {} 1/nm, {}, {}>'.format(np.around(self.amplitude, 3), uround(self.energy, eV, 3), uround(self.k, 1 / nm, 3), self.l, self.m)

    @property
    def bra(self):
        return '{}<{} eV, {} 1/nm, {}, {}|'.format(np.around(self.amplitude, 3), uround(self.energy, eV, 3), uround(self.k, 1 / nm, 3), self.l, self.m)

    @property
    def tex_str(self):
        """Return a LaTeX-formatted string for the HydrogenFreeState."""
        return r'\Psi_{{{},{},{}}}'.format(uround(self.energy, eV, 3), self.l, self.m)

    def radial_function(self, r):
        return np.sqrt(2 * (self.k ** 2) / pi) * special.spherical_jn(self.l, self.k * r)

    def __call__(self, r, theta, phi):
        """
        Evaluate the free spherical wave wavefunction at a point, or vectorized over an array of points.

        :param r: radial coordinate
        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the wavefunction at (r, theta, phi)
        """
        return self.amplitude * self.radial_function(r) * self.spherical_harmonic(theta, phi)


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
    def tuple(self):
        return self.n, self.l, self.m

    @property
    def spherical_harmonic(self):
        """Gets the SphericalHarmonic associated with the HydrogenBoundState's l and m."""
        return cp.math.SphericalHarmonic(l = self.l, m = self.m)

    def __str__(self):
        """Returns the external string representation of the HydrogenBoundState."""
        return self.ket

    def __repr__(self):
        """Returns the internal string representation of the HydrogenBoundState."""
        return cp.utils.field_str(self, 'n', 'l', 'm', 'amplitude')

    @property
    def ket(self):
        """Gets the ket representation of the HydrogenBoundState."""
        return '{}|{},{},{}>'.format(np.around(self.amplitude, 3), *self.tuple)

    @property
    def bra(self):
        """Gets the bra representation of the HydrogenBoundState"""
        return '{}<{},{},{}|'.format(np.around(self.amplitude, 3), *self.tuple)

    @property
    def tex_str(self):
        """Gets a LaTeX-formatted string for the HydrogenBoundState."""
        return r'\psi_{{{},{},{}}}'.format(*self.tuple)

    @staticmethod
    def sort_key(state):
        return state.n, state.l, state.m

    def radial_function(self, r):
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
        return self.amplitude * self.radial_function(r) * self.spherical_harmonic(theta, phi)


class HydrogenFreeState(QuantumState):
    """A class that represents a hydrogenic free state."""

    def __init__(self, energy = 1 * eV, l = 0, m = 0, amplitude = 1):
        """
        Construct a HydrogenFreeState from its energy and angular momentum quantum numbers.

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

    @property
    def tuple(self):
        return self.k, self.l, self.m

    def __str__(self):
        return self.ket

    def __repr__(self):
        return '{}(energy = {} eV, k = {} 1/nm, l = {}, m = {}, amplitude = {})'.format(self.__class__.__name__, uround(self.energy, eV, 3), uround(self.k, 1 / nm, 3), self.l, self.m, self.amplitude)

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
    def __init__(self, spring_constant, mass, n = 0, amplitude = 1, dimension_label = 'x'):
        self.n = n
        self.spring_constant = spring_constant
        self.mass = mass
        self.dimension_label = dimension_label

        super(QHOState, self).__init__(amplitude = amplitude)

    @classmethod
    def from_omega_and_mass(cls, omega, mass, n = 0, amplitude = 1, dimension_label = 'x'):
        return cls(spring_constant = mass * (omega ** 2), mass = mass, n = n, amplitude = amplitude, dimension_label = dimension_label)

    @classmethod
    def from_QHO_potential_and_mass(cls, qho_potential, mass, n = 0, amplitude = 1, dimension_label = 'x'):
        return cls(spring_constant = qho_potential.spring_constant, mass = mass, n = n, amplitude = amplitude, dimension_label = dimension_label)

    @property
    def omega(self):
        return np.sqrt(self.spring_constant / self.mass)

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

    @property
    def tuple(self):
        return self.omega, self.mass, self.n

    def __call__(self, x):
        norm = ((self.mass * self.omega / (pi * hbar)) ** (1 / 4)) / (np.float64(2 ** (self.n / 2)) * np.sqrt(np.float64(sp.math.factorial(self.n))))
        exp = np.exp(-self.mass * self.omega * (x ** 2) / (2 * hbar))
        herm = special.hermite(self.n)(np.sqrt(self.mass * self.omega / hbar) * x)

        # TODO: Stirling's approximation for large enough n in the normalization factor

        return self.amplitude * (norm * exp * herm).astype(np.complex128)


class FiniteSquareWellState(QuantumState):
    def __init__(self, well_depth, well_width, mass, n = 1, well_center = 0, amplitude = 1):
        self.well_depth = well_depth
        self.well_width = well_width
        self.well_center = well_center
        self.mass = mass

        z_0 = (well_width / 2) * np.sqrt(2 * mass * well_depth) / hbar

        if z_0 < n * pi / 2:
            raise IllegalQuantumState('There is no bound state with the given parameters')

        left_bound = (n - 1) * pi / 2
        right_bound = left_bound + (pi / 2)

        if n % 2 != 0:
            z = optimize.brentq(lambda z: np.tan(z) - np.sqrt(((z_0 / z) ** 2) - 1), left_bound, right_bound)
            self.function_inside_well = np.cos
        else:
            z = optimize.brentq(lambda z: (1 / np.tan(z)) + np.sqrt(((z_0 / z) ** 2) - 1), left_bound, right_bound)
            self.function_inside_well = np.sin

        self.wavenumber_inside_well = z / (well_width / 2)
        self.energy = (((hbar * self.wavenumber_inside_well) ** 2) / (2 * mass)) - well_depth
        self.wavenumber_outside_well = np.sqrt(-2 * mass * self.energy) / hbar

        self.normalization_factor_inside_well = 1 / np.sqrt((self.well_width / 2) + (1 / self.wavenumber_outside_well))
        self.normalization_factor_outside_well = np.exp(self.wavenumber_outside_well * (self.well_width / 2)) * self.function_inside_well(self.wavenumber_inside_well * (self.well_width / 2)) / np.sqrt((self.well_width / 2) + (1 / self.wavenumber_outside_well))

        super(FiniteSquareWellState, self).__init__(amplitude = amplitude)

    @classmethod
    def all_states_of_well(cls, well_depth, well_width, mass, well_center = 0, amplitude = 1):
        states = []
        for n in it.count(1):
            try:
                states.append(cls(well_depth, well_width, mass, n = n, well_center = well_center, amplitude = 1))
            except IllegalQuantumState:
                return states

    @property
    def left_edge(self):
        return self.well_center - (self.well_width / 2)

    @property
    def right_edge(self):
        return self.well_center + (self.well_width / 2)

    def __call__(self, x):
        cond = np.greater_equal(x, self.left_edge) * np.less_equal(x, self.right_edge)

        return np.where(cond,
                        self.normalization_factor_inside_well * self.function_inside_well(self.wavenumber_inside_well * x),
                        self.normalization_factor_outside_well * np.exp(-self.wavenumber_outside_well * np.abs(x)))
