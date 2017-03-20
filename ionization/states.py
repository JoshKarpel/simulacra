import datetime as dt
import functools
import logging
import os
import types
import itertools as it
import functools as ft
from copy import deepcopy

import numpy as np
import scipy as sp
import scipy.special as special
import scipy.optimize as optimize
import mpmath

import compy as cp
from compy.units import *
from . import core

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class IllegalQuantumState(cp.CompyException):
    """An exception indicating that a state with an illegal quantum number has been generated."""
    pass


class QuantumState(cp.Summand):
    """A class that represents a quantum state, with an amplitude and some basic multiplication/addition rules. Can be summed to form a Superposition."""

    bound = None
    discrete_eigenvalues = None

    def __init__(self, amplitude = 1):
        """
        Construct a QuantumState with a given amplitude.

        QuantumStates should not be instantiated directly (they have no useful properties).

        :param amplitude: the probability amplitude of the state
        """
        super(QuantumState, self).__init__()
        self.amplitude = amplitude
        self.summation_class = Superposition

    @property
    def free(self):
        return not self.bound

    @property
    def continuous_eigenvalues(self):
        return not self.discrete_eigenvalues

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
        """This property should return a tuple of unique information about the state, which will be used to hash it or perform comparison operations."""
        raise NotImplementedError

    def __hash__(self):
        return hash((self.__class__.__name__, self.__doc__) + self.tuple)

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

    @property
    def tex_str(self):
        """Return a string in TeX notation that should be placed inside bras or kets in output."""
        return r'\psi'

    def __call__(self, *args, **kwargs):
        return 0


class Superposition(cp.Sum, QuantumState):
    """A class that represents a superposition of bound states."""

    container_name = 'states'

    def __init__(self, *states):
        """
        Construct a discrete superposition of states.

        :param states: any number of QuantumStates
        """
        super(Superposition, self).__init__(amplitude = 1)
        norm = np.sqrt(sum(s.norm for s in states))
        self.states = list(s / norm for s in states)  # note that the states are implicitly copied here

    @property
    def tuple(self):
        return sum((s.tuple for s in self.states), tuple())


class FreeSphericalWave(QuantumState):
    """A class that represents a free spherical wave."""

    energy = cp.utils.Checked('energy', check = lambda x: x > 0)

    bound = False
    discrete_eigenvalues = False

    def __init__(self, energy = 1 * eV, l = 0, m = 0, amplitude = 1):
        """
        Construct a FreeState from its energy and angular momentum quantum numbers.

        :param energy: energy of the free state
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
        :param amplitude: the probability amplitude of the state
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
        return self.k, self.l, self.m, self.energy

    @property
    def spherical_harmonic(self):
        """Return the SphericalHarmonic for the state's angular momentum quantum numbers."""
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
        """Return a LaTeX-formatted string for the HydrogenCoulombState."""
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

    bound = True
    discrete_eigenvalues = True

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
            self._n = int(n)
        else:
            raise IllegalQuantumState('n ({}) must be an integer greater than zero'.format(n))

    @property
    def l(self):
        """Gets _l."""
        return self._l

    @l.setter
    def l(self, l):
        if int(l) == l and 0 <= l < self.n:
            self._l = int(l)
        else:
            raise IllegalQuantumState('l ({}) must be greater than or equal to zero and less than n ({})'.format(l, self.n))

    @property
    def m(self):
        """Gets _m."""
        return self._m

    @m.setter
    def m(self, m):
        if int(m) == m and -self.l <= m <= self.l:
            self._m = int(m)
        else:
            IllegalQuantumState('|m| (|{}|) must be less than l ({})'.format(m, self.l))

    @property
    def energy(self):
        return -rydberg / (self.n ** 2)

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
        return '|{},{},{}>'.format(*self.tuple)

    @property
    def bra(self):
        """Gets the bra representation of the HydrogenBoundState"""
        return '<{},{},{}|'.format(*self.tuple)

    @property
    def tex_str(self):
        """Gets a LaTeX-formatted string for the HydrogenBoundState."""
        return r'\psi_{{{},{},{}}}'.format(*self.tuple)

    @staticmethod
    def sort_key(state):
        return state.n, state.l, state.m

    def radial_function(self, r):
        """Return the radial part of the wavefunction R(r) evaluated at r."""
        normalization = np.sqrt(((2 / (self.n * bohr_radius)) ** 3) * (sp.math.factorial(self.n - self.l - 1) / (2 * self.n * sp.math.factorial(self.n + self.l))))  # Griffith's normalization
        r_dep = np.exp(-r / (self.n * bohr_radius)) * ((2 * r / (self.n * bohr_radius)) ** self.l)
        lag_poly = special.eval_genlaguerre(self.n - self.l - 1, (2 * self.l) + 1, 2 * r / (self.n * bohr_radius))

        return self.amplitude * normalization * r_dep * lag_poly

    def __call__(self, r, theta, phi):
        """
        Evaluate the hydrogenic bound state wavefunction at a point, or vectorized over an array of points.

        :param r: radial coordinate
        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the wavefunction at (r, theta, phi)
        """
        return self.radial_function(r) * self.spherical_harmonic(theta, phi)


def coulomb_phase_shift(l, k):
    """

    :param l: angular momentum quantum number
    :param k: wavenumber
    :return:
    """
    gamma = 1j / (k * bohr_radius)
    return np.angle(special.gamma(1 + l + gamma))


class HydrogenCoulombState(QuantumState):
    """A class that represents a hydrogenic free state."""

    bound = False
    discrete_eigenvalues = False

    def __init__(self, energy = 1 * eV, l = 0, m = 0, amplitude = 1):
        """
        Construct a HydrogenCoulombState from its energy and angular momentum quantum numbers.

        :param energy: energy of the free state
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
        """
        super(HydrogenCoulombState, self).__init__(amplitude = amplitude)

        if any(int(x) != x for x in (l, m)):
            raise IllegalQuantumState('l and m must be integers')

        if energy >= 0:
            self._energy = energy
        else:
            raise IllegalQuantumState('energy must be greater than zero')

        if l >= 0:
            self._l = int(l)
        else:
            raise IllegalQuantumState('l ({}) must be greater than or equal to zero'.format(l))

        if -l <= m <= l:
            self._m = int(m)
        else:
            raise IllegalQuantumState('m ({}) must be between -l and l ({} to {})'.format(m, -l, l))

    @classmethod
    def from_wavenumber(cls, k, l = 0, m = 0):
        """Construct a HydrogenCoulombState from its wavenumber and angular momentum quantum numbers."""
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
        """Return the SphericalHarmonic for the state's angular momentum quantum numbers."""
        return cp.math.SphericalHarmonic(l = self.l, m = self.m)

    @property
    def tuple(self):
        return self.k, self.l, self.m, self.energy

    def __str__(self):
        return self.ket

    def __repr__(self):
        return '{}(energy = {} eV, k = {} 1/nm, l = {}, m = {}, amplitude = {})'.format(self.__class__.__name__, uround(self.energy, eV, 3), uround(self.k, 1 / nm, 3), self.l, self.m, self.amplitude)

    @property
    def ket(self):
        return '|{} eV, {} 1/nm, {}, {}>'.format(uround(self.energy, eV, 3), uround(self.k, 1 / nm, 3), self.l, self.m)

    @property
    def bra(self):
        return '<{} eV, {} 1/nm, {}, {}|'.format(uround(self.energy, eV, 3), uround(self.k, 1 / nm, 3), self.l, self.m)

    @property
    def tex_str(self):
        """Return a LaTeX-formatted string for the HydrogenCoulombState."""
        return r'\phi_{{{},{},{}}}'.format(uround(self.energy, eV, 3), self.l, self.m)

    def radial_function(self, r):
        x = r / bohr_radius
        epsilon = self.energy / rydberg
        unit_prefactor = np.sqrt(1 / (bohr_radius * rydberg))

        if epsilon > 0:
            kappa = 1j / np.sqrt(epsilon)

            a = self.l + 1 - kappa
            b = 2 * (self.l + 1)
            hgf = ft.partial(mpmath.hyp1f1, a, b)  # construct a partial function, with a and b filled in
            hgf = np.vectorize(hgf, otypes = [np.complex128])  # vectorize using numpy

            A = (kappa ** (-((2 * self.l) + 1))) * special.gamma(1 + self.l + kappa) / special.gamma(kappa - self.l)
            B = A / (1 - np.exp(-twopi / np.sqrt(epsilon)))
            s_prefactor = np.sqrt(B / 2)

            l_prefactor = (2 ** (self.l + 1)) / special.factorial((2 * self.l) + 1)

            prefactor = s_prefactor * l_prefactor * unit_prefactor

            return self.amplitude * prefactor * hgf(2 * x / kappa) * (x ** (self.l + 1)) * np.exp(-x / kappa) / r

        elif epsilon == 0:
            bessel_order = (2 * self.l) + 1
            prefactor = unit_prefactor
            bessel = ft.partial(special.jv, bessel_order)  # construct a partial function with the Bessel function order filled in

            return self.amplitude * prefactor * bessel(np.sqrt(8 * x)) * np.sqrt(x) / r

    def __call__(self, r, theta, phi):
        """
        Evaluate the hydrogenic Coulomb state wavefunction at a point, or vectorized over an array of points.

        :param r: radial coordinate
        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the wavefunction at (r, theta, phi)
        """
        return self.radial_function(r) * self.spherical_harmonic(theta, phi)


class NumericSphericalHarmonicState(QuantumState):
    discrete_eigenvalues = True

    def __init__(self, radial_mesh, l, m, energy, analytic_state, bound = True, amplitude = 1):
        super().__init__(amplitude = amplitude)

        self.radial_mesh = radial_mesh

        self.l = l
        self.m = m
        self.energy = energy

        self.analytic_state = analytic_state

        self.bound = bound

    def __str__(self):
        return str(self.analytic_state)

    def __repr__(self):
        return repr(self.analytic_state)

    @property
    def n(self):
        return self.analytic_state.n

    @property
    def k(self):
        return self.analytic_state.k

    @property
    def tuple(self):
        return self.analytic_state.tuple

    @property
    def ket(self):
        return self.analytic_state.ket

    @property
    def bra(self):
        return self.analytic_state.bra

    @property
    def tex_str(self):
        """Return a LaTeX-formatted string for the NumericSphericalHarmonicState."""
        return self.analytic_state.tex_str

    @property
    def spherical_harmonic(self):
        """Return the SphericalHarmonic for the state's angular momentum quantum numbers."""
        return cp.math.SphericalHarmonic(l = self.l, m = self.m)

    def radial_function(self, r):
        return self.radial_mesh

    def __call__(self, r, theta, phi):
        return self.radial_function(r) * self.spherical_harmonic(theta, phi)


class OneDFreeParticle(QuantumState):
    """A class representing a free particle in one dimension."""

    def __init__(self, wavenumber = twopi / nm, mass = electron_mass, amplitude = 1, dimension_label = 'x'):
        """
        Construct a OneDFreeParticle from a wavenumber and mass.

        :param wavenumber: the wavenumber (2p / wavelength) of the particle
        :param mass: the mass of the particle
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is travelling in
        """
        self.wavenumber = wavenumber
        self.mass = mass
        self.dimension_label = dimension_label

        super(OneDFreeParticle, self).__init__(amplitude = amplitude)

    @classmethod
    def from_energy(cls, energy = 1.50412 * eV, k_sign = 1, mass = electron_mass, amplitude = 1, dimension_label = 'x'):
        """
        Construct a OneDFreeparticle from an energy and a mass. The sign of the desired k-vector must be included as well.

        :param energy: the energy of the particle
        :param k_sign: a prefactor that will be multiplied by the magnitude of the wavenumber determined from the given energy
        :param mass: the mass of the particle
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is travelling in
        :return: a OneDFreeParticle instance
        """
        return cls(k_sign * np.sqrt(2 * mass * energy) / hbar, mass, amplitude = amplitude, dimension_label = dimension_label)

    @property
    def energy(self):
        return ((hbar * self.wavenumber) ** 2) / (2 * self.mass)

    @property
    def momentum(self):
        return hbar * self.wavenumber

    @property
    def tuple(self):
        return self.wavenumber, self.mass, self.dimension_label

    def __str__(self):
        return r'|k = 2pi * {} 1/nm, E = {} eV>'.format(uround(self.wavenumber / twopi, per_nm), uround(self.energy, eV))

    def __repr__(self):
        return cp.utils.field_str(self, 'wavenumber', 'energy', 'mass', 'amplitude')

    @property
    def tex_str(self):
        return r'k = {} 2\pi/{}, E = {} {}'.format(uround(self.wavenumber / twopi, per_nm), r'\mathrm{nm}', uround(self.energy, eV), r'\mathrm{eV}')

    def __call__(self, x):
        """
        Evaluate the free particle wavefunction at a point, or vectorized over an array of points.

        :param x: the distance coordinate along the direction of motion
        :return: the value(s) of the wavefunction at x
        """
        return np.exp(1j * self.wavenumber * x) / np.sqrt(twopi)


class QHOState(QuantumState):
    """A class representing a bound state of the quantum harmonic oscillator."""

    def __init__(self, spring_constant, mass = electron_mass, n = 0, amplitude = 1, dimension_label = 'x'):
        """
        Construct a QHOState from a spring constant, mass, and energy index n.

        :param spring_constant: the spring constant for the quantum harmonic oscillator
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        """
        self.n = n
        self.spring_constant = spring_constant
        self.mass = mass
        self.dimension_label = dimension_label

        super(QHOState, self).__init__(amplitude = amplitude)

    @classmethod
    def from_omega_and_mass(cls, omega, mass = electron_mass, n = 0, amplitude = 1, dimension_label = 'x'):
        """
        Construct a QHOState from an angular frequency, mass, and energy index n.

        :param omega: the fundamental angular frequency of the quantum harmonic oscillator
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        :return: a QHOState instance
        """
        return cls(spring_constant = mass * (omega ** 2), mass = mass, n = n, amplitude = amplitude, dimension_label = dimension_label)

    @classmethod
    def from_QHO_potential_and_mass(cls, qho_potential, mass, n = 0, amplitude = 1, dimension_label = 'x'):
        """
        Construct a QHOState from a HarmonicOscillator, mass, and energy index n.

        :param qho_potential: a HarmonicOscillator instance
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        :return: a QHOState instance
        """
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
        return self.n, self.mass, self.omega, self.dimension_label

    def __call__(self, x):
        """
        Evaluate the quantum harmonic oscillator bound state wavefunction at a point, or vectorized over an array of points.

        Warning: for large enough n (>= ~60) this will fail due to n! overflowing.

        :param x: the distance coordinate along the direction of confinement
        :return: the value(s) of the wavefunction at x
        """
        norm = ((self.mass * self.omega / (pi * hbar)) ** (1 / 4)) / (np.float64(2 ** (self.n / 2)) * np.sqrt(np.float64(sp.math.factorial(self.n))))
        exp = np.exp(-self.mass * self.omega * (x ** 2) / (2 * hbar))
        herm = special.hermite(self.n)(np.sqrt(self.mass * self.omega / hbar) * x)

        # TODO: Stirling's approximation for large enough n in the normalization factor

        return self.amplitude * (norm * exp * herm).astype(np.complex128)


class FiniteSquareWellState(QuantumState):
    """A class representing a bound state of a finite square well."""

    def __init__(self, well_depth, well_width, mass, n = 1, well_center = 0, amplitude = 1, dimension_label = 'x'):
        """
        Construct a FiniteSquareWellState from the well properties, the particle mass, and an energy index.

        :param well_depth: the depth of the potential well
        :param well_width: the full width of the potential well
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param well_center: the center position of the well
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        """
        self.well_depth = well_depth
        self.well_width = well_width
        self.well_center = well_center
        self.mass = mass
        self.n = n
        self.dimension_label = dimension_label

        z_0 = (well_width / 2) * np.sqrt(2 * mass * well_depth) / hbar

        if n - 1 > z_0 // (pi / 2):
            raise IllegalQuantumState('There is no bound state with the given parameters')

        left_bound = (n - 1) * pi / 2
        right_bound = min(z_0, left_bound + (pi / 2))

        # determine the energy of the state by solving a transcendental equation
        if n % 2 != 0:  # n is odd
            z = optimize.brentq(lambda z: np.tan(z) - np.sqrt(((z_0 / z) ** 2) - 1), left_bound, right_bound)
            self.function_inside_well = np.cos
        else:  # n is even
            z = optimize.brentq(lambda z: (1 / np.tan(z)) + np.sqrt(((z_0 / z) ** 2) - 1), left_bound, right_bound)
            self.function_inside_well = np.sin

        self.wavenumber_inside_well = z / (well_width / 2)
        self.energy = (((hbar * self.wavenumber_inside_well) ** 2) / (2 * mass)) - well_depth
        self.wavenumber_outside_well = np.sqrt(-2 * mass * self.energy) / hbar

        self.normalization_factor_inside_well = 1 / np.sqrt((self.well_width / 2) + (1 / self.wavenumber_outside_well))
        self.normalization_factor_outside_well = np.exp(self.wavenumber_outside_well * (self.well_width / 2)) * self.function_inside_well(self.wavenumber_inside_well * (self.well_width / 2)) / np.sqrt((self.well_width / 2) + (1 / self.wavenumber_outside_well))

        super(FiniteSquareWellState, self).__init__(amplitude = amplitude)

    @classmethod
    def from_square_well_potential(cls, finite_square_well_potential, mass, n = 1, amplitude = 1, dimension_label = 'x'):
        """
        Construct a FiniteSquareWellState from a FiniteSquareWell potential, the particle mass, and an energy index.

        :param finite_square_well_potential: a FiniteSquareWell potential.
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        :return: a FiniteSquareWellState instance
        """
        return cls(finite_square_well_potential.potential_depth, finite_square_well_potential.width, mass, n = n, well_center = finite_square_well_potential.center, amplitude = amplitude, dimension_label = dimension_label)

    def __str__(self):
        return self.ket

    def __repr__(self):
        return '{}(n = {}, mass = {}, well_depth = {}, well_width = {}, energy = {}, amplitude = {})'.format(self.__class__.__name__,
                                                                                                             self.n,
                                                                                                             self.mass,
                                                                                                             self.well_depth,
                                                                                                             self.well_width,
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
        return self.well_depth, self.well_width, self.mass, self.n

    @classmethod
    def all_states_of_well_from_parameters(cls, well_depth, well_width, mass, well_center = 0, amplitude = 1):
        """
        Return a list containing all of the bound states of a well.

        The states are ordered in increasing energy.

        :param well_depth: the depth of the potential well
        :param well_width: the full width of the potential well
        :param mass: the mass of the particle
        :param well_center: the center position of the well
        :param amplitude: the probability amplitude of the states
        :return: a list of FiniteSquareWell instances
        """
        states = []
        for n in it.count(1):
            try:
                states.append(cls(well_depth, well_width, mass, n = n, well_center = well_center, amplitude = amplitude))
            except IllegalQuantumState:
                return states

    @classmethod
    def all_states_of_well_from_well(cls, finite_square_well_potential, mass, amplitude = 1):
        """
        Return a list containing all of the bound states of a well.

        :param finite_square_well_potential: a FiniteSquareWell
        :param mass: the mass of the particle
        :param amplitude: the probability amplitude of the states
        :return:
        """
        return cls.all_states_of_well_from_parameters(finite_square_well_potential.potential_depth,
                                                      finite_square_well_potential.width,
                                                      mass,
                                                      well_center = finite_square_well_potential.center,
                                                      amplitude = 1)

    @property
    def left_edge(self):
        """Return the position of the left edge of the well."""
        return self.well_center - (self.well_width / 2)

    @property
    def right_edge(self):
        """Return the position of the right edge of the well."""
        return self.well_center + (self.well_width / 2)

    def __call__(self, x):
        """
        Evaluate the finite square well bound state wavefunction at a point, or vectorized over an array of points.

        :param x: the distance coordinate along the direction of confinement
        :return: the value(s) of the wavefunction at x
        """
        cond = np.greater_equal(x, self.left_edge) * np.less_equal(x, self.right_edge)

        return np.where(cond,
                        self.normalization_factor_inside_well * self.function_inside_well(self.wavenumber_inside_well * x),
                        self.normalization_factor_outside_well * np.exp(-self.wavenumber_outside_well * np.abs(x))).astype(np.complex128)
