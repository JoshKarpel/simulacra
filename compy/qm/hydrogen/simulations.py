import logging

from compy import core, misc, utils
from compy.units import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BoundState:
    """A class that represents a hydrogen bound state."""

    __slots__ = ('_n', '_l', '_m')

    def __init__(self, n = 1, l = 0, m = 0):
        """

        :param n:
        :param l:
        :param m:
        """
        if any(int(x) != x for x in (n, l, m)):
            raise misc.IllegalQuantumState('n, l, and m must be integers')

        if n > 0:
            self._n = n
        else:
            raise misc.IllegalQuantumState('n must be greater than zero')

        if 0 <= l < n:
            self._l = l
        else:
            raise misc.IllegalQuantumState('l must be less than n and greater than or equal to zero')

        if -l <= m <= l:
            self._m = m
        else:
            raise misc.IllegalQuantumState('m must be between -l and l')

    @property
    def n(self):
        return self._n

    @property
    def l(self):
        return self._l

    @property
    def m(self):
        return self._m

    @property
    def energy(self):
        return rydberg / (self.n ** 2)

    @property
    def spherical_harmonic(self):
        return misc.SphericalHarmonic(l = self.l, m = self.m)

    def __str__(self):
        return self.ket

    def __repr__(self):
        return '{}(n={}, l={}, m={})'.format(self.__class__.__name__, self.n, self.l, self.m)

    @property
    def ket(self):
        return '|{},{},{}>'.format(self.n, self.l, self.m)

    @property
    def bra(self):
        return '<{},{},{}|'.format(self.n, self.l, self.m)

    @property
    def plot_str(self):
        """Return a LaTeX-formatted string for the BoundState."""
        return r'\psi_{{{},{},{}}}'.format(self.n, self.l, self.m)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.n == other.n and self.l == other.l and self.m == other.m

    def __hash__(self):
        return hash((self.n, self.l, self.m))

    @utils.memoize()
    def __call__(self, r, theta, phi):
        """
        Evaluate the hydrogen bound state wavefunction at a point, or vectorized over an array of points.

        :param r: radial coordinate
        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the wavefunction at (r, theta, phi)
        """
        raise NotImplementedError


class FreeState:
    """A class that represents a hydrogen free state."""

    __slots__ = ('_energy', '_l', '_m')

    def __init__(self, energy = 1 * eV, l = 0, m = 0):
        """

        :param n:
        :param l:
        :param m:
        """
        if any(int(x) != x for x in (l, m)):
            raise misc.IllegalQuantumState('l and m must be integers')

        if energy > 0:
            self._energy = energy
        else:
            raise misc.IllegalQuantumState('energy must be greater than zero')

        if l >= 0:
            self._l = l
        else:
            raise misc.IllegalQuantumState('l must be greater than or equal to zero')

        if -l <= m <= l:
            self._m = m
        else:
            raise misc.IllegalQuantumState('m must be between -l and l')

    @classmethod
    def from_wavenumber(cls, k, l = 0, m = 0):
        energy = misc.electron_energy_from_wavenumber(k)

        return cls(energy, l, m)

    @property
    def energy(self):
        return self._energy

    @property
    def k(self):
        return misc.electron_wavenumber_from_energy(self.energy)

    @property
    def l(self):
        return self._l

    @property
    def m(self):
        return self._m

    @property
    def spherical_harmonic(self):
        return misc.SphericalHarmonic(l = self.l, m = self.m)

    def __str__(self):
        return self.ket

    def __repr__(self):
        return '{}(T={} eV, k={} 1/nm, l={}, m={})'.format(self.__class__.__name__, np.around(self.energy / eV, 2), np.around(self.k * nm, 2), self.l, self.m)

    @property
    def ket(self):
        return '|{} eV,{} 1/nm, {}, {}>'.format(np.around(self.energy / eV, 2), np.around(self.k * nm, 2), self.l, self.m)

    @property
    def bra(self):
        return '<{} eV,{} 1/nm, {}, {}|'.format(np.around(self.energy / eV, 2), np.around(self.k * nm, 2), self.l, self.m)

    @property
    def plot_str(self):
        """Return a LaTeX-formatted string for the BoundState."""
        return r'\phi_{{{},{},{}}}'.format(np.around(self.energy / eV, 2), self.l, self.m)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.energy == other.energy and self.l == other.l and self.m == other.m

    def __hash__(self):
        return hash((self.energy, self.l, self.m))

    @utils.memoize()
    def __call__(self, r, theta, phi):
        """
        Evaluate the hydrogen bound state wavefunction at a point, or vectorized over an array of points.

        :param r: radial coordinate
        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the wavefunction at (r, theta, phi)
        """
        raise NotImplementedError


class IonizationParameters(core.Parameters):
    def __init__(self, name, file_name = None):
        super(IonizationParameters, self).__init__(name, file_name = file_name)


class IonizationSimulation(core.Simulation):
    def __init__(self, parameters):
        self.parameters = parameters

        super(IonizationSimulation, self).__init__(self.parameters.name, self.parameters.file_name)  # inherit name and file_name from parameters

    def run_simulation(self):
        raise NotImplementedError
