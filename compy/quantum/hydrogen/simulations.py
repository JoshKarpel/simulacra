import logging

import numpy as np

from compy import core, misc, utils
from compy.quantum.core import QuantumMesh
import compy.units as un
import compy.cy as cy


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BoundState:
    """A class that represents a hydrogen bound state."""

    __slots__ = ('_n', '_l', '_m')

    def __init__(self, n = 1, l = 0, m = 0):
        """
        Construct a BoundState from its three quantum numbers (n, l, m).

        :param n: principal quantum number
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
        """
        if any(int(x) != x for x in (n, l, m)):
            raise misc.IllegalQuantumState('n, l, and m must be integers')

        if n > 0:
            self._n = n
        else:
            raise misc.IllegalQuantumState('n ({}) must be greater than zero'.format(n))

        if 0 <= l < n:
            self._l = l
        else:
            raise misc.IllegalQuantumState('l ({}) must be less than n ({}) and greater than or equal to zero'.format(l, n))

        if -l <= m <= l:
            self._m = m
        else:
            raise misc.IllegalQuantumState('m ({}) must be between -l and l ({} to {})'.format(m, -l, l))

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
    def tex_str(self):
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


class BoundStateSuperposition:
    """A class that represents a superposition of bound states."""

    __slots__ = ['state']

    def __init__(self, state, normalize = True):
        """
        Construct a discrete superposition of states.

        If normalize is True the initial amplitudes are rescaled so that the state is normalized.

        :param state: a dict of BoundState:state amplitude (complex number) pairs.
        :param normalize: if True, renormalize the state amplitudes.
        """
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
    def norm(self):
        return np.sum(np.abs(np.array(self.state.values())) ** 2)

    def __abs__(self):
        return self.norm


class FreeState:
    """A class that represents a hydrogen free state."""

    __slots__ = ('_energy', '_l', '_m')

    def __init__(self, energy = 1 * un.eV, l = 0, m = 0):
        """
        Construct a FreeState from its energy and angular momentum quantum numbers.

        :param energy: energy of the free state
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
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
            raise misc.IllegalQuantumState('l ({}) must be greater than or equal to zero'.format(l))

        if -l <= m <= l:
            self._m = m
        else:
            raise misc.IllegalQuantumState('m ({}) must be between -l and l ({} to {})'.format(m, -l, l))

    @classmethod
    def from_wavenumber(cls, k, l = 0, m = 0):
        """Construct a FreeState from its wavenumber and angular momentum quantum numbers."""
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
        return '{}(T={} eV, k={} 1/nm, l={}, m={})'.format(self.__class__.__name__, np.around(self.energy / un.eV, 2), np.around(self.k * un.nm, 2), self.l, self.m)

    @property
    def ket(self):
        return '|{} eV,{} 1/nm, {}, {}>'.format(np.around(self.energy / un.eV, 2), np.around(self.k * un.nm, 2), self.l, self.m)

    @property
    def bra(self):
        return '<{} eV,{} 1/nm, {}, {}|'.format(np.around(self.energy / un.eV, 2), np.around(self.k * un.nm, 2), self.l, self.m)

    @property
    def plot_str(self):
        """Return a LaTeX-formatted string for the BoundState."""
        return r'\phi_{{{},{},{}}}'.format(np.around(self.energy / un.eV, 2), self.l, self.m)

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


class CylindricalFiniteDifferenceMesh(QuantumMesh):
    def __init__(self, parameters, simulation):
        super(CylindricalFiniteDifferenceMesh, self).__init__(parameters, simulation)


class SphericalFiniteDifferenceMesh(QuantumMesh):
    def __init__(self):
        super(SphericalFiniteDifferenceMesh, self).__init__(parameters, simulation)


class CoupledSphericalHarmonicMesh(QuantumMesh):
    def __init__(self):
        super(CoupledSphericalHarmonicMesh, self).__init__(parameters, simulation)


class IonizationParameters(core.Parameters):
    def __init__(self, name, file_name = None):
        super(IonizationParameters, self).__init__(name, file_name = file_name)


class IonizationSimulation(core.Simulation):
    def __init__(self, parameters):
        super(IonizationSimulation, self).__init__(parameters)

    def run_simulation(self):
        raise NotImplementedError
