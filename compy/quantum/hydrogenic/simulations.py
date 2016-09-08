import logging

import numpy as np

from compy import core, math, utils
import compy.quantum.core as qm
from compy.quantum.hydrogenic import animators, potentials
import compy.units as un
import compy.cy as cy


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BoundState:
    """A class that represents a hydrogenic bound state."""

    __slots__ = ('_n', '_l', '_m')

    def __init__(self, n = 1, l = 0, m = 0):
        """
        Construct a BoundState from its three quantum numbers (n, l, m).

        :param n: principal quantum number
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
        """
        if any(int(x) != x for x in (n, l, m)):
            raise qm.IllegalQuantumState('n, l, and m must be integers')

        if n > 0:
            self._n = n
        else:
            raise qm.IllegalQuantumState('n ({}) must be greater than zero'.format(n))

        if 0 <= l < n:
            self._l = l
        else:
            raise qm.IllegalQuantumState('l ({}) must be less than n ({}) and greater than or equal to zero'.format(l, n))

        if -l <= m <= l:
            self._m = m
        else:
            raise qm.IllegalQuantumState('m ({}) must be between -l and l ({} to {})'.format(m, -l, l))

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
    def spherical_harmonic(self):
        return math.SphericalHarmonic(l = self.l, m = self.m)

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
        Evaluate the hydrogenic bound state wavefunction at a point, or vectorized over an array of points.

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


class FreeState:
    """A class that represents a hydrogenic free state."""

    __slots__ = ('_energy', '_l', '_m')

    def __init__(self, energy = 1 * un.eV, l = 0, m = 0):
        """
        Construct a FreeState from its energy and angular momentum quantum numbers.

        :param energy: energy of the free state
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
        """
        if any(int(x) != x for x in (l, m)):
            raise qm.IllegalQuantumState('l and m must be integers')

        if energy > 0:
            self._energy = energy
        else:
            raise qm.IllegalQuantumState('energy must be greater than zero')

        if l >= 0:
            self._l = l
        else:
            raise qm.IllegalQuantumState('l ({}) must be greater than or equal to zero'.format(l))

        if -l <= m <= l:
            self._m = m
        else:
            raise qm.IllegalQuantumState('m ({}) must be between -l and l ({} to {})'.format(m, -l, l))

    @classmethod
    def from_wavenumber(cls, k, l = 0, m = 0):
        """Construct a FreeState from its wavenumber and angular momentum quantum numbers."""
        energy = qm.electron_energy_from_wavenumber(k)

        return cls(energy, l, m)

    @property
    def energy(self):
        return self._energy

    @property
    def k(self):
        return qm.electron_wavenumber_from_energy(self.energy)

    @property
    def l(self):
        return self._l

    @property
    def m(self):
        return self._m

    @property
    def spherical_harmonic(self):
        return math.SphericalHarmonic(l = self.l, m = self.m)

    def __str__(self):
        return self.ket

    def __repr__(self):
        return '{}(T={} eV, k={} 1/nm, l={}, m={})'.format(self.__class__.__name__, un.round(self.energy, un.eV, 3), un.round(self.k, 1 / un.nm, 3), self.l, self.m)

    @property
    def ket(self):
        return '|{} eV,{} 1/nm, {}, {}>'.format(un.round(self.energy, un.eV, 3), un.round(self.k, 1 / un.nm, 3), self.l, self.m)

    @property
    def bra(self):
        return '<{} eV,{} 1/nm, {}, {}|'.format(un.round(self.energy, un.eV, 3), un.round(self.k, 1 / un.nm, 3), self.l, self.m)

    @property
    def plot_str(self):
        """Return a LaTeX-formatted string for the BoundState."""
        return r'\phi_{{{},{},{}}}'.format(un.round(self.energy, un.eV, 3), self.l, self.m)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.energy == other.energy and self.l == other.l and self.m == other.m

    def __hash__(self):
        return hash((self.energy, self.l, self.m))

    @utils.memoize()
    def __call__(self, r, theta, phi):
        """
        Evaluate the hydrogenic bound state wavefunction at a point, or vectorized over an array of points.

        :param r: radial coordinate
        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the wavefunction at (r, theta, phi)
        """
        raise NotImplementedError


class CylindricalSliceFiniteDifferenceMesh(qm.QuantumMesh):
    def __init__(self, parameters, simulation):
        super(CylindricalSliceFiniteDifferenceMesh, self).__init__(parameters, simulation)


class SphericalSliceFiniteDifferenceMesh(qm.QuantumMesh):
    def __init__(self, parameters, simulation):
        super(SphericalSliceFiniteDifferenceMesh, self).__init__(parameters, simulation)


class SphericalHarmonicFiniteDifferenceMesh(qm.QuantumMesh):
    def __init__(self, parameters, simulation):
        super(SphericalHarmonicFiniteDifferenceMesh, self).__init__(parameters, simulation)


class HydrogenicSpecification(core.Specification):
    def __init__(self, name, file_name = None,
                 test_mass = un.electron_mass_reduced, test_charge = un.electron_charge,
                 internal_potential = None, electric_potential = None,
                 time_initial = 0 * un.asec, time_final = 200 * un.asec, time_step = 1 * un.asec,
                 extra_time = 0 * un.asec, extra_time_step = 1 * un.asec):
        super(HydrogenicSpecification, self).__init__(name, file_name = file_name)

        self.test_mass = test_mass
        self.test_charge = test_charge

        self.internal_potential = internal_potential
        self.electric_potential = electric_potential

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        self.extra_time = extra_time
        self.extra_time_step = extra_time_step


class HydrogenicSimulation(core.Simulation):
    def __init__(self, spec):
        super(HydrogenicSimulation, self).__init__(spec)

    def run_simulation(self):
        raise NotImplementedError
