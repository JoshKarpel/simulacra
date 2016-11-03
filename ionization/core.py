import datetime as dt
import functools
import logging
import os
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as nfft
import scipy as sp
import scipy.sparse as sparse
import scipy.special as special
from cycler import cycler

import compy as cp
import compy.cy as cy
from compy.units import *
from ionization import animators, potentials

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class IllegalQuantumState(Exception):
    pass


def electron_energy_from_wavenumber(k):
    return (hbar * k) ** 2 / (2 * electron_mass)


def electron_wavenumber_from_energy(energy):
    return np.sqrt(2 * electron_mass * energy) / hbar


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
            raise IllegalQuantumState('n, l, and m must be integers')

        if n > 0:
            self._n = n
        else:
            raise IllegalQuantumState('n ({}) must be greater than zero'.format(n))

        if 0 <= l < n:
            self._l = l
        else:
            raise IllegalQuantumState('l ({}) must be less than n ({}) and greater than or equal to zero'.format(l, n))

        if -l <= m <= l:
            self._m = m
        else:
            raise IllegalQuantumState('m ({}) must be between -l and l ({} to {})'.format(m, -l, l))

    @property
    def n(self):
        """Gets _n."""
        return self._n

    @property
    def l(self):
        """Gets _l."""
        return self._l

    @property
    def m(self):
        """Gets _m."""
        return self._m

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


class BoundStateSuperposition:
    """A class that represents a superposition of bound states."""

    __slots__ = ('state',)

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

    def __init__(self, energy = 1 * eV, l = 0, m = 0):
        """
        Construct a FreeState from its energy and angular momentum quantum numbers.

        :param energy: energy of the free state
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
        """
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
        energy = electron_energy_from_wavenumber(k)

        return cls(energy, l, m)

    @property
    def energy(self):
        return self._energy

    @property
    def k(self):
        return electron_wavenumber_from_energy(self.energy)

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


class QuantumMesh:
    def __init__(self, simulation):
        self.sim = simulation
        self.spec = simulation.spec

    def __str__(self):
        return '{} for {}'.format(self.__class__.__name__, str(self.sim))

    def __repr__(self):
        return '{}(parameters = {}, simulation = {})'.format(self.__class__.__name__, repr(self.spec), repr(self.sim))

    @property
    def norm(self):
        raise NotImplementedError

    def __abs__(self):
        return self.norm

    def copy(self):
        return deepcopy(self)


class ElectricFieldSpecification(cp.core.Specification):
    """A base Specification for a Simulation with an electric field."""

    def __init__(self, name,
                 mesh_type = None, animator_type = None,
                 test_mass = electron_mass_reduced, test_charge = electron_charge,
                 initial_state = BoundState(1, 0),
                 test_states = tuple(BoundState(n, l) for n in range(5) for l in range(n)),
                 dipole_gauges = tuple('length'),
                 internal_potential = potentials.NuclearPotential(charge = proton_charge) + potentials.RadialImaginaryPotential(center = 20 * bohr_radius, width = 1 * bohr_radius, amplitude = 1 * atomic_electric_potential),
                 electric_potential = None,
                 time_initial = 0 * asec, time_final = 200 * asec, time_step = 1 * asec,
                 extra_time = None, extra_time_step = 1 * asec,
                 checkpoints = False, checkpoint_at = 20, checkpoint_dir = None,
                 animated = False, animation_time = 30, animation_fps = 30, animation_plot_limit = None, animation_normalize = True, animation_log_g = False, animation_overlay_probability_current = False, animation_dir = None,
                 **kwargs):
        super(ElectricFieldSpecification, self).__init__(name, simulation_type = ElectricFieldSimulation, **kwargs)

        if mesh_type is None:
            raise ValueError('{} must have a mesh_type'.format(name))
        self.mesh_type = mesh_type
        self.animator_type = animator_type

        self.test_mass = test_mass
        self.test_charge = test_charge
        self.initial_state = initial_state
        self.test_states = tuple(test_states)  # consume input iterators
        self.dipole_gauges = tuple(dipole_gauges)

        self.internal_potential = internal_potential
        self.electric_potential = electric_potential

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        self.extra_time = extra_time
        self.extra_time_step = extra_time_step

        self.checkpoints = checkpoints
        self.checkpoint_at = checkpoint_at
        self.checkpoint_dir = checkpoint_dir

        self.animated = animated
        self.animation_time = animation_time
        self.animation_fps = animation_fps
        self.animation_plot_limit = animation_plot_limit
        self.animation_normalize = animation_normalize
        self.animation_log_g = animation_log_g
        self.animation_overlay_probability_current = animation_overlay_probability_current
        self.animation_dir = animation_dir

    def info(self):
        checkpoint = ['Checkpointing: ']
        if self.checkpoints:
            if self.animation_dir is not None:
                working_in = self.checkpoint_dir
            else:
                working_in = os.getcwd()
            checkpoint[0] += 'every {} time steps, working in {}'.format(self.checkpoint_at, working_in)
        else:
            checkpoint[0] += 'disabled'

        animation = ['Animation: ']
        if self.animated:
            if self.animation_dir is not None:
                working_in = self.animation_dir
            else:
                working_in = os.getcwd()
            animation[0] += 'enabled, working in {}'.format(working_in)

            animation += ['   Movie Ideal FPS: {} fps'.format(self.animation_fps),
                          '   Normalized: {}'.format(self.animation_normalize),
                          '   Log g: {}'.format(self.animation_log_g),
                          '   Overlay Probability Current: {}'.format(self.animation_overlay_probability_current)]
        else:
            animation[0] += 'disabled'

        time_evolution = ['Time Evolution:',
                          '   Initial State: {}'.format(self.initial_state),
                          '   Initial Time: {} as'.format(uround(self.time_initial, asec, 3)),
                          '   Final Time: {} as'.format(uround(self.time_final, asec, 3)),
                          '   Time Step: {} as'.format(uround(self.time_step, asec, 3))]

        if self.extra_time is not None:
            time_evolution += ['   Extra Time: {} as'.format(uround(self.extra_time, asec, 3)),
                               '   Extra Time Step: {} as'.format(uround(self.extra_time, asec, 3))]

        potentials = ['Potentials:']
        potentials += ['   ' + str(potential) for potential in self.internal_potential]
        if self.electric_potential is not None:
            potentials += ['   ' + str(potential) for potential in self.electric_potential]

        return '\n'.join(checkpoint + animation + time_evolution + potentials)


class CylindricalSliceSpecification(ElectricFieldSpecification):
    def __init__(self, name,
                 z_bound = 20 * bohr_radius, rho_bound = 20 * bohr_radius,
                 z_points = 2 ** 9, rho_points = 2 ** 8,
                 **kwargs):
        super(CylindricalSliceSpecification, self).__init__(name, mesh_type = CylindricalSliceMesh, animator_type = animators.CylindricalSliceAnimator, **kwargs)

        self.z_bound = z_bound
        self.rho_bound = rho_bound
        self.z_points = int(z_points)
        self.rho_points = int(rho_points)

    def info(self):
        mesh = ['Mesh: {}'.format(self.mesh_type.__name__),
                '   Z Boundary: {} Bohr radii'.format(uround(self.z_bound, bohr_radius, 3)),
                '   Z Points: {}'.format(self.z_points),
                '   Z Mesh Spacing: ~{} Bohr radii'.format(uround(2 * self.z_bound / self.z_points, bohr_radius, 3)),
                '   Rho Boundary: {} Bohr radii'.format(uround(self.rho_bound, bohr_radius, 3)),
                '   Rho Points: {}'.format(self.rho_points),
                '   Rho Mesh Spacing: ~{} Bohr radii'.format(uround(self.rho_bound / self.rho_points, bohr_radius, 3)),
                '   Total Mesh Points: {}'.format(int(self.z_points * self.rho_points))]

        return '\n'.join((super(CylindricalSliceSpecification, self).info(), *mesh))


class CylindricalSliceMesh(QuantumMesh):
    def __init__(self, simulation):
        super(CylindricalSliceMesh, self).__init__(simulation)

        self.z = np.linspace(-self.spec.z_bound, self.spec.z_bound, self.spec.z_points)
        self.rho = np.delete(np.linspace(0, self.spec.rho_bound, self.spec.rho_points + 1), 0)

        self.delta_z = self.z[1] - self.z[0]
        self.delta_rho = self.rho[1] - self.rho[0]

        self.rho -= self.delta_rho / 2

        self.z_center_index = int(self.spec.z_points // 2)
        self.z_max = np.max(self.z)
        self.rho_max = np.max(self.rho)

        # self.z_mesh, self.rho_mesh = np.meshgrid(self.z, self.rho, indexing = 'ij')
        self.g_mesh = self.g_for_state(self.spec.initial_state)

        self.mesh_points = len(self.z) * len(self.rho)
        self.matrix_operator_shape = (self.mesh_points, self.mesh_points)
        self.mesh_shape = np.shape(self.r_mesh)

    @property
    @cp.utils.memoize()
    def z_mesh(self):
        return np.meshgrid(self.z, self.rho, indexing = 'ij')[0]

    @property
    @cp.utils.memoize()
    def rho_mesh(self):
        return np.meshgrid(self.z, self.rho, indexing = 'ij')[1]

    @property
    def g_factor(self):
        return np.sqrt(twopi * self.rho_mesh)

    @property
    def psi_mesh(self):
        return self.g_mesh / self.g_factor

    @property
    def r_mesh(self):
        return np.sqrt((self.z_mesh ** 2) + (self.rho_mesh ** 2))

    @property
    def theta_mesh(self):
        return np.arccos(self.z_mesh / self.r_mesh)  # either of these work
        # return np.arctan2(self.rho_mesh, self.z_mesh)  # I have a slight preference for arccos because it will never divide by zero

    @property
    def sin_theta_mesh(self):
        return np.sin(self.theta_mesh)

    @property
    def cos_theta_mesh(self):
        return np.cos(self.theta_mesh)

    def flatten_mesh(self, mesh, flatten_along):
        """Return a mesh flattened along one of the mesh coordinates ('z' or 'rho')."""
        if flatten_along == 'z':
            flat = 'F'
        elif flatten_along == 'rho':
            flat = 'C'
        else:
            raise ValueError("{} is not a valid specifier for flatten_along (valid specifiers: 'z', 'rho')".format(flatten_along))

        return mesh.flatten(flat)

    def wrap_vector(self, vector, wrap_along):
        if wrap_along == 'z':
            wrap = 'F'
        elif wrap_along == 'rho':
            wrap = 'C'
        else:
            raise ValueError("{} is not a valid specifier for wrap_vector (valid specifiers: 'z', 'rho')".format(wrap_along))

        return np.reshape(vector, self.mesh_shape, wrap)

    def inner_product(self, mesh_a = None, mesh_b = None):
        """Inner product between two meshes. If either mesh is None, the state on the g_mesh is used for that state."""
        if mesh_a is None:
            mesh_a = self.g_mesh
        if mesh_b is None:
            mesh_b = self.g_mesh

        return np.einsum('ij,ij->', np.conj(mesh_a), mesh_b) * (self.delta_z * self.delta_rho)

    def state_overlap(self, state_a = None, state_b = None):
        """State overlap between two states. If either state is None, the state on the g_mesh is used for that state."""
        if state_a is None:
            mesh_a = self.g_mesh
        else:
            mesh_a = self.g_for_state(state_a)
        if state_b is None:
            mesh_b = self.g_mesh
        else:
            mesh_b = self.g_for_state(state_b)

        return np.abs(self.inner_product(mesh_a, mesh_b)) ** 2

    @property
    def norm(self):
        return np.abs(self.inner_product())

    @cp.utils.memoize()
    def g_for_state(self, state):
        return self.g_factor * state(self.r_mesh, self.theta_mesh, 0)

    def dipole_moment(self, gauge = 'length'):
        """Get the dipole moment in the specified gauge."""
        if gauge == 'length':
            return self.spec.test_charge * self.inner_product(mesh_b = self.z_mesh * self.g_mesh)
        elif gauge == 'velocity':
            raise NotImplementedError

    @cp.utils.memoize(copy_output = True)
    def get_kinetic_energy_matrix_operators(self):
        """Get the mesh kinetic energy operator matrices for z and rho."""
        z_prefactor = -(hbar ** 2) / (2 * self.spec.test_mass * (self.delta_z ** 2))
        rho_prefactor = -(hbar ** 2) / (2 * self.spec.test_mass * (self.delta_rho ** 2))

        z_diagonal = z_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        z_offdiagonal = z_prefactor * np.array([1 if (z_index + 1) % self.spec.z_points != 0 else 0 for z_index in range(self.mesh_points - 1)], dtype = np.complex128)

        @cp.utils.memoize()
        def c(j):
            return j / np.sqrt((j ** 2) - 0.25)

        rho_diagonal = rho_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        rho_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for rho_index in range(self.mesh_points - 1):
            if (rho_index + 1) % self.spec.rho_points != 0:
                j = (rho_index % self.spec.rho_points) + 1  # get j for the upper diagonal
                rho_offdiagonal[rho_index] = c(j)
        rho_offdiagonal *= rho_prefactor

        z_kinetic = sparse.diags([z_offdiagonal, z_diagonal, z_offdiagonal], offsets = (-1, 0, 1))
        rho_kinetic = sparse.diags([rho_offdiagonal, rho_diagonal, rho_offdiagonal], offsets = (-1, 0, 1))

        return z_kinetic, rho_kinetic

    @cp.utils.memoize(copy_output = True)
    def get_internal_hamiltonian_matrix_operators(self):
        """Get the mesh internal Hamiltonian matrix operators for z and rho."""
        z_kinetic, rho_kinetic = self.get_kinetic_energy_matrix_operators()
        potential_mesh = self.spec.internal_potential(r = self.r_mesh, test_charge = self.spec.test_charge)

        z_kinetic.data[1] += 0.5 * self.flatten_mesh(potential_mesh, 'z')
        rho_kinetic.data[1] += 0.5 * self.flatten_mesh(potential_mesh, 'rho')

        return z_kinetic, rho_kinetic

    def tg_mesh(self, use_abs_g = False):
        hamiltonian_z, hamiltonian_rho = self.get_kinetic_energy_matrix_operators()

        if use_abs_g:
            g_mesh = np.abs(self.g_mesh)
        else:
            g_mesh = self.g_mesh

        g_vector_z = self.flatten_mesh(g_mesh, 'z')
        hg_vector_z = hamiltonian_z.dot(g_vector_z)
        hg_mesh_z = self.wrap_vector(hg_vector_z, 'z')

        g_vector_rho = self.flatten_mesh(g_mesh, 'rho')
        hg_vector_rho = hamiltonian_rho.dot(g_vector_rho)
        hg_mesh_rho = self.wrap_vector(hg_vector_rho, 'rho')

        return hg_mesh_z + hg_mesh_rho

    def hg_mesh(self):
        hamiltonian_z, hamiltonian_rho = self.get_internal_hamiltonian_matrix_operators()

        g_vector_z = self.flatten_mesh(self.g_mesh, 'z')
        hg_vector_z = hamiltonian_z.dot(g_vector_z)
        hg_mesh_z = self.wrap_vector(hg_vector_z, 'z')

        g_vector_rho = self.flatten_mesh(self.g_mesh, 'rho')
        hg_vector_rho = hamiltonian_rho.dot(g_vector_rho)
        hg_mesh_rho = self.wrap_vector(hg_vector_rho, 'rho')

        return hg_mesh_z + hg_mesh_rho

    @property
    def energy_expectation_value(self):
        return np.real(self.inner_product(mesh_b = self.hg_mesh()))

    @cp.utils.memoize(copy_output = True)
    def get_probability_current_matrix_operators(self):
        """Get the mesh probability current operators for z and rho."""
        z_prefactor = hbar / (4 * pi * self.spec.test_mass * self.delta_rho * self.delta_z)
        rho_prefactor = hbar / (4 * pi * self.spec.test_mass * (self.delta_rho ** 2))

        # construct the diagonals of the z probability current matrix operator
        z_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for z_index in range(0, self.mesh_points - 1):
            if (z_index + 1) % self.spec.z_points == 0:  # detect edge of mesh
                z_offdiagonal[z_index] = 0
            else:
                j = z_index // self.spec.z_points
                z_offdiagonal[z_index] = 1 / (j + 0.5)
        z_offdiagonal *= z_prefactor

        @cp.utils.memoize()
        def d(j):
            return 1 / np.sqrt((j ** 2) - 0.25)

        # construct the diagonals of the rho probability current matrix operator
        rho_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for rho_index in range(0, self.mesh_points - 1):
            if (rho_index + 1) % self.spec.rho_points == 0:  # detect edge of mesh
                rho_offdiagonal[rho_index] = 0
            else:
                j = (rho_index % self.spec.rho_points) + 1
                rho_offdiagonal[rho_index] = d(j)
        rho_offdiagonal *= rho_prefactor

        z_current = sparse.diags([-z_offdiagonal, z_offdiagonal], offsets = [-1, 1])
        rho_current = sparse.diags([-rho_offdiagonal, rho_offdiagonal], offsets = [-1, 1])

        return z_current, rho_current

    def get_probability_current_vector_field(self):
        z_current, rho_current = self.get_probability_current_matrix_operators()

        g_vector_z = self.flatten_mesh(self.g_mesh, 'z')
        current_vector_z = z_current.dot(g_vector_z)
        gradient_mesh_z = self.wrap_vector(current_vector_z, 'z')
        current_mesh_z = np.imag(np.conj(self.g_mesh) * gradient_mesh_z)

        g_vector_rho = self.flatten_mesh(self.g_mesh, 'rho')
        current_vector_rho = rho_current.dot(g_vector_rho)
        gradient_mesh_rho = self.wrap_vector(current_vector_rho, 'rho')
        current_mesh_rho = np.imag(np.conj(self.g_mesh) * gradient_mesh_rho)

        return current_mesh_z, current_mesh_rho

    def get_spline_for_mesh(self, mesh):
        return sp.interp.RectBivariateSpline(self.z, self.rho, mesh)

    def evolve(self, time_step):
        """
        Evolve the mesh forward in time by time_step.

        :param time_step:
        :return:
        """
        tau = time_step / (2 * hbar)

        if self.spec.electric_potential is not None:
            electric_potential_energy_mesh = self.spec.electric_potential(t = self.sim.time, distance_along_polarization = self.z_mesh, test_charge = self.spec.test_charge)
        else:
            electric_potential_energy_mesh = np.zeros(self.mesh_shape)

        # add the external potential to the Hamiltonian matrices and multiply them by i * tau to get them ready for the next steps
        hamiltonian_z, hamiltonian_rho = self.get_internal_hamiltonian_matrix_operators()

        hamiltonian_z.data[1] += 0.5 * self.flatten_mesh(electric_potential_energy_mesh, 'z')
        hamiltonian_z *= 1j * tau

        hamiltonian_rho.data[1] += 0.5 * self.flatten_mesh(electric_potential_energy_mesh, 'rho')
        hamiltonian_rho *= 1j * tau

        # STEP 1
        hamiltonian = -1 * hamiltonian_rho
        hamiltonian.data[1] += 1  # add identity to matrix operator
        g_vector = self.flatten_mesh(self.g_mesh, 'rho')
        g_vector = hamiltonian.dot(g_vector)
        self.g_mesh = self.wrap_vector(g_vector, 'rho')

        # STEP 2
        hamiltonian = hamiltonian_z.copy()
        hamiltonian.data[1] += 1  # add identity to matrix operator
        g_vector = self.flatten_mesh(self.g_mesh, 'z')
        g_vector = cy.tdma(hamiltonian, g_vector)

        # STEP 3
        hamiltonian = -1 * hamiltonian_z
        hamiltonian.data[1] += 1  # add identity to matrix operator
        g_vector = hamiltonian.dot(g_vector)
        self.g_mesh = self.wrap_vector(g_vector, 'z')

        # STEP 4
        hamiltonian = hamiltonian_rho.copy()
        hamiltonian.data[1] += 1  # add identity to matrix operator
        g_vector = self.flatten_mesh(self.g_mesh, 'rho')
        g_vector = cy.tdma(hamiltonian, g_vector)
        self.g_mesh = self.wrap_vector(g_vector, 'rho')

    @cp.utils.memoize()
    def get_mesh_slicer(self, distance_from_center = None):
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if distance_from_center is None:
            mesh_slicer = (slice(None, None, 1), slice(None, None, 1))
        else:
            z_lim_points = round(distance_from_center / self.delta_z)
            rho_lim_points = round(distance_from_center / self.delta_rho)
            mesh_slicer = (slice(round(self.z_center_index - z_lim_points), round(self.z_center_index + z_lim_points + 1), 1), slice(0, rho_lim_points + 1, 1))

        return mesh_slicer

    def attach_mesh_to_axis(self, axis, mesh, plot_limit = None):
        color_mesh = axis.pcolormesh(self.z_mesh[self.get_mesh_slicer(plot_limit)] / bohr_radius,
                                     self.rho_mesh[self.get_mesh_slicer(plot_limit)] / bohr_radius,
                                     mesh[self.get_mesh_slicer(plot_limit)],
                                     shading = 'gouraud', cmap = plt.cm.viridis)

        return color_mesh

    def attach_probability_current_to_axis(self, axis, plot_limit = None):
        current_mesh_z, current_mesh_rho = self.get_probability_current_vector_field()

        current_mesh_z *= self.delta_z
        current_mesh_rho *= self.delta_rho

        skip_count = int(self.z_mesh.shape[0] / 50), int(self.z_mesh.shape[1] / 50)
        skip = (slice(None, None, skip_count[0]), slice(None, None, skip_count[1]))
        normalization = np.max(np.sqrt(current_mesh_z ** 2 + current_mesh_rho ** 2)[skip])
        if normalization == 0 or normalization is np.NaN:
            normalization = 1

        quiv = axis.quiver(self.z_mesh[self.get_mesh_slicer(plot_limit)][skip] / bohr_radius,
                           self.rho_mesh[self.get_mesh_slicer(plot_limit)][skip] / bohr_radius,
                           current_mesh_z[self.get_mesh_slicer(plot_limit)][skip] / normalization,
                           current_mesh_rho[self.get_mesh_slicer(plot_limit)][skip] / normalization,
                           pivot = 'middle', units = 'width', scale = 10, scale_units = 'width', width = 0.0015, alpha = 0.5)

        return quiv

    def plot_mesh(self, mesh, name = '', target_dir = None, img_format = 'png', title = None, overlay_probability_current = False, probability_current_time_step = 0, plot_limit = None, **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        color_mesh = self.attach_mesh_to_axis(axis, mesh, plot_limit = plot_limit)
        if overlay_probability_current:
            quiv = self.attach_probability_current_to_axis(axis, plot_limit = plot_limit)

        axis.set_xlabel(r'$z$ (Bohr radii)', fontsize = 15)
        axis.set_ylabel(r'$\rho$ (Bohr radii)', fontsize = 15)
        if title is not None:
            title = axis.set_title(title, fontsize = 15)
            title.set_y(1.05)  # move title up a bit

        # make a colorbar
        cbar = fig.colorbar(mappable = color_mesh, ax = axis)
        cbar.ax.tick_params(labelsize = 10)

        axis.axis('tight')  # removes blank space between color mesh and axes

        axis.grid(True, color = 'pink', linestyle = ':')  # change grid color to make it show up against the colormesh

        axis.tick_params(labelright = True, labeltop = True)  # ticks on all sides
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)  # increase size of tick labels
        axis.tick_params(axis = 'both', which = 'both', length = 0)

        # set upper and lower y ticks to not display to avoid collisions with the x ticks at the edges
        y_ticks = axis.yaxis.get_major_ticks()
        y_ticks[0].label1.set_visible(False)
        y_ticks[0].label2.set_visible(False)
        y_ticks[-1].label1.set_visible(False)
        y_ticks[-1].label2.set_visible(False)

        cp.utils.save_current_figure(name = '{}_{}'.format(self.spec.name, name), target_dir = target_dir, img_format = img_format, **kwargs)

        plt.close()

    def abs_g_squared(self, normalize = False, log = False):
        out = np.abs(self.g_mesh) ** 2
        if normalize:
            out /= np.nanmax(out)
        if log:
            out = np.log10(out)

        return out

    def attach_g_to_axis(self, axis, normalize = False, log = False, plot_limit = None):
        return self.attach_mesh_to_axis(axis, self.abs_g_squared(normalize = normalize, log = log), plot_limit = plot_limit)

    def update_g_mesh(self, colormesh, normalize = False, log = False, plot_limit = None):
        new_mesh = self.abs_g_squared(normalize = normalize, log = log)[self.get_mesh_slicer(plot_limit)]

        colormesh.set_array(new_mesh.ravel())

    def plot_g(self, normalize = True, name_postfix = '', **kwargs):
        """Plot |g|^2. kwargs are for plot_mesh."""
        title = ''
        if normalize:
            title = r'Normalized '
        title += r'$|g|^2$'
        name = 'g' + name_postfix

        self.plot_mesh(self.abs_g_squared(normalize = normalize), name = name, title = title, **kwargs)

    def abs_psi_squared(self, normalize = False, log = False):
        out = np.abs(self.psi_mesh) ** 2
        if normalize:
            out /= np.nanmax(out)
        if log:
            out = np.log10(out)

        return out

    def attach_psi_to_axis(self, axis, normalize = False, log = False, plot_limit = None):
        return self.attach_mesh_to_axis(axis, self.abs_psi_squared(normalize = normalize, log = log), plot_limit = plot_limit)

    def update_psi_mesh(self, colormesh, normalize = False, log = False, plot_limit = None):
        new_mesh = self.abs_psi_squared(normalize = normalize, log = log)[self.get_mesh_slicer(plot_limit)]

        colormesh.set_array(new_mesh.ravel())

    def plot_psi(self, normalize = True, name_postfix = '', **kwargs):
        """Plot |psi|^2. kwargs are for plot_mesh."""
        title = ''
        if normalize:
            title = r'Normalized '
        title += r'$|\psi|^2$'
        name = 'psi' + name_postfix

        self.plot_mesh(self.abs_psi_squared(normalize = normalize), name = name, title = title, **kwargs)


class SphericalSliceSpecification(ElectricFieldSpecification):
    def __init__(self, name,
                 r_bound = 20 * bohr_radius,
                 r_points = 2 ** 10, theta_points = 2 ** 10,
                 **kwargs):
        super(SphericalSliceSpecification, self).__init__(name, mesh_type = SphericalSliceMesh, animator_type = animators.SphericalSliceAnimator, **kwargs)

        self.r_bound = r_bound

        self.r_points = int(r_points)
        self.theta_points = int(theta_points)

    def info(self):
        mesh = ['Mesh: {}'.format(self.mesh_type.__name__),
                '   R Boundary: {} Bohr radii'.format(uround(self.r_bound, bohr_radius, 3)),
                '   R Points: {}'.format(self.r_points),
                '   R Mesh Spacing: ~{} Bohr radii'.format(uround(self.r_bound / self.r_points, bohr_radius, 3)),
                '   Theta Points: {}'.format(self.theta_points),
                '   Theta Mesh Spacing: ~{} rad | ~{} deg'.format(round(pi / self.theta_points, 3), round(180 / self.theta_points, 3)),
                '   Maximum Adjacent-Point Spacing: ~{} Bohr radii'.format(uround(pi * self.r_bound / self.theta_points, bohr_radius, 3)),
                '   Total Mesh Points: {}'.format(int(self.r_points * self.theta_points))]

        return '\n'.join((super(SphericalSliceSpecification, self).info(), *mesh))


class SphericalSliceMesh(QuantumMesh):
    def __init__(self, simulation):
        super(SphericalSliceMesh, self).__init__(simulation)

        self.r = np.linspace(0, self.spec.r_bound, self.spec.r_points)
        self.theta = np.delete(np.linspace(0, pi, self.spec.theta_points + 1), 0)

        self.delta_r = self.r[1] - self.r[0]
        self.delta_theta = self.theta[1] - self.theta[0]

        self.r += self.delta_r / 2
        self.theta -= self.delta_theta / 2

        self.r_max = np.max(self.r)

        # self.r_mesh, self.theta_mesh = np.meshgrid(self.r, self.theta, indexing = 'ij')
        self.g_mesh = self.g_for_state(self.spec.initial_state)

        self.mesh_points = len(self.r) * len(self.theta)
        self.matrix_operator_shape = (self.mesh_points, self.mesh_points)
        self.mesh_shape = np.shape(self.r_mesh)

    @property
    @cp.utils.memoize()
    def r_mesh(self):
        return np.meshgrid(self.r, self.theta, indexing = 'ij')[0]

    @property
    @cp.utils.memoize()
    def theta_mesh(self):
        return np.meshgrid(self.r, self.theta, indexing = 'ij')[1]

    @property
    def g_factor(self):
        return np.sqrt(twopi * np.sin(self.theta_mesh)) * self.r_mesh

    @property
    def psi_mesh(self):
        return self.g_mesh / self.g_factor

    @property
    def z_mesh(self):
        return self.r_mesh * np.cos(self.theta_mesh)

    def flatten_mesh(self, mesh, flatten_along):
        """Return a mesh flattened along one of the mesh coordinates ('theta' or 'r')."""
        if flatten_along == 'r':
            flat = 'F'
        elif flatten_along == 'theta':
            flat = 'C'
        else:
            raise ValueError("{} is not a valid specifier for flatten_mesh (valid specifiers: 'r', 'theta')".format(flatten_along))

        return mesh.flatten(flat)

    def wrap_vector(self, mesh, wrap_along):
        if wrap_along == 'r':
            wrap = 'F'
        elif wrap_along == 'theta':
            wrap = 'C'
        else:
            raise ValueError("{} is not a valid specifier for wrap_vector (valid specifiers: 'r', 'theta')".format(wrap_along))

        return np.reshape(mesh, self.mesh_shape, wrap)

    def inner_product(self, mesh_a = None, mesh_b = None):
        """Inner product between two meshes. If either mesh is None, the state on the g_mesh is used for that state."""
        if mesh_a is None:
            mesh_a = self.g_mesh
        if mesh_b is None:
            mesh_b = self.g_mesh

        return np.einsum('ij,ij->', np.conj(mesh_a), mesh_b) * (self.delta_r * self.delta_theta)

    def state_overlap(self, state_a = None, state_b = None):
        """State overlap between two states. If either state is None, the state on the g_mesh is used for that state."""
        if state_a is None:
            mesh_a = self.g_mesh
        else:
            mesh_a = self.g_for_state(state_a)
        if state_b is None:
            mesh_b = self.g_mesh
        else:
            mesh_b = self.g_for_state(state_b)

        return np.abs(self.inner_product(mesh_a, mesh_b)) ** 2

    @property
    def norm(self):
        return np.abs(self.inner_product())

    @cp.utils.memoize()
    def g_for_state(self, state):
        return self.g_factor * state(self.r_mesh, self.theta_mesh, 0)

    def dipole_moment(self, gauge = 'length'):
        """Get the dipole moment in the specified gauge."""
        if gauge == 'length':
            return self.spec.test_charge * self.inner_product(mesh_b = self.z_mesh * self.g_mesh)
        elif gauge == 'velocity':
            raise NotImplementedError

    @cp.utils.memoize(copy_output = True)
    def get_kinetic_energy_matrix_operators(self):
        r_prefactor = -(hbar ** 2) / (2 * electron_mass_reduced * (self.delta_r ** 2))
        theta_prefactor = -(hbar ** 2) / (2 * electron_mass_reduced * ((self.delta_r * self.delta_theta) ** 2))

        r_diagonal = r_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        r_offdiagonal = r_prefactor * np.array([1 if (z_index + 1) % self.spec.r_points != 0 else 0 for z_index in range(self.mesh_points - 1)], dtype = np.complex128)

        @cp.utils.memoize()
        def theta_j_prefactor(x):
            return 1 / (x + 0.5) ** 2

        @cp.utils.memoize()
        def sink(x):
            return np.sin(x * self.delta_theta)

        @cp.utils.memoize()
        def sqrt_sink_ratio(x_num, x_den):
            return np.sqrt(sink(x_num) / sink(x_den))

        @cp.utils.memoize()
        def cotank(x):
            return 1 / np.tan(x * self.delta_theta)

        theta_diagonal = (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        for theta_index in range(self.mesh_points):
            j = theta_index // self.spec.theta_points
            theta_diagonal[theta_index] *= theta_j_prefactor(j)
        theta_diagonal *= theta_prefactor

        theta_upper_diagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        theta_lower_diagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for theta_index in range(self.mesh_points - 1):
            if (theta_index + 1) % self.spec.theta_points != 0:
                j = theta_index // self.spec.theta_points
                k = theta_index % self.spec.theta_points
                k_p = k + 1  # add 1 because the entry for the lower diagonal is really for the next point (k -> k + 1), not this one
                theta_upper_diagonal[theta_index] = theta_j_prefactor(j) * (1 + (self.delta_theta / 2) * cotank(k + 0.5)) * sqrt_sink_ratio(k + 0.5, k + 1.5)
                theta_lower_diagonal[theta_index] = theta_j_prefactor(j) * (1 - (self.delta_theta / 2) * cotank(k_p + 0.5)) * sqrt_sink_ratio(k_p + 0.5, k_p - 0.5)
        theta_upper_diagonal *= theta_prefactor
        theta_lower_diagonal *= theta_prefactor

        r_kinetic = sparse.diags([r_offdiagonal, r_diagonal, r_offdiagonal], offsets = (-1, 0, 1))
        theta_kinetic = sparse.diags([theta_lower_diagonal, theta_diagonal, theta_upper_diagonal], offsets = (-1, 0, 1))

        return r_kinetic, theta_kinetic

    @cp.utils.memoize(copy_output = True)
    def get_internal_hamiltonian_matrix_operators(self):
        r_kinetic, theta_kinetic = self.get_kinetic_energy_matrix_operators()
        potential_mesh = self.spec.internal_potential(r = self.r_mesh, test_charge = self.spec.test_charge)

        r_kinetic.data[1] += 0.5 * self.flatten_mesh(potential_mesh, 'r')
        theta_kinetic.data[1] += 0.5 * self.flatten_mesh(potential_mesh, 'theta')

        return r_kinetic, theta_kinetic

    def tg_mesh(self, use_abs_g = False):
        hamiltonian_r, hamiltonian_theta = self.get_kinetic_energy_matrix_operators()

        if use_abs_g:
            g_mesh = np.abs(self.g_mesh)
        else:
            g_mesh = self.g_mesh

        g_vector_r = self.flatten_mesh(g_mesh, 'r')
        hg_vector_r = hamiltonian_r.dot(g_vector_r)
        hg_mesh_r = self.wrap_vector(hg_vector_r, 'r')

        g_vector_theta = self.flatten_mesh(g_mesh, 'theta')
        hg_vector_theta = hamiltonian_theta.dot(g_vector_theta)
        hg_mesh_theta = self.wrap_vector(hg_vector_theta, 'theta')

        return hg_mesh_r + hg_mesh_theta

    def hg_mesh(self):
        hamiltonian_r, hamiltonian_theta = self.get_internal_hamiltonian_matrix_operators()

        g_vector_r = self.flatten_mesh(self.g_mesh, 'r')
        hg_vector_r = hamiltonian_r.dot(g_vector_r)
        hg_mesh_r = self.wrap_vector(hg_vector_r, 'r')

        g_vector_theta = self.flatten_mesh(self.g_mesh, 'theta')
        hg_vector_theta = hamiltonian_theta.dot(g_vector_theta)
        hg_mesh_theta = self.wrap_vector(hg_vector_theta, 'theta')

        return hg_mesh_r + hg_mesh_theta

    @property
    def energy_expectation_value(self):
        return np.real(self.inner_product(mesh_b = self.hg_mesh()))

    @cp.utils.memoize()
    def get_probability_current_matrix_operators(self):
        raise NotImplementedError

    def get_probability_current_vector_field(self):
        raise NotImplementedError

    def get_spline_for_mesh(self, mesh):
        return sp.interp.RectBivariateSpline(self.r, self.theta, mesh)

    def evolve(self, delta_t):
        tau = delta_t / (2 * hbar)

        if self.spec.electric_potential is not None:
            electric_potential_energy_mesh = self.spec.electric_potential(t = self.sim.time, distance_along_polarization = self.z_mesh, test_charge = self.spec.test_charge)
        else:
            electric_potential_energy_mesh = np.zeros(self.mesh_shape)

        # add the external potential to the Hamiltonian matrices and multiply them by i * tau to get them ready for the next steps
        hamiltonian_r, hamiltonian_theta = self.get_internal_hamiltonian_matrix_operators()
        hamiltonian_r = hamiltonian_r.copy()
        hamiltonian_theta = hamiltonian_theta.copy()

        hamiltonian_r.data[1] += 0.5 * self.flatten_mesh(electric_potential_energy_mesh, 'r')
        hamiltonian_r *= 1j * tau

        hamiltonian_theta.data[1] += 0.5 * self.flatten_mesh(electric_potential_energy_mesh, 'theta')
        hamiltonian_theta *= 1j * tau

        # STEP 1
        hamiltonian = -1 * hamiltonian_theta
        hamiltonian.data[1] += 1  # add identity to matrix operator
        g_vector = self.flatten_mesh(self.g_mesh, 'theta')
        g_vector = hamiltonian.dot(g_vector)
        self.g_mesh = self.wrap_vector(g_vector, 'theta')

        # STEP 2
        hamiltonian = hamiltonian_r.copy()
        hamiltonian.data[1] += 1  # add identity to matrix operator
        g_vector = self.flatten_mesh(self.g_mesh, 'r')
        g_vector = cy.tdma(hamiltonian, g_vector)

        # STEP 3
        hamiltonian = -1 * hamiltonian_r
        hamiltonian.data[1] += 1  # add identity to matrix operator
        g_vector = hamiltonian.dot(g_vector)
        self.g_mesh = self.wrap_vector(g_vector, 'r')

        # STEP 4
        hamiltonian = hamiltonian_theta.copy()
        hamiltonian.data[1] += 1  # add identity to matrix operator
        g_vector = self.flatten_mesh(self.g_mesh, 'theta')
        g_vector = cy.tdma(hamiltonian, g_vector)
        self.g_mesh = self.wrap_vector(g_vector, 'theta')

    @cp.utils.memoize()
    def get_mesh_slicer(self, distance_from_center = None):
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if distance_from_center is None:
            mesh_slicer = slice(None, None, 1)
        else:
            r_lim_points = round(distance_from_center / self.delta_r)
            mesh_slicer = slice(0, r_lim_points + 1, 1)

        return mesh_slicer

    def attach_mesh_to_axis(self, axis, mesh, plot_limit = None):
        color_mesh = axis.pcolormesh(self.theta_mesh[self.get_mesh_slicer(plot_limit)],
                                     self.r_mesh[self.get_mesh_slicer(plot_limit)] / bohr_radius,
                                     mesh[self.get_mesh_slicer(plot_limit)],
                                     shading = 'gouraud', cmap = plt.cm.viridis)
        color_mesh_mirror = axis.pcolormesh(-self.theta_mesh[self.get_mesh_slicer(plot_limit)] + (2 * pi),
                                            self.r_mesh[self.get_mesh_slicer(plot_limit)] / bohr_radius,
                                            mesh[self.get_mesh_slicer(plot_limit)],
                                            shading = 'gouraud', cmap = plt.cm.viridis)  # another colormesh, mirroring the first mesh onto pi to 2pi

        return color_mesh, color_mesh_mirror

    def attach_probability_current_to_axis(self, axis, plot_limit = None):
        raise NotImplementedError

    def plot_mesh(self, mesh,
                  name = '', target_dir = None, img_format = 'png', title = None,
                  overlay_probability_current = False, probability_current_time_step = 0, plot_limit = None,
                  **kwargs):
        plt.close()  # close any old figures

        fig = plt.figure(figsize = (7, 7), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111, projection = 'polar')

        color_mesh, color_mesh_mirror = self.attach_mesh_to_axis(axis, mesh, plot_limit = plot_limit)
        if overlay_probability_current:
            quiv = self.attach_probability_current_to_axis(axis, plot_limit = plot_limit)

        if title is not None:
            title = axis.set_title(title, fontsize = 15)
            title.set_x(.03)  # move title to the upper left corner
            title.set_y(.97)

        # make a colorbar
        cbar_axis = fig.add_axes([1.01, .1, .04, .8])  # add a new axis for the cbar so that the old axis can stay square
        cbar = plt.colorbar(mappable = color_mesh, cax = cbar_axis)
        cbar.ax.tick_params(labelsize = 10)

        axis.set_rmax((self.r_max - (self.delta_r / 2)) / bohr_radius)

        axis.grid(True, color = 'pink', linestyle = ':')  # change grid color to make it show up against the colormesh
        axis.set_thetagrids(np.arange(0, 360, 30), frac = 1.05)

        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)  # increase size of tick labels
        axis.tick_params(axis = 'y', which = 'major', colors = 'pink', pad = 3)  # make r ticks a color that shows up against the colormesh
        axis.tick_params(axis = 'both', which = 'both', length = 0)

        axis.set_rlabel_position(10)
        last_r_label = axis.get_yticklabels()[-1]
        last_r_label.set_color('black')  # last r tick is outside the colormesh, so make it black again

        cp.utils.save_current_figure(name = '{}_{}'.format(self.spec.name, name), target_dir = target_dir, img_format = img_format, **kwargs)

        plt.close()

    def abs_g_squared(self, normalize = False, log = False):
        out = np.abs(self.g_mesh) ** 2
        if normalize:
            out /= np.nanmax(out)
        if log:
            out = np.log10(out)

        return out

    def attach_g_to_axis(self, axis, normalize = False, log = False, plot_limit = None):
        return self.attach_mesh_to_axis(axis, self.abs_g_squared(normalize = normalize, log = log), plot_limit = plot_limit)

    def update_g_mesh(self, colormesh, normalize = False, log = False, plot_limit = None):
        new_mesh = self.abs_g_squared(normalize = normalize, log = log)[self.get_mesh_slicer(plot_limit)]

        colormesh.set_array(new_mesh.ravel())

    def plot_g(self, normalize = True, name_postfix = '', **kwargs):
        """Plot |g|^2. kwargs are for plot_mesh."""
        title = ''
        if normalize:
            title = r'Normalized '
        title += r'$|g|^2$'
        name = 'g' + name_postfix

        self.plot_mesh(self.abs_g_squared(normalize = normalize), name = name, title = title, **kwargs)

    def abs_psi_squared(self, normalize = False, log = False):
        out = np.abs(self.psi_mesh) ** 2
        if normalize:
            out /= np.nanmax(out)
        if log:
            out = np.log10(out)

        return out

    def attach_psi_to_axis(self, axis, normalize = False, log = False, plot_limit = None):
        return self.attach_mesh_to_axis(axis, self.abs_psi_squared(normalize = normalize, log = log), plot_limit = plot_limit)

    def update_psi_mesh(self, colormesh, normalize = False, log = False, plot_limit = None):
        new_mesh = self.abs_psi_squared(normalize = normalize, log = log)[self.get_mesh_slicer(plot_limit)]

        colormesh.set_array(new_mesh.ravel())

    def plot_psi(self, normalize = True, name_postfix = '', **kwargs):
        """Plot |psi|^2. kwargs are for plot_mesh."""
        title = ''
        if normalize:
            title = r'Normalized '
        title += r'$|\psi|^2$'
        name = 'psi' + name_postfix

        self.plot_mesh(self.abs_psi_squared(normalize = normalize), name = name, title = title, **kwargs)


class SphericalHarmonicSpecification(ElectricFieldSpecification):
    def __init__(self, name,
                 r_bound = 20 * bohr_radius,
                 r_points = 2 ** 9,
                 spherical_harmonics_max_l = 20,
                 mesh_type = SphericalHarmonicMesh,
                 **kwargs):
        super(SphericalHarmonicSpecification, self).__init__(name, mesh_type = mesh_type, **kwargs)

        self.r_bound = r_bound
        self.r_points = int(r_points)
        self.spherical_harmonics_max_l = spherical_harmonics_max_l

    def info(self):
        mesh = ['Mesh: {}'.format(self.mesh_type.__name__),
                '   R Boundary: {} Bohr radii'.format(uround(self.r_bound, bohr_radius, 3)),
                '   R Points: {}'.format(self.r_points),
                '   R Mesh Spacing: ~{} Bohr radii'.format(uround(self.r_bound / self.r_points, bohr_radius, 3)),
                '   Spherical Harmonics: {}'.format(self.spherical_harmonics_max_l + 1),
                '   Total Mesh Points: {}'.format(self.r_points * (self.spherical_harmonics_max_l + 1))]

        return '\n'.join((super(SphericalHarmonicSpecification, self).info(), *mesh))


class SphericalHarmonicMesh(QuantumMesh):
    def __init__(self, simulation):
        super(SphericalHarmonicMesh, self).__init__(simulation)

        self.r = np.linspace(0, self.spec.r_bound, self.spec.r_points)
        self.delta_r = self.r[1] - self.r[0]
        self.r += self.delta_r / 2
        self.r_max = np.max(self.r)

        self.l = np.array(range(0, self.spec.spherical_harmonics_max_l + 1))
        self.spherical_harmonics = tuple(cp.math.SphericalHarmonic(l, 0) for l in range(self.spec.spherical_harmonics_max_l + 1))
        self.l_points = len(self.l)
        self.spec.l_points = self.l_points

        self.l_mesh, self.r_mesh = np.meshgrid(self.l, self.r, indexing = 'ij')

        self.mesh_points = len(self.r) * len(self.l)
        self.mesh_shape = np.shape(self.r_mesh)

        self.g_mesh = self.g_for_state(self.spec.initial_state)

    @property
    def g_factor(self):
        return self.r

    def flatten_mesh(self, mesh, flatten_along):
        """Return a mesh flattened along one of the mesh coordinates ('theta' or 'r')."""
        if flatten_along == 'l':
            flat = 'F'
        elif flatten_along == 'r':
            flat = 'C'
        else:
            raise ValueError("{} is not a valid specifier for flatten_mesh (valid specifiers: 'l', 'r')".format(flatten_along))

        return mesh.flatten(flat)

    def wrap_vector(self, mesh, wrap_along):
        if wrap_along == 'l':
            wrap = 'F'
        elif wrap_along == 'r':
            wrap = 'C'
        else:
            raise ValueError("{} is not a valid specifier for wrap_vector (valid specifiers: 'l', 'r')".format(wrap_along))

        return np.reshape(mesh, self.mesh_shape, wrap)

    def inner_product(self, mesh_a = None, mesh_b = None):
        if mesh_a is None:
            mesh_a = self.g_mesh
        if mesh_b is None:
            mesh_b = self.g_mesh

        return np.einsum('ij,ij->', np.conj(mesh_a), mesh_b) * self.delta_r

    def state_overlap(self, state_a = None, state_b = None):
        if state_a is None:
            mesh_a = self.g_mesh
        else:
            mesh_a = self.g_for_state(state_a)
        if state_b is None:
            mesh_b = self.g_mesh
        else:
            mesh_b = self.g_for_state(state_b)

        return np.abs(self.inner_product(mesh_a, mesh_b)) ** 2

    @property
    def norm(self):
        return np.abs(self.inner_product())

    @cp.utils.memoize(copy_output = True)
    def g_for_state(self, state):
        g = np.zeros(self.mesh_shape)

        g[state.l, :] = state.radial_part(self.r) * self.g_factor

        return g

    @cp.utils.memoize(copy_output = True)
    def get_kinetic_energy_matrix_operators(self):
        r_prefactor = -(hbar ** 2) / (2 * electron_mass_reduced * (self.delta_r ** 2))
        l_prefactor = 1

        r_diagonal = r_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        r_offdiagonal = r_prefactor * np.ones(self.mesh_points - 1, dtype = np.complex128)

        @cp.utils.memoize()
        def a(l):
            return l * np.sqrt(1 / (4 * (l ** 2) - 1))

        l_diagonal = np.zeros(self.mesh_points, dtype = np.complex128)
        l_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for l_index in range(self.mesh_points - 1):
            if (l_index + 1) % self.spec.l_points != 0:
                l = (l_index % self.spec.l_points) + 1
                l_offdiagonal[l_index] = a(l)
        l_offdiagonal *= l_prefactor

        effective_potential_mesh = ((hbar ** 2) / (2 * electron_mass_reduced)) * self.l_mesh * (self.l_mesh + 1) / (self.r_mesh ** 2)
        r_diagonal += self.flatten_mesh(effective_potential_mesh, 'r')

        r_kinetic = sparse.diags([r_offdiagonal, r_diagonal, r_offdiagonal], offsets = (-1, 0, 1))
        l_kinetic = sparse.diags([l_offdiagonal, l_diagonal, l_offdiagonal], offsets = (-1, 0, 1))

        return r_kinetic, l_kinetic

    @cp.utils.memoize(copy_output = True)
    def get_internal_hamiltonian_matrix_operators(self):
        r_kinetic, l_kinetic = self.get_kinetic_energy_matrix_operators()
        potential_mesh = self.spec.internal_potential(r = self.r_mesh, test_charge = self.spec.test_charge)

        r_kinetic.data[1] += self.flatten_mesh(potential_mesh, 'r')

        return r_kinetic, l_kinetic

    def tg_mesh(self, use_abs_g = False):
        hamiltonian_r, hamiltonian_l = self.get_kinetic_energy_matrix_operators()

        if use_abs_g:
            g_mesh = np.abs(self.g_mesh)
        else:
            g_mesh = self.g_mesh

        g_vector_r = self.flatten_mesh(g_mesh, 'r')
        hg_vector_r = hamiltonian_r.dot(g_vector_r)
        hg_mesh_r = self.wrap_vector(hg_vector_r, 'r')

        g_vector_l = self.flatten_mesh(g_mesh, 'l')
        hg_vector_l = hamiltonian_l.dot(g_vector_l)
        hg_mesh_l = self.wrap_vector(hg_vector_l, 'l')

        return hg_mesh_r + hg_mesh_l

    def hg_mesh(self):
        hamiltonian_r, hamiltonian_l = self.get_internal_hamiltonian_matrix_operators()

        g_vector_r = self.flatten_mesh(self.g_mesh, 'r')
        hg_vector_r = hamiltonian_r.dot(g_vector_r)
        hg_mesh_r = self.wrap_vector(hg_vector_r, 'r')

        # g_vector_l = self.flatten_mesh(self.g_mesh, 'l')
        # hg_vector_l = hamiltonian_l.dot(g_vector_l)
        # hg_mesh_l = self.wrap_vector(hg_vector_l, 'l')

        # return hg_mesh_r + hg_mesh_l
        return hg_mesh_r

    @property
    def energy_expectation_value(self):
        return np.real(self.inner_product(mesh_b = self.hg_mesh()))

    @cp.utils.memoize()
    def get_probability_current_matrix_operators(self):
        raise NotImplementedError

    def get_probability_current_vector_field(self):
        raise NotImplementedError

    def evolve(self, delta_t):
        tau = delta_t / (2 * hbar)

        if self.spec.electric_potential is not None:
            electric_field_amplitude = self.spec.electric_potential.get_amplitude(self.sim.time) * self.flatten_mesh(self.r_mesh, 'l')
        else:
            electric_field_amplitude = 0
        l_multiplier = self.spec.test_charge * electric_field_amplitude

        hamiltonian_r, hamiltonian_l = self.get_internal_hamiltonian_matrix_operators()
        hamiltonian_r *= 1j * tau
        hamiltonian_l.data[0] *= 1j * tau * l_multiplier
        hamiltonian_l.data[2] *= 1j * tau * l_multiplier

        # STEP 1
        hamiltonian = -1 * hamiltonian_l
        hamiltonian.data[1] += 1  # add identity to matrix operator
        g_vector = self.flatten_mesh(self.g_mesh, 'l')
        g_vector = hamiltonian.dot(g_vector)
        self.g_mesh = self.wrap_vector(g_vector, 'l')

        # STEP 2
        hamiltonian = hamiltonian_r.copy()
        hamiltonian.data[1] += 1  # add identity to matrix operator
        g_vector = self.flatten_mesh(self.g_mesh, 'r')
        g_vector = cy.tdma(hamiltonian, g_vector)

        # STEP 3
        hamiltonian = -1 * hamiltonian_r
        hamiltonian.data[1] += 1  # add identity to matrix operator
        g_vector = hamiltonian.dot(g_vector)
        self.g_mesh = self.wrap_vector(g_vector, 'r')

        # STEP 4
        hamiltonian = hamiltonian_l.copy()
        hamiltonian.data[1] += 1  # add identity to matrix operator
        g_vector = self.flatten_mesh(self.g_mesh, 'l')
        g_vector = cy.tdma(hamiltonian, g_vector)
        self.g_mesh = self.wrap_vector(g_vector, 'l')


class LagrangianSphericalHarmonicMesh(SphericalHarmonicMesh):
    def __init__(self, spec):
        super(LagrangianSphericalHarmonicMesh, self).__init__(spec)

    def evolve(self, delta_t):
        raise NotImplementedError


class ElectricFieldSimulation(cp.core.Simulation):
    def __init__(self, spec):
        super(ElectricFieldSimulation, self).__init__(spec)

        self.mesh = None
        self.animator = None

        self.initialize_mesh()

        total_time = self.spec.time_final - self.spec.time_initial
        self.times = np.linspace(self.spec.time_initial, self.spec.time_final, int(total_time / self.spec.time_step) + 1)
        if self.spec.extra_time is not None:
            extra_times = np.delete(np.linspace(self.spec.time_final, self.spec.time_final + self.spec.extra_time, (self.spec.extra_time / self.spec.extra_time_step) + 1), 0)
            self.times = np.concatenate((self.times, extra_times))
        self.time_index = 0
        self.time_steps = len(self.times)

        # simulation data storage
        self.norm_vs_time = np.zeros(self.time_steps) * np.NaN
        self.energy_expectation_value_vs_time_internal = np.zeros(self.time_steps) * np.NaN
        self.inner_products_vs_time = {state: np.zeros(self.time_steps, dtype = np.complex128) * np.NaN for state in self.spec.test_states}
        self.electric_field_amplitude_vs_time = np.zeros(self.time_steps) * np.NaN
        self.electric_dipole_moment_vs_time = {gauge: np.zeros(self.time_steps, dtype = np.complex128) * np.NaN for gauge in self.spec.dipole_gauges}

        if self.spec.animated:
            self.animator = self.spec.animator_type(self)

    @property
    def time(self):
        return self.times[self.time_index]

    @property
    def state_overlaps_vs_time(self):
        return {state: np.abs(inner_product) ** 2 for state, inner_product in self.inner_products_vs_time.items()}

    @property
    def total_overlap_vs_time(self):
        return np.sum(overlap for overlap in self.state_overlaps_vs_time.values())

    def initialize_mesh(self):
        self.mesh = self.spec.mesh_type(self)

        if not (.99 < self.mesh.norm < 1.01):
            logger.warning('Initial wavefunction for {} {} may not be normalized (norm = {})'.format(self.__class__.__name__, self.name, self.mesh.norm))

        logger.debug('Initialized mesh for simulation {}'.format(self.name))

    def store_data(self, time_index):
        """Update the time-indexed data arrays with the current values."""
        norm = self.mesh.norm
        if norm > self.norm_vs_time[0]:
            logger.warning('Wavefunction norm ({}) has exceeded initial norm ({}) for {} {}'.format(norm, self.norm_vs_time[0], self.__class__.__name__, self.name))
        self.norm_vs_time[time_index] = norm
        self.energy_expectation_value_vs_time_internal[time_index] = self.mesh.energy_expectation_value
        for gauge in self.spec.dipole_gauges:
            self.electric_dipole_moment_vs_time[gauge][time_index] = self.mesh.dipole_moment(gauge = gauge)

        for state in self.spec.test_states:
            self.inner_products_vs_time[state][time_index] = self.mesh.inner_product(self.mesh.g_for_state(state))

        if self.spec.electric_potential is not None:
            self.electric_field_amplitude_vs_time[time_index] = self.spec.electric_potential.get_amplitude(t = self.times[time_index])

        logger.debug('{} {} stored data for time index {}'.format(self.__class__.__name__, self.name, time_index))

    def run_simulation(self, only_end_data = False, store_intermediate_meshes = False):
        logger.info('Performing time evolution on {} ({})'.format(self.name, self.file_name))

        if self.animator is not None:
            self.animator.initialize()

        self.status = 'running'
        logger.debug("{} {} status set to 'running'".format(self.__class__.__name__, self.name))

        if 0 < self.time_index < self.time_steps:
            self.evictions += 1

        while True:
            if not only_end_data or self.time_index == self.time_steps - 1:  # if last time step or taking all data
                self.store_data(self.time_index)

            if self.animator is not None and (self.time_index == 0 or self.time_index == self.time_steps or self.time_index % self.animator.animation_decimation == 0):
                self.animator.update_frame()
                self.animator.send_frame_to_ffmpeg()
                logger.debug('Made animation frame for time index {} / {}'.format(self.time_index, self.time_steps - 1))

            self.time_index += 1
            if self.time_index == self.time_steps:
                break

            self.mesh.evolve(self.times[self.time_index] - self.times[self.time_index - 1])  # evolve the mesh forward to the next time step

            logger.debug('{} {} evolved to time index {} / {} ({}%)'.format(self.__class__.__name__, self.name, self.time_index, self.time_steps - 1,
                                                                            np.around(100 * (self.time_index + 1) / self.time_steps, 2)))

            if self.spec.checkpoints:
                if (self.time_index + 1) % self.spec.checkpoint_at == 0:
                    self.save(target_dir = self.spec.checkpoint_dir, save_mesh = True)
                    logger.info('Checkpointed {} {} ({}) at time step {} / {}'.format(self.__class__.__name__, self.name, self.file_name, self.time_index + 1, self.time_steps))

        if self.animator is not None:
            self.animator.cleanup()

        self.end_time = dt.datetime.now()
        self.elapsed_time = self.end_time - self.start_time

        self.status = 'finished'
        logger.debug("Simulation status set to 'finished'")
        logger.info('Finished performing time evolution on {} ({})'.format(self.name, self.file_name))

    def plot_wavefunction_vs_time(self, grayscale = False, **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)

        grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [4, 1], hspace = 0.04)
        ax_overlaps = plt.subplot(grid_spec[0])
        ax_field = plt.subplot(grid_spec[1], sharex = ax_overlaps)

        if self.spec.electric_potential is not None:
            ax_field.plot(self.times / asec, self.electric_field_amplitude_vs_time / atomic_electric_field, color = 'black', linewidth = 2)

        ax_overlaps.plot(self.times / asec, self.norm_vs_time, label = r'$\left\langle \psi|\psi \right\rangle$', color = 'black', linewidth = 3, linestyle = '--')

        if grayscale:  # stackplot with two lines in grayscale (initial and non-initial)
            initial_overlap = [self.state_overlaps_vs_time[self.spec.initial_state]]
            non_initial_overlaps = [self.state_overlaps_vs_time[state] for state in self.spec.test_states if state != self.spec.initial_state]
            total_non_initial_overlaps = functools.reduce(np.add, non_initial_overlaps)
            overlaps = [initial_overlap, total_non_initial_overlaps]
            ax_overlaps.stackplot(self.times / asec, *overlaps, alpha = 1,
                                  labels = [r'$\left| \left\langle \psi|\psi_{init} \right\rangle \right|^2$', r'$\left| \left\langle \psi|\psi_{n\leq5} \right\rangle \right|^2$'],
                                  colors = ['.3', '.5'])
        else:  # stackplot with all states broken out in full color
            overlaps = [self.state_overlaps_vs_time[state] for state in self.spec.test_states]
            num_colors = len(overlaps)
            ax_overlaps.set_prop_cycle(cycler('color', [plt.get_cmap('gist_rainbow')(n / num_colors) for n in range(num_colors)]))
            ax_overlaps.stackplot(self.times / asec, *overlaps, alpha = 1, labels = [r'$\left| \left\langle \psi| {} \right\rangle \right|^2$'.format(state.tex_str) for state in self.spec.test_states])

        ax_overlaps.set_ylim(0.0, 1.0)
        ax_overlaps.set_xlim(self.spec.time_initial / asec, self.spec.time_final / asec)

        ax_overlaps.grid()
        ax_field.grid()

        ax_field.set_xlabel('Time $t$ (as)', fontsize = 15)
        ax_overlaps.set_ylabel('Wavefunction Metric', fontsize = 15)
        ax_field.set_ylabel('E-Field (a.u.)', fontsize = 11)

        if grayscale:
            ax_overlaps.legend(loc = 'lower left', fontsize = 12)
        else:
            ax_overlaps.legend(bbox_to_anchor = (1.075, 1), loc = 'upper left', borderaxespad = 0., fontsize = 10)

        ax_overlaps.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax_overlaps.tick_params(labelright = True)
        ax_field.tick_params(labelright = True)
        ax_overlaps.xaxis.tick_top()

        plt.rcParams['xtick.major.pad'] = 5
        plt.rcParams['ytick.major.pad'] = 5

        # Find at most n+1 ticks on the y-axis at 'nice' locations
        max_yticks = 6
        yloc = plt.MaxNLocator(max_yticks, prune = 'upper')
        ax_field.yaxis.set_major_locator(yloc)

        max_xticks = 6
        xloc = plt.MaxNLocator(max_xticks, prune = 'both')
        ax_field.xaxis.set_major_locator(xloc)

        ax_field.tick_params(axis = 'x', which = 'major', labelsize = 10)
        ax_field.tick_params(axis = 'y', which = 'major', labelsize = 10)
        ax_overlaps.tick_params(axis = 'both', which = 'major', labelsize = 10)

        postfix = ''
        if grayscale:
            postfix += '_GS'
        cp.utils.save_current_figure(name = self.spec.file_name + '__wavefunction_vs_time{}'.format(postfix), **kwargs)

        plt.close()

    def plot_dipole_moment_vs_time(self, gauge = 'length', **kwargs):
        cp.utils.xy_plot(self.times, np.real(self.electric_dipole_moment_vs_time[gauge]),
                         x_scale = 'as', y_scale = 'atomic_electric_dipole',
                         x_label = 'Time $t$', y_label = 'Dipole Moment $d(t)$',
                         name = self.spec.file_name + '__dipole_moment_vs_time',
                         **kwargs)

    def dipole_moment_vs_frequency(self, gauge = 'length', first_time = None, last_time = None):
        if first_time is None:
            first_time_index, first_time = 0, self.times[0]
        else:
            first_time_index, first_time, _ = cp.utils.find_nearest(self.times, first_time)
        if last_time is None:
            last_time_index, last_time = self.time_steps - 1, self.times[self.time_steps - 1]
        else:
            last_time_index, last_time, _ = cp.utils.find_nearest(self.times, last_time)
        points = last_time_index - first_time_index
        frequency = nfft.fftshift(nfft.fftfreq(points, self.spec.time_step))
        dipole_moment = nfft.fftshift(nfft.fft(self.electric_dipole_moment_vs_time[gauge][first_time_index: last_time_index], norm = 'ortho'))

        return frequency, dipole_moment

    def plot_dipole_moment_vs_frequency(self, gauge = 'length', frequency_range = 10000 * THz, first_time = None, last_time = None, **kwargs):
        frequency, dipole_moment = self.dipole_moment_vs_frequency(gauge = gauge, first_time = first_time, last_time = last_time)
        cp.utils.xy_plot(frequency, np.abs(dipole_moment) ** 2 / (atomic_electric_dipole ** 2),
                         x_scale = 'THz',
                         log_y = True,
                         x_label = 'Frequency $f$', y_label = r'Dipole Moment $\left| d(\omega) \right|^2$ $\left( e^2 \, a_0^2 \right)$',
                         x_range = frequency_range / 2, x_center = frequency_range / 2,
                         name = self.spec.file_name + '__dipole_moment_vs_frequency',
                         **kwargs)

    def save(self, target_dir = None, file_extension = '.sim', save_mesh = False):
        """
        Atomically pickle the Simulation to {target_dir}/{self.file_name}.{file_extension}, and gzip it for reduced disk usage.

        :param target_dir: directory to save the Simulation to
        :param file_extension: file extension to name the Simulation with
        :param save_mesh: if True, save the mesh as well as the Simulation. If False, don't.
        :return: None
        """

        if not save_mesh:
            mesh = self.mesh.copy()
            self.mesh = None

        out = super(ElectricFieldSimulation, self).save(target_dir = target_dir, file_extension = file_extension)

        if not save_mesh:
            self.mesh = mesh

        return out

    @staticmethod
    def load(file_path, initialize_mesh = False):
        """Return a simulation loaded from the file_path. kwargs are for Beet.load."""
        sim = cp.core.Simulation.load(file_path)

        if initialize_mesh:
            sim.initialize_mesh()

        return sim
