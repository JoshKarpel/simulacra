import datetime as dt
import logging

import numpy as np
import scipy as sp
import scipy.special as special
import scipy.sparse as sparse

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

    # @utils.memoize()
    # def evaluate(self, r, theta, phi):
    #     normalization = np.sqrt(((2 / (self.n * un.bohr_radius)) ** 3) * (sp.math.factorial(self.n - self.l - 1) / (2 * self.n * sp.math.factorial(self.n + self.l))))  # Griffith's normalization
    #     r_dep = np.exp(-r / (self.n * un.bohr_radius)) * ((2 * r / (self.n * un.bohr_radius)) ** self.l)
    #     lag_poly = special.eval_genlaguerre(self.n - self.l - 1, (2 * self.l) + 1, 2 * r / (self.n * un.bohr_radius))
    #     sph_harm = self.spherical_harmonic(theta, phi)
    #
    #     return normalization * r_dep * lag_poly * sph_harm

    def __call__(self, r, theta, phi):
        """
        Evaluate the hydrogenic bound state wavefunction at a point, or vectorized over an array of points.

        :param r: radial coordinate
        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the wavefunction at (r, theta, phi)
        """
        # return self.evaluate(r, theta, phi)
        normalization = np.sqrt(((2 / (self.n * un.bohr_radius)) ** 3) * (sp.math.factorial(self.n - self.l - 1) / (2 * self.n * sp.math.factorial(self.n + self.l))))  # Griffith's normalization
        r_dep = np.exp(-r / (self.n * un.bohr_radius)) * ((2 * r / (self.n * un.bohr_radius)) ** self.l)
        lag_poly = special.eval_genlaguerre(self.n - self.l - 1, (2 * self.l) + 1, 2 * r / (self.n * un.bohr_radius))
        sph_harm = self.spherical_harmonic(theta, phi)

        return normalization * r_dep * lag_poly * sph_harm


class BoundStateSuperposition:
    """A class that represents a superposition of bound states."""

    __slots__ = ('state', )

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

    def __call__(self, r, theta, phi):
        """
        Evaluate the hydrogenic bound state wavefunction at a point, or vectorized over an array of points.

        :param r: radial coordinate
        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the wavefunction at (r, theta, phi)
        """
        raise NotImplementedError


class ElectricFieldSpecification(core.Specification):
    def __init__(self, name, mesh_type = None, animator_type = None,
                 test_mass = un.electron_mass_reduced, test_charge = un.electron_charge,
                 initial_state = BoundState(1, 0),
                 test_states = (BoundState(n, l) for n in range(5) for l in range(n)),
                 internal_potential = potentials.NuclearPotential(charge = 1) + potentials.RadialImaginaryPotential(center = 20 * un.bohr_radius, width = 1 * un.bohr_radius, amplitude = 1 * un.atomic_electric_potential),
                 electric_potential = None,
                 time_initial = 0 * un.asec, time_final = 200 * un.asec, time_step = 1 * un.asec,
                 extra_time = 0 * un.asec, extra_time_step = 1 * un.asec,
                 checkpoints = False, checkpoint_at = 20, checkpoint_dir = None,
                 **kwargs):
        super(ElectricFieldSpecification, self).__init__(name, **kwargs)

        if mesh_type is None:
            raise ValueError('{} must have a mesh_type'.format(name))
        self.mesh_type = mesh_type
        self.animator_type = animator_type

        self.test_mass = test_mass
        self.test_charge = test_charge
        self.initial_state = initial_state
        self.test_states = tuple(test_states)

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


class CylindricalSliceSpecification(ElectricFieldSpecification):
    def __init__(self, name,
                 z_bound = 20 * un.bohr_radius, rho_bound = 20 * un.bohr_radius,
                 z_points = 2 ** 9, rho_points = 2 ** 8,
                 **kwargs):
        super(CylindricalSliceSpecification, self).__init__(name, mesh_type = CylindricalSliceFiniteDifferenceMesh, **kwargs)

        self.z_bound = z_bound
        self.rho_bound = rho_bound
        self.z_points = z_points
        self.rho_points = rho_points


class CylindricalSliceFiniteDifferenceMesh(qm.QuantumMesh):
    def __init__(self, specification, simulation):
        super(CylindricalSliceFiniteDifferenceMesh, self).__init__(specification, simulation)

        self.z = np.linspace(-self.spec.z_bound, self.spec.z_bound, self.spec.z_points)
        self.rho = np.delete(np.linspace(0, self.spec.rho_bound, self.spec.rho_points + 1), 0)

        self.delta_z = self.z[1] - self.z[0]
        self.delta_rho = self.rho[1] - self.rho[0]

        self.rho -= self.delta_rho / 2

        self.z_center_index = self.spec.z_points / 2
        self.z_max = np.max(self.z)
        self.rho_max = np.max(self.rho)

        self.z_mesh, self.rho_mesh = np.meshgrid(self.z, self.rho, indexing = 'ij')

        self.g_mesh = self.g_for_state(self.spec.initial_state)

        self.mesh_points = len(self.z) * len(self.rho)
        self.matrix_operator_shape = (self.mesh_points, self.mesh_points)
        self.mesh_shape = np.shape(self.r_mesh)

    def flatten_mesh(self, mesh, flatten_along):
        """Return a mesh flattened along one of the mesh coordinates ('z' or 'rho')."""
        if flatten_along == 'z':
            flat = 'F'
        elif flatten_along == 'rho':
            flat = 'C'
        else:
            raise ValueError("{} is not a valid specifier for flatten_along (valid specifiers: 'z', 'rho')".format(flatten_along))

        return mesh.flatten(flat)

    def wrap_vector(self, mesh, wrap_along):
        if wrap_along == 'z':
            wrap = 'F'
        elif wrap_along == 'rho':
            wrap = 'C'
        else:
            raise ValueError("{} is not a valid specifier for flatten_along (valid specifiers: 'z', 'rho')".format(wrap_along))

        return np.reshape(mesh, self.mesh_shape, wrap)

    @property
    def g_factor(self):
        return np.sqrt(un.twopi * self.rho_mesh)

    @property
    def r_mesh(self):
        return np.sqrt((self.z_mesh ** 2) + (self.rho_mesh ** 2))

    @property
    def theta_mesh(self):
        return np.arctan(self.rho_mesh / self.z_mesh)

    def inner_product(self, state_a = None, state_b = None):
        """Inner product between two states. If mesh_b is None, the state on the g_mesh is used."""
        if state_a is None:
            mesh_a = self.g_mesh
        else:
            mesh_a = self.g_for_state(state_a)
        if state_b is None:
            mesh_b = self.g_mesh
        else:
            mesh_b = self.g_for_state(state_b)

        return np.einsum('ij,ij->', np.conj(mesh_a), mesh_b) * (self.delta_z * self.delta_rho)

    def state_overlap(self, state_a = None, state_b = None):
        """State overlap between two states. If either state is None, the state on the g_mesh is used for that state."""
        return np.abs(self.inner_product(state_a, state_b)) ** 2

    @property
    def norm(self):
        return np.real(self.inner_product())

    @utils.memoize()
    def g_for_state(self, state):
        return self.g_factor * state(self.r_mesh, self.theta_mesh, 0)

    @utils.memoize()
    def get_kinetic_energy_matrix_operators(self):
        z_prefactor = -(un.hbar ** 2) / (2 * self.spec.test_mass * (self.delta_z ** 2))
        rho_prefactor = -(un.hbar ** 2) / (2 * self.spec.test_mass * (self.delta_rho ** 2))

        z_diagonal = z_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        z_offdiagonal = z_prefactor * np.array([1 if (z_index + 1) % self.spec.z_points != 0 else 0 for z_index in range(self.mesh_points - 1)], dtype = np.complex128)

        rho_diagonal = rho_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        rho_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for rho_index in range(self.mesh_points - 1):
            if (rho_index + 1) % self.spec.rho_points != 0:
                j = (rho_index % self.spec.rho_points) + 1  # get j for the upper diagonal
                rho_offdiagonal[rho_index] = j / np.sqrt((j ** 2) - 0.25)
        rho_offdiagonal *= rho_prefactor

        z_kinetic = sparse.diags([z_offdiagonal, z_diagonal, z_offdiagonal], offsets = (-1, 0, 1))
        rho_kinetic = sparse.diags([rho_offdiagonal, rho_diagonal, rho_offdiagonal], offsets = (-1, 0, 1))

        return z_kinetic, rho_kinetic

    @utils.memoize()
    def get_internal_hamiltonian_matrix_operators(self):
        z_kinetic, rho_kinetic = self.get_kinetic_energy_matrix_operators()
        potential_mesh = self.spec.internal_potential(r = self.r_mesh)

        z_kinetic.data[1] += 0.5 * self.flatten_mesh(potential_mesh, 'z')
        rho_kinetic.data[1] += 0.5 * self.flatten_mesh(potential_mesh, 'rho')

        return z_kinetic, rho_kinetic

    @utils.memoize()
    def get_probability_current_matrix_operators(self):
        z_prefactor = un.hbar / (4 * un.pi * self.spec.test_mass * self.delta_rho * self.delta_z)
        rho_prefactor = un.hbar / (4 * un.pi * self.spec.test_mass * (self.delta_rho ** 2))

        # construct the diagonals of the z probability current matrix operator
        z_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for z_index in range(0, self.mesh_points - 1):
            if (z_index + 1) % self.spec.z_points == 0:  # detect edge of mesh
                z_offdiagonal[z_index] = 0
            else:
                j = z_index // self.spec.z_points
                z_offdiagonal[z_index] = 1 / (j + 0.5)
        z_offdiagonal *= z_prefactor

        # construct the diagonals of the rho probability current matrix operator
        rho_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for rho_index in range(0, self.mesh_points - 1):
            if (rho_index + 1) % self.spec.rho_points == 0:  # detect edge of mesh
                rho_offdiagonal[rho_index] = 0
            else:
                j = (rho_index % self.spec.rho_points) + 1
                rho_offdiagonal[rho_index] = 1 / np.sqrt((j ** 2) - 0.25)
        rho_offdiagonal *= rho_prefactor

        z_current = sparse.diags([-z_offdiagonal, z_offdiagonal], offsets = [-1, 1])
        rho_current = sparse.diags([-rho_offdiagonal, rho_offdiagonal], offsets = [-1, 1])

        return z_current, rho_current

    def evolve(self, time_step):
        tau = time_step / (2 * un.hbar)

        if self.spec.electric_potential is not None:
            electric_potential_energy_mesh = self.spec.electric_potential(t = self.sim.time, distance_along_polarization = self.z_mesh, test_charge = self.spec.test_charge)
        else:
            electric_potential_energy_mesh = np.zeros(self.mesh_shape)

        # add the external potential to the Hamiltonian matrices and multiply them by i * tau to get them ready for the next steps
        hamiltonian_z, hamiltonian_rho = self.get_internal_hamiltonian_matrix_operators()
        hamiltonian_z = hamiltonian_z.copy()  # we're going to directly modify the data structure for speed, so we need to make copies
        hamiltonian_rho = hamiltonian_rho.copy()

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


class SphericalSliceSpecification(ElectricFieldSpecification):
    def __init__(self, name,
                 r_bound = 20 * un.bohr_radius,
                 r_points = 2 ** 10, theta_points = 2 ** 10,
                 **kwargs):
        super(SphericalSliceSpecification, self).__init__(name, mesh_type = SphericalSliceFiniteDifferenceMesh, **kwargs)

        self.r_bound = r_bound
        self.r_points = r_points
        self.theta_points = theta_points


class SphericalSliceFiniteDifferenceMesh(qm.QuantumMesh):
    def __init__(self, parameters, simulation):
        super(SphericalSliceFiniteDifferenceMesh, self).__init__(parameters, simulation)

    def norm(self):
        raise NotImplementedError


class SphericalHarmonicFiniteDifferenceMesh(qm.QuantumMesh):
    def __init__(self, parameters, simulation):
        super(SphericalHarmonicFiniteDifferenceMesh, self).__init__(parameters, simulation)

    def norm(self):
        raise NotImplementedError
    

class SphericalHarmonicSpecification(ElectricFieldSpecification):
    def __init__(self, name,
                 r_bound = 20 * un.bohr_radius,
                 r_points = 2 ** 10, spherical_harmonics = (math.SphericalHarmonic(l) for l in range(5)),
                 **kwargs):
        super(SphericalHarmonicSpecification, self).__init__(name, mesh_type = SphericalHarmonicFiniteDifferenceMesh, **kwargs)

        self.r_bound = r_bound
        self.r_points = r_points
        self.spherical_harmonics = spherical_harmonics


class ElectricFieldSimulation(core.Simulation):
    def __init__(self, spec):
        super(ElectricFieldSimulation, self).__init__(spec)

        self.mesh = self.initialize_mesh()
        # self.animator = self.spec.animator(self)

        total_time = self.spec.time_final - self.spec.time_initial
        self.times = np.linspace(self.spec.time_initial, self.spec.time_final, int(total_time / self.spec.time_step) + 1)
        if self.spec.extra_time is not None:
            extra_times = np.delete(np.linspace(self.spec.time_final, self.spec.time_final + self.spec.extra_time, (self.spec.extra_time / self.spec.extra_time_step) + 1), 0)
            self.times = np.concatenate((self.times, extra_times))
        self.time_index = 0
        self.time_steps = len(self.times)

        # simulation data storage
        self.norm_vs_time = np.zeros(self.time_steps) * np.NaN
        self.inner_products_vs_time = {state: np.zeros(self.time_steps, dtype = np.complex128) * np.NaN for state in self.spec.test_states}
        self.electric_field_amplitude_vs_time = np.zeros(self.time_steps) * np.NaN

        # diagnostic data
        self.evictions = 0
        self.start_time = dt.datetime.now()
        self.end_time = None
        self.elapsed_time = None
        self.latest_load_time = dt.datetime.now()
        self.run_time = dt.timedelta()

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
        logger.debug('Initialized mesh for simulation {}'.format(self.name))
        return self.spec.mesh_type(self.spec, self)

    def store_data(self, time_index):
        """Update the time-indexed data arrays with the current values."""
        self.norm_vs_time[time_index] = self.mesh.norm

        for state in self.spec.test_states:
            self.inner_products_vs_time[state][time_index] = self.mesh.inner_product(state)

        if self.spec.electric_potential is not None:
            self.electric_field_amplitude_vs_time[time_index] = self.spec.electric_potential.get_amplitude(t = self.times[time_index])

        logger.debug('{} {} stored data for time index {}'.format(self.__class__.__name__, self.name, time_index))

    def run_simulation(self, only_end_data = False, store_intermediate_meshes = False):
        logger.info('Performing time evolution on {} ({})'.format(self.name, self.file_name))

        # if self.animator is not None:
        #     self.animator.initialize()

        self.status = 'running'
        logger.debug("{} {} status set to 'running'".format(self.__class__.__name__, self.name))

        if 0 < self.time_index < self.time_steps:
            self.evictions += 1

        while True:
            logger.debug('{} {} working on time index {} / {} ({}%)'.format(self.__class__.__name__, self.name, self.time_index, self.time_steps - 1, np.around(100 * (self.time_index + 1) / self.time_steps, 2)))

            if not only_end_data or self.time_index == self.time_steps - 1:  # if last time step or taking all data
                self.store_data(self.time_index)

            # if self.animator is not None and (self.time_index == 0 or self.time_index == self.time_steps or self.time_index % self.animator.animation_decimation == 0):
            #     self.animator.update_frame()
            #     self.animator.send_frame_to_ffmpeg()
            #     self.logger.debug('Made animation frame for time step {} / {}'.format(self.time_index + 1, self.time_steps))

            self.time_index += 1
            if self.time_index == self.time_steps:
                break

            self.mesh.evolve(self.times[self.time_index] - self.times[self.time_index - 1])  # evolve the mesh forward to the next time step

            if self.spec.checkpoints:
                if (self.time_index + 1) % self.spec.checkpoint_at == 0:
                    self.save(target_dir = self.spec.checkpoint_dir, save_mesh = True)
                    logger.info('Checkpointed {} {} ({}) at time step {} / {}'.format(self.__class__.__name__, self.name, self.file_name, self.time_index + 1, self.time_steps))

        # if self.animator is not None:
        #     self.animator.cleanup()

        self.end_time = dt.datetime.now()
        self.elapsed_time = self.end_time - self.start_time

        self.status = 'finished'
        logger.debug("Simulation status set to 'finished'")
        logger.info('Finished performing time evolution on {} ({})'.format(self.name, self.file_name))

    def save(self, target_dir = None, file_extension = '.sim', save_mesh = False):
        """
        Atomically pickle the Simulation to {target_dir}/{self.file_name}.{file_extension}, and gzip it for reduced disk usage.

        :param target_dir: directory to save the Simulation to
        :param file_extension: file extension to name the Simulation with
        :param save_mesh: if True, save the mesh as well as the Simulation. If False, don't.
        :return: None
        """

        if not save_mesh:
            self.mesh = None

        if self.status != 'finished':
            self.run_time += dt.datetime.now() - self.latest_load_time
            self.latest_load_time = dt.datetime.now()

        super(ElectricFieldSimulation, self).save(target_dir = target_dir, file_extension = file_extension)

