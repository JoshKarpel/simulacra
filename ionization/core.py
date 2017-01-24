import datetime as dt
import functools
import logging
import itertools as it
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
from . import potentials, states

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

GRID_COLOR = 'dodgerblue'  # the color to use for the gridlines on colormesh plots


def electron_energy_from_wavenumber(k):
    return (hbar * k) ** 2 / (2 * electron_mass)


def electron_wavenumber_from_energy(energy):
    return np.sqrt(2 * electron_mass * energy) / hbar


class ElectricFieldSpecification(cp.core.Specification):
    """A base Specification for a Simulation with an electric field."""

    evolution_equations = cp.utils.RestrictedValues('evolution_equations', {'L', 'H'})
    evolution_method = cp.utils.RestrictedValues('evolution_method', {'CN', 'SO', 'S'})

    def __init__(self, name,
                 mesh_type = None,
                 test_mass = electron_mass_reduced, test_charge = electron_charge,
                 initial_state = states.HydrogenBoundState(1, 0),
                 test_states = tuple(),
                 dipole_gauges = ('length',),
                 internal_potential = potentials.Coulomb(charge = proton_charge),
                 electric_potential = potentials.NoElectricField(),
                 mask = potentials.NoMask(),
                 evolution_method = 'CN', evolution_equations = 'H',
                 time_initial = 0 * asec, time_final = 200 * asec, time_step = 1 * asec,
                 minimum_time_final = 0 * asec, extra_time_step = 1 * asec,
                 checkpoints = False, checkpoint_every = 20, checkpoint_dir = None,
                 animators = tuple(),
                 store_norm_by_l = False,
                 **kwargs):
        """
        Initialize an ElectricFieldSpecification instance from the given parameters.

        This class should generally not be instantiated directly - its subclasses contain information about the mesh that is necessary to create an actual ElectricFieldSimulation.

        :param name: the name of the Specification
        :param mesh_type: the type of QuantumMesh that will be used in the Simulation.
        :param test_mass: the mass of the test particle
        :param test_charge: the charge of the test particle
        :param initial_state: the initial QuantumState of the test particle
        :param test_states: a list of states to perform inner products with during time evolution
        :param dipole_gauges: a list of dipole gauges to check the expectation value of during time evolution
        :param internal_potential: the time-independent part of the potential the particle experiences
        :param electric_potential: the electric field (possibly time-dependent) that the particle experiences
        :param mask: a mask function to be applied to the QuantumMesh after every time step
        :param evolution_method: which evolution method to use. Options are 'CN' (Crank-Nicolson), 'SO' (Split-Operator), and 'S' (Spectral). Only certain options are available for certain meshes.
        :param evolution_equations: which form of the evolution equations to use. Options are 'L' (Lagrangian) and 'H' (Hamiltonian). Most meshes use 'H'.
        :param time_initial: the initial time
        :param time_final: the final time
        :param time_step: the time step
        :param minimum_time_final: the minimum final time the Simulation will run to (i.e., if this is longer than time_final, more time will be added to the Simulation)
        :param extra_time_step: the time step to use during extra time from minimum_final_time
        :param checkpoints: if True, a checkpoint file will be maintained during time evolution
        :param checkpoint_every: how many time steps to calculate between checkpoints
        :param checkpoint_dir: a directory path to store the checkpoint file in
        :param animators: a list of Animators which will be run during time evolution
        :param store_norm_by_l: if True, the Simulation will store the amount of norm in each spherical harmonic.
        :param kwargs:
        """
        super(ElectricFieldSpecification, self).__init__(name, simulation_type = ElectricFieldSimulation, **kwargs)

        if mesh_type is None:
            raise ValueError('{} must have a mesh_type'.format(name))
        self.mesh_type = mesh_type

        self.test_mass = test_mass
        self.test_charge = test_charge
        self.initial_state = initial_state
        self.test_states = tuple(sorted(tuple(test_states)))  # consume input iterators
        self.dipole_gauges = tuple(sorted(tuple(dipole_gauges)))

        self.internal_potential = internal_potential
        self.electric_potential = electric_potential
        self.mask = mask

        self.evolution_method = evolution_method
        self.evolution_equations = evolution_equations

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        self.minimum_time_final = minimum_time_final
        self.extra_time_step = extra_time_step

        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir

        self.animators = tuple(animators)

        self.store_norm_by_l = store_norm_by_l

    def info(self):
        checkpoint = ['Checkpointing: ']
        if self.checkpoints:
            if self.checkpoint_dir is not None:
                working_in = self.checkpoint_dir
            else:
                working_in = 'cwd'
            checkpoint[0] += 'every {} time steps, working in {}'.format(self.checkpoint_every, working_in)
        else:
            checkpoint[0] += 'disabled'

        animation = ['Animation: ']
        if len(self.animators) > 0:
            animation += ['   ' + str(animator) for animator in self.animators]
        else:
            animation[0] += 'None'

        time_evolution = ['Time Evolution:',
                          '   Initial State: {}'.format(self.initial_state),
                          '   Initial Time: {} as'.format(uround(self.time_initial, asec)),
                          '   Final Time: {} as'.format(uround(self.time_final, asec)),
                          '   Time Step: {} as'.format(uround(self.time_step, asec))]

        if self.minimum_time_final is not 0:
            time_evolution += ['   Minimum Final Time: {} as'.format(uround(self.minimum_time_final, asec)),
                               '   Extra Time Step: {} as'.format(uround(self.extra_time_step, asec))]

        potentials = ['Potentials and Masks:']
        potentials += ['   {}'.format(x) for x in it.chain(self.internal_potential, self.electric_potential, self.mask)]

        analysis = ['Analysis:',
                    '   Test Charge: {} e'.format(uround(self.test_charge, proton_charge)),
                    '   Test Mass: {} m_e'.format(uround(self.test_mass, electron_mass)),
                    '   Test States: {}'.format(', '.join(str(s) for s in self.test_states)),
                    '   Dipole Gauges: {}'.format(', '.join(self.dipole_gauges))]

        return '\n'.join(checkpoint + animation + time_evolution + potentials + analysis)


class QuantumMesh:
    def __init__(self, simulation):
        self.sim = simulation
        self.spec = simulation.spec

        self.g_mesh = None
        self.inner_product_multiplier = None

    def __str__(self):
        return '{} for {}'.format(self.__class__.__name__, str(self.sim))

    def __repr__(self):
        return '{}(parameters = {}, simulation = {})'.format(self.__class__.__name__, repr(self.spec), repr(self.sim))

    def inner_product(self, mesh_a = None, mesh_b = None):
        """Inner product between two meshes. If either mesh is None, the state on the g_mesh is used for that state."""
        if mesh_a is None:
            mesh_a = self.g_mesh
        if mesh_b is None:
            mesh_b = self.g_mesh

        # return np.einsum('ij,ij->', np.conj(mesh_a), mesh_b) * self.inner_product_multiplier
        return np.sum(np.conj(mesh_a) * mesh_b) * self.inner_product_multiplier

    def mesh_overlap(self, mesh_a = None, mesh_b = None):
        return np.abs(self.inner_product(mesh_a, mesh_b)) ** 2

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

    @property
    def energy_expectation_value(self):
        raise NotImplementedError

    def dipole_moment_expectation_value(self, gauge = 'length'):
        raise NotImplementedError

    def __abs__(self):
        return self.norm

    def copy(self):
        return deepcopy(self)

    @property
    def psi_mesh(self):
        return self.g_mesh / self.g_factor

    def evolve(self, time_step):
        pre_evolve_norm = self.norm
        getattr(self, '_evolve_' + self.spec.evolution_method)(time_step)
        norm_diff_evolve = pre_evolve_norm - self.norm
        if norm_diff_evolve / pre_evolve_norm > .001:
            logger.warning('Evolution may be dangerously non-unitary, norm decreased by {} ({} %) during evolution step'.format(norm_diff_evolve, norm_diff_evolve / pre_evolve_norm))

        pre_mask_norm = self.norm
        self.g_mesh *= self.spec.mask(r = self.r_mesh)
        norm_diff_mask = pre_mask_norm - self.norm
        logger.debug('Applied mask {} to g_mesh for {} {}, removing {} norm'.format(self.spec.mask, self.sim.__class__.__name__, self.sim.name, norm_diff_mask))
        return norm_diff_mask

    def get_mesh_slicer(self, plot_limit):
        raise NotImplementedError

    def attach_mesh_to_axis(self, axis, mesh, plot_limit = None, distance_unit = 'nm', **kwargs):
        raise NotImplementedError

    def plot_mesh(self, mesh, distance_unit = 'nm', **kwargs):
        raise NotImplementedError

    def abs_g_squared(self, normalize = False, log = False):
        out = np.abs(self.g_mesh) ** 2
        if normalize:
            out /= np.nanmax(out)
        if log:
            out = np.log10(out)

        return out

    def attach_g_to_axis(self, axis, normalize = False, log = False, plot_limit = None, **kwargs):
        """Attach a colormesh of |g|^2 to the given axis."""
        return self.attach_mesh_to_axis(axis, self.abs_g_squared(normalize = normalize, log = log), plot_limit = plot_limit, **kwargs)

    def update_g_mesh(self, mesh, normalize = False, log = False, plot_limit = None, slicer = 'get_mesh_slicer'):
        """Update a colormesh with with the current value of |g|^2."""
        new_mesh = self.abs_g_squared(normalize = normalize, log = log)[getattr(self, slicer)(plot_limit)]

        try:
            mesh.set_array(new_mesh.ravel())
        except AttributeError as e:  # if the mesh is 1D we can't .ravel() it and instead should just set the y data with the mesh
            mesh.set_ydata(new_mesh)

    def plot_g(self, normalize = True, name_postfix = '', **kwargs):
        """Plot |g|^2. kwargs are for plot_mesh."""
        title = ''
        if normalize:
            title = r'Normalized '
        title += r'$|g|^2$'
        name = 'g' + name_postfix

        self.plot_mesh(self.abs_g_squared(normalize = normalize), name = name, title = title, color_map_min = 0, **kwargs)

    def abs_psi_squared(self, normalize = False, log = False):
        out = np.abs(self.psi_mesh) ** 2
        if normalize:
            out /= np.nanmax(out)
        if log:
            out = np.log10(out)

        return out

    def attach_psi_to_axis(self, axis, normalize = False, log = False, plot_limit = None, **kwargs):
        """Attach a colormesh of |psi|^2 to the given axis."""
        return self.attach_mesh_to_axis(axis, self.abs_psi_squared(normalize = normalize, log = log), plot_limit = plot_limit, **kwargs)

    def update_psi_mesh(self, colormesh, normalize = False, log = False, plot_limit = None):
        """Update a colormesh with with the current value of |psi|^2."""
        new_mesh = self.abs_psi_squared(normalize = normalize, log = log)[self.get_mesh_slicer(plot_limit)]

        try:
            colormesh.set_array(new_mesh.ravel())
        except AttributeError as e:
            colormesh.set_ydata(new_mesh)

    def plot_psi(self, normalize = True, name_postfix = '', **kwargs):
        """Plot |psi|^2. kwargs are for plot_mesh."""
        title = ''
        if normalize:
            title = r'Normalized '
        title += r'$|\psi|^2$'
        name = 'psi' + name_postfix

        self.plot_mesh(self.abs_psi_squared(normalize = normalize), name = name, title = title, color_map_min = 0, **kwargs)


class LineSpecification(ElectricFieldSpecification):
    def __init__(self, name,
                 x_bound = 10 * nm,
                 x_points = 2 ** 9,
                 **kwargs):
        super(LineSpecification, self).__init__(name, mesh_type = LineMesh,
                                                evolution_method = 'S',
                                                **kwargs)

        self.x_bound = x_bound
        self.x_points = int(x_points)

    def info(self):
        mesh = ['Mesh: {}'.format(self.mesh_type.__name__),
                '   X Bound: {} Bohr radii | {} nm'.format(uround(self.x_bound, bohr_radius, 3), uround(self.x_bound, nm, 3)),
                '   X Points: {}'.format(self.x_points),
                '   X Mesh Spacing: ~{} Bohr radii | ~{} nm'.format(uround(2 * self.x_bound / self.x_points, bohr_radius, 3), uround(2 * self.x_bound / self.x_points, nm, 3))]

        return '\n'.join((super(LineSpecification, self).info(), *mesh))


class LineMesh(QuantumMesh):
    mesh_storage_method = ['x']

    def __init__(self, simulation):
        super(LineMesh, self).__init__(simulation)

        self.x_mesh = np.linspace(-self.spec.x_bound, self.spec.x_bound, self.spec.x_points)
        self.delta_x = np.abs(self.x_mesh[1] - self.x_mesh[0])
        self.x_center_index = cp.utils.find_nearest_entry(self.x_mesh, 0).index

        self.wavenumbers = twopi * nfft.fftfreq(len(self.x_mesh), d = self.delta_x)
        self.delta_k = np.abs(self.wavenumbers[1] - self.wavenumbers[0])

        self.inner_product_multiplier = self.delta_x

        self.g_mesh = self.get_g_for_state(self.spec.initial_state)
        self.g_factor = 1

        self.free_evolution_prefactor = -1j * (hbar / (2 * self.spec.test_mass)) * (self.wavenumbers ** 2)

    @property
    def r_mesh(self):
        return self.x_mesh

    @cp.utils.memoize
    def get_g_for_state(self, state):
        return state(self.x_mesh)

    @property
    def energy_expectation_value(self):
        potential = self.inner_product(mesh_b = self.spec.internal_potential(t = self.sim.time, r = self.x_mesh, distance = self.x_mesh) * self.g_mesh)
        kinetic = np.sum((((hbar * self.wavenumbers) ** 2) / (2 * self.spec.test_mass)) * (np.abs(self.fft(self.g_mesh)) ** 2)) / np.sum(np.abs(self.fft(self.g_mesh)) ** 2)

        return np.real(potential + kinetic)

    def dipole_moment_expectation_value(self, gauge = 'length'):
        """Get the dipole moment in the specified gauge."""
        if gauge == 'length':
            return self.spec.test_charge * self.inner_product(mesh_b = self.x_mesh * self.g_mesh)
        elif gauge == 'velocity':
            raise NotImplementedError

    def fft(self, mesh):
        return nfft.fft(mesh, norm = 'ortho')

    def ifft(self, mesh):
        return nfft.ifft(mesh, norm = 'ortho')

    def _evolve_potential(self, time_step):
        pot = self.spec.internal_potential(t = self.sim.time, r = self.x_mesh, distance = self.x_mesh) + self.spec.electric_potential(t = self.sim.time, r = self.x_mesh, distance = self.x_mesh, distance_along_polarization = self.x_mesh, test_charge = self.spec.test_charge)
        self.g_mesh *= np.exp(-1j * time_step * pot / hbar)

    def _evolve_free(self, time_step):
        self.g_mesh = self.ifft(self.fft(self.g_mesh) * np.exp(self.free_evolution_prefactor * time_step))

    def _evolve_S(self, time_step):
        self._evolve_potential(time_step / 2)
        self._evolve_free(time_step)  # splitting order chosen for computational efficiency
        self._evolve_potential(time_step / 2)

    def get_mesh_slicer(self, plot_limit):
        if plot_limit is None:
            mesh_slicer = slice(None, None, 1)
        else:
            x_lim_points = round(plot_limit / self.delta_x)
            mesh_slicer = slice(int(self.x_center_index - x_lim_points), int(self.x_center_index + x_lim_points + 1), 1)

        return mesh_slicer

    def attach_mesh_to_axis(self, axis, mesh, plot_limit = None, distance_unit = 'nm', **kwargs):
        unit_value, unit_name = unit_value_and_name_from_unit(distance_unit)

        line, = axis.plot(self.x_mesh[self.get_mesh_slicer(plot_limit)] / unit_value, mesh[self.get_mesh_slicer(plot_limit)], **kwargs)

        return line

    def plot_mesh(self, mesh, distance_unit = 'nm', **kwargs):
        unit_value, unit_name = unit_value_and_name_from_unit(distance_unit)

        cp.utils.xy_plot(self.sim.name + '_' + kwargs.pop('name'), self.x_mesh, mesh,
                         x_label = 'Distance $x$', x_scale = distance_unit, **kwargs)

    def plot_fft(self):
        raise NotImplementedError


class CylindricalSliceSpecification(ElectricFieldSpecification):
    def __init__(self, name,
                 z_bound = 20 * bohr_radius, rho_bound = 20 * bohr_radius,
                 z_points = 2 ** 9, rho_points = 2 ** 8,
                 **kwargs):
        super(CylindricalSliceSpecification, self).__init__(name, mesh_type = CylindricalSliceMesh, **kwargs)

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
    mesh_storage_method = ('z', 'rho')

    def __init__(self, simulation):
        super(CylindricalSliceMesh, self).__init__(simulation)

        self.z = np.linspace(-self.spec.z_bound, self.spec.z_bound, self.spec.z_points)
        self.rho = np.delete(np.linspace(0, self.spec.rho_bound, self.spec.rho_points + 1), 0)

        self.delta_z = self.z[1] - self.z[0]
        self.delta_rho = self.rho[1] - self.rho[0]
        self.inner_product_multiplier = self.delta_z * self.delta_rho

        self.rho -= self.delta_rho / 2

        self.z_center_index = int(self.spec.z_points // 2)
        self.z_max = np.max(self.z)
        self.rho_max = np.max(self.rho)

        # self.z_mesh, self.rho_mesh = np.meshgrid(self.z, self.rho, indexing = 'ij')
        self.g_mesh = self.get_g_for_state(self.spec.initial_state)

        self.mesh_points = len(self.z) * len(self.rho)
        self.matrix_operator_shape = (self.mesh_points, self.mesh_points)
        self.mesh_shape = np.shape(self.r_mesh)

    @property
    @cp.utils.memoize
    def z_mesh(self):
        return np.meshgrid(self.z, self.rho, indexing = 'ij')[0]

    @property
    @cp.utils.memoize
    def rho_mesh(self):
        return np.meshgrid(self.z, self.rho, indexing = 'ij')[1]

    @property
    def g_factor(self):
        return np.sqrt(twopi * self.rho_mesh)

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

    @cp.utils.memoize
    def get_g_for_state(self, state):
        return self.g_factor * state(self.r_mesh, self.theta_mesh, 0)

    def dipole_moment_expectation_value(self, gauge = 'length'):
        """Get the dipole moment in the specified gauge."""
        if gauge == 'length':
            return self.spec.test_charge * self.inner_product(mesh_b = self.z_mesh * self.g_mesh)
        elif gauge == 'velocity':
            raise NotImplementedError

    @cp.utils.memoize
    def _get_kinetic_energy_matrix_operators(self):
        """Get the mesh kinetic energy operator matrices for z and rho."""
        z_prefactor = -(hbar ** 2) / (2 * self.spec.test_mass * (self.delta_z ** 2))
        rho_prefactor = -(hbar ** 2) / (2 * self.spec.test_mass * (self.delta_rho ** 2))

        z_diagonal = z_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        z_offdiagonal = z_prefactor * np.array([1 if (z_index + 1) % self.spec.z_points != 0 else 0 for z_index in range(self.mesh_points - 1)], dtype = np.complex128)

        @cp.utils.memoize
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

    @cp.utils.memoize
    def _get_internal_hamiltonian_matrix_operators(self):
        """Get the mesh internal Hamiltonian matrix operators for z and rho."""
        z_kinetic, rho_kinetic = self._get_kinetic_energy_matrix_operators()
        potential_mesh = self.spec.internal_potential(r = self.r_mesh, test_charge = self.spec.test_charge)

        z_kinetic.data[1] += 0.5 * self.flatten_mesh(potential_mesh, 'z')
        rho_kinetic.data[1] += 0.5 * self.flatten_mesh(potential_mesh, 'rho')

        return z_kinetic, rho_kinetic

    def tg_mesh(self, use_abs_g = False):
        hamiltonian_z, hamiltonian_rho = self._get_kinetic_energy_matrix_operators()

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
        hamiltonian_z, hamiltonian_rho = self._get_internal_hamiltonian_matrix_operators()

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

    @cp.utils.memoize
    def _get_probability_current_matrix_operators(self):
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

        @cp.utils.memoize
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
        z_current, rho_current = self._get_probability_current_matrix_operators()

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

        electric_potential_energy_mesh = self.spec.electric_potential(t = self.sim.time, distance_along_polarization = self.z_mesh, test_charge = self.spec.test_charge)

        # add the external potential to the Hamiltonian matrices and multiply them by i * tau to get them ready for the next steps
        hamiltonian_z, hamiltonian_rho = self._get_internal_hamiltonian_matrix_operators()
        hamiltonian_z, hamiltonian_rho = hamiltonian_z.copy(), hamiltonian_rho.copy()

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

    @cp.utils.memoize
    def get_mesh_slicer(self, plot_limit = None):
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if plot_limit is None:
            mesh_slicer = (slice(None, None, 1), slice(None, None, 1))
        else:
            z_lim_points = round(plot_limit / self.delta_z)
            rho_lim_points = round(plot_limit / self.delta_rho)
            mesh_slicer = (slice(int(self.z_center_index - z_lim_points), int(self.z_center_index + z_lim_points + 1), 1), slice(0, int(rho_lim_points + 1), 1))

        return mesh_slicer

    def attach_mesh_to_axis(self, axis, mesh, plot_limit = None, distance_unit = 'bohr_radius', **kwargs):
        unit_value, unit_name = unit_value_and_name_from_unit(distance_unit)

        color_mesh = axis.pcolormesh(self.z_mesh[self.get_mesh_slicer(plot_limit)] / unit_value,
                                     self.rho_mesh[self.get_mesh_slicer(plot_limit)] / unit_value,
                                     mesh[self.get_mesh_slicer(plot_limit)],
                                     shading = 'gouraud',
                                     **kwargs)

        return color_mesh

    def attach_probability_current_to_axis(self, axis, plot_limit = None, distance_unit = 'bohr_radius'):
        unit_value, unit_name = unit_value_and_name_from_unit(distance_unit)

        current_mesh_z, current_mesh_rho = self.get_probability_current_vector_field()

        current_mesh_z *= self.delta_z
        current_mesh_rho *= self.delta_rho

        skip_count = int(self.z_mesh.shape[0] / 50), int(self.z_mesh.shape[1] / 50)
        skip = (slice(None, None, skip_count[0]), slice(None, None, skip_count[1]))
        normalization = np.max(np.sqrt(current_mesh_z ** 2 + current_mesh_rho ** 2)[skip])
        if normalization == 0 or normalization is np.NaN:
            normalization = 1

        quiv = axis.quiver(self.z_mesh[self.get_mesh_slicer(plot_limit)][skip] / unit_value,
                           self.rho_mesh[self.get_mesh_slicer(plot_limit)][skip] / unit_value,
                           current_mesh_z[self.get_mesh_slicer(plot_limit)][skip] / normalization,
                           current_mesh_rho[self.get_mesh_slicer(plot_limit)][skip] / normalization,
                           pivot = 'middle', units = 'width', scale = 10, scale_units = 'width', width = 0.0015, alpha = 0.5)

        return quiv

    def plot_mesh(self, mesh, name = '', target_dir = None, title = None,
                  overlay_probability_current = False, probability_current_time_step = 0, plot_limit = None,
                  distance_unit = 'nm',
                  **kwargs):
        plt.close()

        unit_value, unit_name = unit_value_and_name_from_unit(distance_unit)

        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        color_mesh = self.attach_mesh_to_axis(axis, mesh, plot_limit = plot_limit, distance_unit = distance_unit)
        if overlay_probability_current:
            quiv = self.attach_probability_current_to_axis(axis, plot_limit = plot_limit, distance_unit = distance_unit)

        axis.set_xlabel(r'$z$ ({})'.format(unit_name), fontsize = 15)
        axis.set_ylabel(r'$\rho$ ({})'.format(unit_name), fontsize = 15)
        if title is not None:
            title = axis.set_title(title, fontsize = 15)
            title.set_y(1.05)  # move title up a bit

        # make a colorbar
        cbar = fig.colorbar(mappable = color_mesh, ax = axis)
        cbar.ax.tick_params(labelsize = 10)

        axis.axis('tight')  # removes blank space between color mesh and axes

        axis.grid(True, color = GRID_COLOR, linestyle = ':')  # change grid color to make it show up against the colormesh

        axis.tick_params(labelright = True, labeltop = True)  # ticks on all sides
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)  # increase size of tick labels
        axis.tick_params(axis = 'both', which = 'both', length = 0)

        # set upper and lower y ticks to not display to avoid collisions with the x ticks at the edges
        y_ticks = axis.yaxis.get_major_ticks()
        y_ticks[0].label1.set_visible(False)
        y_ticks[0].label2.set_visible(False)
        y_ticks[-1].label1.set_visible(False)
        y_ticks[-1].label2.set_visible(False)

        cp.utils.save_current_figure(name = '{}_{}'.format(self.spec.name, name), target_dir = target_dir, **kwargs)

        plt.close()


class SphericalSliceSpecification(ElectricFieldSpecification):
    def __init__(self, name,
                 r_bound = 20 * bohr_radius,
                 r_points = 2 ** 10, theta_points = 2 ** 10,
                 **kwargs):
        super(SphericalSliceSpecification, self).__init__(name, mesh_type = SphericalSliceMesh, **kwargs)

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
    mesh_storage_method = ('r', 'theta')

    def __init__(self, simulation):
        super(SphericalSliceMesh, self).__init__(simulation)

        self.r = np.linspace(0, self.spec.r_bound, self.spec.r_points)
        self.theta = np.delete(np.linspace(0, pi, self.spec.theta_points + 1), 0)

        self.delta_r = self.r[1] - self.r[0]
        self.delta_theta = self.theta[1] - self.theta[0]
        self.inner_product_multiplier = self.delta_r * self.delta_theta

        self.r += self.delta_r / 2
        self.theta -= self.delta_theta / 2

        self.r_max = np.max(self.r)

        # self.r_mesh, self.theta_mesh = np.meshgrid(self.r, self.theta, indexing = 'ij')
        self.g_mesh = self.get_g_for_state(self.spec.initial_state)

        self.mesh_points = len(self.r) * len(self.theta)
        self.matrix_operator_shape = (self.mesh_points, self.mesh_points)
        self.mesh_shape = np.shape(self.r_mesh)

    @property
    @cp.utils.memoize
    def r_mesh(self):
        return np.meshgrid(self.r, self.theta, indexing = 'ij')[0]

    @property
    @cp.utils.memoize
    def theta_mesh(self):
        return np.meshgrid(self.r, self.theta, indexing = 'ij')[1]

    @property
    def g_factor(self):
        return np.sqrt(twopi * np.sin(self.theta_mesh)) * self.r_mesh

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

    def state_overlap(self, state_a = None, state_b = None):
        """State overlap between two states. If either state is None, the state on the g_mesh is used for that state."""
        if state_a is None:
            mesh_a = self.g_mesh
        else:
            mesh_a = self.get_g_for_state(state_a)
        if state_b is None:
            mesh_b = self.g_mesh
        else:
            mesh_b = self.get_g_for_state(state_b)

        return np.abs(self.inner_product(mesh_a, mesh_b)) ** 2

    @property
    def norm(self):
        return np.abs(self.inner_product())

    @cp.utils.memoize
    def get_g_for_state(self, state):
        return self.g_factor * state(self.r_mesh, self.theta_mesh, 0)

    def dipole_moment_expectation_value(self, gauge = 'length'):
        """Get the dipole moment in the specified gauge."""
        if gauge == 'length':
            return self.spec.test_charge * self.inner_product(mesh_b = self.z_mesh * self.g_mesh)
        elif gauge == 'velocity':
            raise NotImplementedError

    @cp.utils.memoize
    def _get_kinetic_energy_matrix_operators(self):
        r_prefactor = -(hbar ** 2) / (2 * electron_mass_reduced * (self.delta_r ** 2))
        theta_prefactor = -(hbar ** 2) / (2 * electron_mass_reduced * ((self.delta_r * self.delta_theta) ** 2))

        r_diagonal = r_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        r_offdiagonal = r_prefactor * np.array([1 if (z_index + 1) % self.spec.r_points != 0 else 0 for z_index in range(self.mesh_points - 1)], dtype = np.complex128)

        @cp.utils.memoize
        def theta_j_prefactor(x):
            return 1 / (x + 0.5) ** 2

        @cp.utils.memoize
        def sink(x):
            return np.sin(x * self.delta_theta)

        @cp.utils.memoize
        def sqrt_sink_ratio(x_num, x_den):
            return np.sqrt(sink(x_num) / sink(x_den))

        @cp.utils.memoize
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

    @cp.utils.memoize
    def get_internal_hamiltonian_matrix_operators(self):
        r_kinetic, theta_kinetic = self._get_kinetic_energy_matrix_operators()
        potential_mesh = self.spec.internal_potential(r = self.r_mesh, test_charge = self.spec.test_charge)

        r_kinetic.data[1] += 0.5 * self.flatten_mesh(potential_mesh, 'r')
        theta_kinetic.data[1] += 0.5 * self.flatten_mesh(potential_mesh, 'theta')

        return r_kinetic, theta_kinetic

    def tg_mesh(self, use_abs_g = False):
        hamiltonian_r, hamiltonian_theta = self._get_kinetic_energy_matrix_operators()

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

    @cp.utils.memoize
    def get_probability_current_matrix_operators(self):
        raise NotImplementedError

    def get_probability_current_vector_field(self):
        raise NotImplementedError

    def get_spline_for_mesh(self, mesh):
        return sp.interp.RectBivariateSpline(self.r, self.theta, mesh)

    def evolve(self, time_step):
        tau = time_step / (2 * hbar)

        electric_potential_energy_mesh = self.spec.electric_potential(t = self.sim.time, distance_along_polarization = self.z_mesh, test_charge = self.spec.test_charge)

        # add the external potential to the Hamiltonian matrices and multiply them by i * tau to get them ready for the next steps
        hamiltonian_r, hamiltonian_theta = self.get_internal_hamiltonian_matrix_operators()
        hamiltonian_r, hamiltonian_theta = hamiltonian_r.copy(), hamiltonian_theta.copy()

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


    @cp.utils.memoize
    def get_mesh_slicer(self, distance_from_center = None):
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if distance_from_center is None:
            mesh_slicer = slice(None, None, 1)
        else:
            r_lim_points = round(distance_from_center / self.delta_r)
            mesh_slicer = slice(0, int(r_lim_points + 1), 1)

        return mesh_slicer

    def attach_mesh_to_axis(self, axis, mesh, plot_limit = None, distance_unit = 'bohr_radius', **kwargs):
        unit_value, unit_name = unit_value_and_name_from_unit(distance_unit)

        color_mesh = axis.pcolormesh(self.theta_mesh[self.get_mesh_slicer(plot_limit)],
                                     self.r_mesh[self.get_mesh_slicer(plot_limit)] / unit_value,
                                     mesh[self.get_mesh_slicer(plot_limit)],
                                     shading = 'gouraud',
                                     **kwargs)
        color_mesh_mirror = axis.pcolormesh(twopi - self.theta_mesh[self.get_mesh_slicer(plot_limit)],
                                            self.r_mesh[self.get_mesh_slicer(plot_limit)] / unit_value,
                                            mesh[self.get_mesh_slicer(plot_limit)],
                                            shading = 'gouraud',
                                            **kwargs)  # another colormesh, mirroring the first mesh onto pi to 2pi

        return color_mesh, color_mesh_mirror

    def attach_probability_current_to_axis(self, axis, plot_limit = None, distance_unit = 'bohr_radius'):
        raise NotImplementedError

    def plot_mesh(self, mesh,
                  name = '', target_dir = None, img_format = 'png', title = None,
                  overlay_probability_current = False, probability_current_time_step = 0, plot_limit = None,
                  distance_unit = 'nm',
                  **kwargs):
        plt.close()  # close any old figures

        unit_value, unit_name = unit_value_and_name_from_unit(distance_unit)

        fig = plt.figure(figsize = (7, 7), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111, projection = 'polar')
        axis.set_theta_zero_location('N')
        axis.set_theta_direction('clockwise')

        color_mesh, color_mesh_mirror = self.attach_mesh_to_axis(axis, mesh, plot_limit = plot_limit, distance_unit = distance_unit)
        if overlay_probability_current:
            quiv = self.attach_probability_current_to_axis(axis, plot_limit = plot_limit, distance_unit = distance_unit)

        if title is not None:
            title = axis.set_title(title, fontsize = 15)
            title.set_x(.03)  # move title to the upper left corner
            title.set_y(.97)

        # make a colorbar
        cbar_axis = fig.add_axes([1.01, .1, .04, .8])  # add a new axis for the cbar so that the old axis can stay square
        cbar = plt.colorbar(mappable = color_mesh, cax = cbar_axis)
        cbar.ax.tick_params(labelsize = 10)

        axis.grid(True, color = GRID_COLOR, linestyle = ':')  # change grid color to make it show up against the colormesh
        angle_labels = ['{}\u00b0'.format(s) for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
        axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)

        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)  # increase size of tick labels
        axis.tick_params(axis = 'y', which = 'major', colors = GRID_COLOR, pad = 3)  # make r ticks a color that shows up against the colormesh
        axis.tick_params(axis = 'both', which = 'both', length = 0)

        axis.set_rlabel_position(80)

        max_yticks = 5
        yloc = plt.MaxNLocator(max_yticks, symmetric = False, prune = 'both')
        axis.yaxis.set_major_locator(yloc)

        fig.canvas.draw()  # must draw early to modify the axis text

        tick_labels = axis.get_yticklabels()
        for t in tick_labels:
            t.set_text(t.get_text() + r'{}'.format(unit_name))
        axis.set_yticklabels(tick_labels)

        axis.set_rmax((self.r_max - (self.delta_r / 2)) / unit_value)

        cp.utils.save_current_figure(name = '{}_{}'.format(self.spec.name, name), target_dir = target_dir, img_format = img_format, **kwargs)

        plt.close()


class SphericalHarmonicSpecification(ElectricFieldSpecification):
    def __init__(self, name,
                 r_bound = 20 * bohr_radius,
                 r_points = 2 ** 9,
                 l_points = 2 ** 5,
                 evolution_equations = 'L',
                 evolution_method = 'SO',
                 **kwargs):
        """
        Specification for an ElectricFieldSimulation using a SphericalHarmonicMesh.

        :param name:
        :param r_bound:
        :param r_points:
        :param l_points:
        :param evolution_equations: 'L' (recommended) or 'H'
        :param evolution_method: 'SO' (recommended) or 'CN'
        :param kwargs:
        """
        super(SphericalHarmonicSpecification, self).__init__(name, mesh_type = SphericalHarmonicMesh, **kwargs)

        self.r_bound = r_bound
        self.r_points = int(r_points)
        self.l_points = l_points
        self.spherical_harmonics = tuple(cp.math.SphericalHarmonic(l, 0) for l in range(self.l_points))

        self.evolution_equations = evolution_equations
        self.evolution_method = evolution_method

    def info(self):
        mesh = ['Mesh: {}'.format(self.mesh_type.__name__),
                '   Evolution: {} equations, {} method'.format(self.evolution_equations, self.evolution_method),
                '   R Boundary: {} Bohr radii'.format(uround(self.r_bound, bohr_radius, 3)),
                '   R Points: {}'.format(self.r_points),
                '   R Mesh Spacing: ~{} Bohr radii'.format(uround(self.r_bound / self.r_points, bohr_radius, 3)),
                '   Spherical Harmonics: {}'.format(self.l_points),
                '   Total Mesh Points: {}'.format(self.r_points * self.l_points)]

        return '\n'.join((super(SphericalHarmonicSpecification, self).info(), *mesh))


class SphericalHarmonicMesh(QuantumMesh):
    mesh_storage_method = ('l', 'r')

    def __init__(self, simulation):
        super(SphericalHarmonicMesh, self).__init__(simulation)

        self.r = np.linspace(0, self.spec.r_bound, self.spec.r_points)
        self.delta_r = self.r[1] - self.r[0]
        self.r += self.delta_r / 2
        self.r_max = np.max(self.r)
        self.inner_product_multiplier = self.delta_r

        self.l = np.array(range(self.spec.l_points))
        self.theta_points = min(self.spec.l_points * 5, 360)
        self.phi_points = self.theta_points

        self.mesh_points = len(self.r) * len(self.l)
        self.mesh_shape = np.shape(self.r_mesh)

        self.g_mesh = self.get_g_for_state(self.spec.initial_state)

    @property
    @cp.utils.memoize
    def r_mesh(self):
        return np.meshgrid(self.l, self.r, indexing = 'ij')[1]

    @property
    @cp.utils.memoize
    def l_mesh(self):
        return np.meshgrid(self.l, self.r, indexing = 'ij')[0]

    @property
    def g_factor(self):
        return self.r

    @cp.utils.memoize
    def c(self, l):
        return (l + 1) / np.sqrt(((2 * l) + 1) * ((2 * l) + 3))

    def flatten_mesh(self, mesh, flatten_along):
        """Return a mesh flattened along one of the mesh coordinates ('theta' or 'r')."""
        try:
            if flatten_along == 'l':
                flat = 'F'
            elif flatten_along == 'r':
                flat = 'C'
            else:
                raise ValueError("{} is not a valid specifier for flatten_mesh (valid specifiers: 'l', 'r')".format(flatten_along))

            return mesh.flatten(flat)
        except AttributeError:  # occurs if the "mesh" is actually an int or float, in which case we should should just return it
            return mesh

    def wrap_vector(self, mesh, wrap_along):
        if wrap_along == 'l':
            wrap = 'F'
        elif wrap_along == 'r':
            wrap = 'C'
        else:
            raise ValueError("{} is not a valid specifier for wrap_vector (valid specifiers: 'l', 'r')".format(wrap_along))

        return np.reshape(mesh, self.mesh_shape, wrap)

    @property
    def norm_by_l(self):
        return np.abs(np.sum(np.conj(self.g_mesh) * self.g_mesh, axis = 1) * self.delta_r)

    def dipole_moment_expectation_value(self, gauge = 'length'):
        """Get the dipole moment in the specified gauge."""
        if gauge == 'length':
            _, operator = self._get_kinetic_energy_matrix_operators()
            g = self.wrap_vector(operator.dot(self.flatten_mesh(self.g_mesh, 'l')), 'l')
            return self.spec.test_charge * self.inner_product(mesh_b = g)
        elif gauge == 'velocity':
            raise NotImplementedError

    @cp.utils.memoize
    def get_g_for_state(self, state):
        g = np.zeros(self.mesh_shape, dtype = np.complex128)

        for s in state:
            g[s.l, :] += s.radial_function(self.r) * self.g_factor

        return g

    @cp.utils.memoize
    def _get_kinetic_energy_matrix_operators(self):
        return getattr(self, '_get_kinetic_energy_matrix_operators_' + self.spec.evolution_equations)()

    def _get_kinetic_energy_matrix_operators_H(self):
        r_prefactor = -(hbar ** 2) / (2 * electron_mass_reduced * (self.delta_r ** 2))
        l_prefactor = self.flatten_mesh(self.r_mesh, 'l')[:-1]  # TODO: ???

        r_diagonal = r_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        r_offdiagonal = r_prefactor * np.ones(self.mesh_points - 1, dtype = np.complex128)

        @cp.utils.memoize
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

    def _get_kinetic_energy_matrix_operators_L(self):
        r_prefactor = -(hbar ** 2) / (2 * electron_mass_reduced * (self.delta_r ** 2))
        l_prefactor = self.flatten_mesh(self.r_mesh, 'l')[:-1]  # TODO: ???

        @cp.utils.memoize
        def alpha(j):
            x = (j ** 2) + (2 * j)
            return (x + 1) / (x + 0.75)

        @cp.utils.memoize
        def beta(j):
            x = 2 * (j ** 2) + (2 * j)
            return (x + 1) / (x + 0.5)

        r_diagonal = np.zeros(self.mesh_points, dtype = np.complex128)
        r_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for r_index in range(self.mesh_points):
            j = r_index % self.spec.r_points
            r_diagonal[r_index] = beta(j)

        for r_index in range(self.mesh_points - 1):
            if (r_index + 1) % self.spec.r_points != 0:  # TODO: should be possible to clean this if up
                j = (r_index % self.spec.r_points)
                r_offdiagonal[r_index] = alpha(j)
        r_diagonal *= -2 * r_prefactor
        r_offdiagonal *= r_prefactor

        l_diagonal = np.zeros(self.mesh_points, dtype = np.complex128)
        l_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for l_index in range(self.mesh_points - 1):
            if (l_index + 1) % self.spec.l_points != 0:
                l = (l_index % self.spec.l_points) + 1
                l_offdiagonal[l_index] = self.c(l)
        l_offdiagonal *= l_prefactor

        effective_potential_mesh = ((hbar ** 2) / (2 * electron_mass_reduced)) * self.l_mesh * (self.l_mesh + 1) / (self.r_mesh ** 2)
        r_diagonal += self.flatten_mesh(effective_potential_mesh, 'r')

        r_kinetic = sparse.diags([r_offdiagonal, r_diagonal, r_offdiagonal], offsets = (-1, 0, 1))
        l_kinetic = sparse.diags([l_offdiagonal, l_diagonal, l_offdiagonal], offsets = (-1, 0, 1))

        return r_kinetic, l_kinetic

    @cp.utils.memoize
    def _get_internal_hamiltonian_matrix_operators(self):
        r_kinetic, l_kinetic = self._get_kinetic_energy_matrix_operators()
        potential_mesh = self.spec.internal_potential(r = self.r_mesh, test_charge = self.spec.test_charge)

        r_kinetic.data[1] += self.flatten_mesh(potential_mesh, 'r')

        return r_kinetic, l_kinetic

    def tg_mesh(self, use_abs_g = False):
        hamiltonian_r, hamiltonian_l = self._get_kinetic_energy_matrix_operators()

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
        hamiltonian_r, hamiltonian_l = self._get_internal_hamiltonian_matrix_operators()

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

    @cp.utils.memoize
    def get_probability_current_matrix_operators(self):
        raise NotImplementedError

    def get_probability_current_vector_field(self):
        raise NotImplementedError

    def _evolve_CN(self, time_step):
        tau = time_step / (2 * hbar)

        electric_field_amplitude = self.spec.electric_potential.get_electric_field_amplitude(self.sim.time)
        l_multiplier = self.spec.test_charge * electric_field_amplitude

        hamiltonian_r, hamiltonian_l = self._get_internal_hamiltonian_matrix_operators()
        hamiltonian_r, hamiltonian_l = hamiltonian_r.copy(), hamiltonian_l.copy()
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

    def _get_split_operator_evolution_matrices(self, a):
        a_even, a_odd = a[::2], a[1::2]

        even_diag = np.zeros(len(a) + 1, dtype = np.complex128)
        even_diag[:] = np.cos(a_even).repeat(2)

        even_offdiag = np.zeros(len(a), dtype = np.complex128)
        even_offdiag[0::2] = -1j * np.sin(a_even)

        odd_diag = np.zeros(len(a) + 1, dtype = np.complex128)
        odd_diag[0] = odd_diag[-1] = 1
        odd_diag[1:-1] = np.cos(a_odd).repeat(2)

        odd_offdiag = np.zeros(len(a), dtype = np.complex128)
        odd_offdiag[1::2] = -1j * np.sin(a_odd)

        even = sparse.diags([even_offdiag, even_diag, even_offdiag], offsets = [-1, 0, 1])
        odd = sparse.diags([odd_offdiag, odd_diag, odd_offdiag], offsets = [-1, 0, 1])

        return even, odd

    def _evolve_SO(self, time_step):
        tau = time_step / (2 * hbar)

        electric_field_amplitude = self.spec.electric_potential.get_electric_field_amplitude(self.sim.time)
        l_multiplier = self.spec.test_charge * electric_field_amplitude

        hamiltonian_r, hamiltonian_l = self._get_internal_hamiltonian_matrix_operators()
        hamiltonian_r, hamiltonian_l = hamiltonian_r.copy(), hamiltonian_l.copy()
        hamiltonian_r *= 1j * tau
        hamiltonian_l.data[0] *= tau * l_multiplier

        even, odd = self._get_split_operator_evolution_matrices(hamiltonian_l.data[0][:-1])

        # STEP 1
        g_vector = self.flatten_mesh(self.g_mesh, 'l')
        g_vector = even.dot(g_vector)

        # STEP 2
        g_vector = odd.dot(g_vector)
        self.g_mesh = self.wrap_vector(g_vector, 'l')

        # STEP 3
        g_vector = self.flatten_mesh(self.g_mesh, 'r')
        hamiltonian = -1 * hamiltonian_r
        hamiltonian.data[1] += 1  # add identity to matrix operator
        g_vector = hamiltonian.dot(g_vector)

        # STEP 4
        hamiltonian = hamiltonian_r.copy()
        hamiltonian.data[1] += 1  # add identity to matrix operator
        g_vector = cy.tdma(hamiltonian, g_vector)
        self.g_mesh = self.wrap_vector(g_vector, 'r')

        # STEP 5
        g_vector = self.flatten_mesh(self.g_mesh, 'l')
        g_vector = odd.dot(g_vector)

        # STEP 6
        g_vector = even.dot(g_vector)
        self.g_mesh = self.wrap_vector(g_vector, 'l')

    @cp.utils.memoize
    def get_mesh_slicer(self, distance_from_center = None):
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if distance_from_center is None:
            mesh_slicer = (slice(None, None, 1), slice(None, None, 1))
        else:
            r_lim_points = int(distance_from_center / self.delta_r)
            mesh_slicer = (slice(None, None, 1), slice(0, int(r_lim_points + 1), 1))

        return mesh_slicer

    @cp.utils.memoize
    def get_mesh_slicer_spatial(self, distance_from_center = None):
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if distance_from_center is None:
            mesh_slicer = slice(None, None, 1)
        else:
            r_lim_points = int(distance_from_center / self.delta_r)
            mesh_slicer = slice(0, int(r_lim_points + 1), 1)

        return mesh_slicer

    @property
    @cp.utils.memoize
    def theta(self):
        return np.linspace(0, twopi, self.theta_points)

    @property
    @cp.utils.memoize
    def theta_mesh(self):
        return np.meshgrid(self.r, self.theta, indexing = 'ij')[1]

    @property
    @cp.utils.memoize
    def r_theta_mesh(self):
        return np.meshgrid(self.r, self.theta, indexing = 'ij')[0]

    @property
    # @cp.utils.memoize
    def theta_l_r_meshes(self):
        return np.meshgrid(self.theta, self.l, self.r, indexing = 'ij')

    @property
    @cp.utils.memoize
    def _sph_harm_l_theta_mesh(self):
        theta_mesh, l_mesh, _ = self.theta_l_r_meshes
        return special.sph_harm(0, l_mesh, 0, theta_mesh)

    def _reconstruct_spatial_mesh(self, mesh):
        """Reconstruct the spatial (r, theta) representation of a mesh from the (r, l) representation."""
        return np.sum(np.tile(mesh, (self.theta_points, 1, 1)) * self._sph_harm_l_theta_mesh, axis = 1).T

    @property
    @cp.utils.watcher(lambda s: s.sim.time)
    def space_g(self):
        return self._reconstruct_spatial_mesh(self.g_mesh)

    @property
    @cp.utils.watcher(lambda s: s.sim.time)
    def space_psi(self):
        return self._reconstruct_spatial_mesh(self.psi_mesh)  # TODO: should reference space_g, remove watcher

    def abs_g_squared(self, normalize = False, log = False):
        out = np.abs(self.space_g) ** 2
        if normalize:
            out /= np.nanmax(out)
        if log:
            out = np.log10(out)

        return out

    def abs_psi_squared(self, normalize = False, log = False):
        out = np.abs(self.space_psi) ** 2
        if normalize:
            out /= np.nanmax(out)
        if log:
            out = np.log10(out)

        return out

    def attach_mesh_to_axis(self, axis, mesh, plot_limit = None, color_map_min = 0, distance_unit = 'bohr_radius', **kwargs):
        unit_value, unit_name = unit_value_and_name_from_unit(distance_unit)

        color_mesh = axis.pcolormesh(self.theta_mesh[self.get_mesh_slicer_spatial(plot_limit)],
                                     self.r_theta_mesh[self.get_mesh_slicer_spatial(plot_limit)] / unit_value,
                                     mesh[self.get_mesh_slicer_spatial(plot_limit)],
                                     shading = 'gouraud', vmin = color_map_min,
                                     **kwargs)

        return color_mesh

    def attach_probability_current_to_axis(self, axis, plot_limit = None, distance_unit = 'bohr_radius'):
        raise NotImplementedError

    def plot_mesh(self, mesh,
                  name = '', target_dir = None, img_format = 'png', title = None,
                  overlay_probability_current = False, probability_current_time_step = 0, plot_limit = None,
                  color_map_min = 0,
                  distance_unit = 'bohr_radius',
                  **kwargs):
        plt.close()  # close any old figures

        unit_value, unit_name = unit_value_and_name_from_unit(distance_unit)

        fig = plt.figure(figsize = (7, 7), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111, projection = 'polar')
        axis.set_theta_zero_location('N')
        axis.set_theta_direction('clockwise')

        color_mesh = self.attach_mesh_to_axis(axis, mesh, plot_limit = plot_limit, color_map_min = color_map_min, distance_unit = distance_unit)
        if overlay_probability_current:
            quiv = self.attach_probability_current_to_axis(axis, plot_limit = plot_limit, distance_unit = distance_unit)

        if title is not None:
            title = axis.set_title(title, fontsize = 15)
            title.set_x(.03)  # move title to the upper left corner
            title.set_y(.97)

        # make a colorbar
        cbar_axis = fig.add_axes([1.01, .1, .04, .8])  # add a new axis for the cbar so that the old axis can stay square
        cbar = plt.colorbar(mappable = color_mesh, cax = cbar_axis)
        cbar.ax.tick_params(labelsize = 10)

        axis.grid(True, color = GRID_COLOR, linestyle = ':')  # change grid color to make it show up against the colormesh
        angle_labels = ['{}\u00b0'.format(s) for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
        axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)

        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)  # increase size of tick labels
        axis.tick_params(axis = 'y', which = 'major', colors = GRID_COLOR, pad = 3)  # make r ticks a color that shows up against the colormesh
        axis.tick_params(axis = 'both', which = 'both', length = 0)

        axis.set_rlabel_position(80)

        max_yticks = 5
        yloc = plt.MaxNLocator(max_yticks, symmetric = False, prune = 'both')
        axis.yaxis.set_major_locator(yloc)

        fig.canvas.draw()  # must draw early to modify the axis text

        tick_labels = axis.get_yticklabels()
        for t in tick_labels:
            t.set_text(t.get_text() + r'{}'.format(unit_name))
        axis.set_yticklabels(tick_labels)

        axis.set_rmax((self.r_max - (self.delta_r / 2)) / unit_value)

        cp.utils.save_current_figure(name = '{}_{}'.format(self.spec.name, name), target_dir = target_dir, img_format = img_format, **kwargs)

        plt.close()


class ElectricFieldSimulation(cp.core.Simulation):
    def __init__(self, spec):
        super(ElectricFieldSimulation, self).__init__(spec)

        self.mesh = None
        self.animators = self.spec.animators

        self.initialize_mesh()

        total_time = self.spec.time_final - self.spec.time_initial
        self.times = np.linspace(self.spec.time_initial, self.spec.time_final, int(total_time / self.spec.time_step) + 1)
        if self.spec.time_final < self.spec.minimum_time_final:
            extra_times = np.delete(np.linspace(self.spec.time_final, self.spec.minimum_time_final, ((self.spec.minimum_time_final - self.spec.time_final) / self.spec.extra_time_step) + 1), 0)
            self.times = np.concatenate((self.times, extra_times))
        self.time_index = 0
        self.time_steps = len(self.times)

        # simulation data storage
        self.norm_vs_time = np.zeros(self.time_steps, dtype = np.float64) * np.NaN
        self.norm_diff_mask_vs_time = np.zeros(self.time_steps, dtype = np.float64) * np.NaN
        self.energy_expectation_value_vs_time_internal = np.zeros(self.time_steps, dtype = np.float64) * np.NaN
        self.inner_products_vs_time = {state: np.zeros(self.time_steps, dtype = np.complex128) * np.NaN for state in self.spec.test_states}
        self.electric_field_amplitude_vs_time = np.zeros(self.time_steps, dtype = np.float64) * np.NaN
        self.electric_dipole_moment_vs_time = {gauge: np.zeros(self.time_steps, dtype = np.complex128) * np.NaN for gauge in self.spec.dipole_gauges}

        # optional data storage
        if 'l' in self.mesh.mesh_storage_method and self.spec.store_norm_by_l:
            self.norm_by_harmonic_vs_time = {sph_harm: np.zeros(self.time_steps, dtype = np.float64) * np.NaN for sph_harm in self.spec.spherical_harmonics}

    @property
    def available_animation_frames(self):
        return self.time_steps

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

        logger.debug('Initialized mesh for {} {}'.format(self.__class__.__name__, self.name))

        if not (.99 < self.mesh.norm < 1.01):
            logger.warning('Initial wavefunction for {} {} may not be normalized (norm = {})'.format(self.__class__.__name__, self.name, self.mesh.norm))

    def store_data(self, time_index):
        """Update the time-indexed data arrays with the current values."""
        norm = self.mesh.norm
        self.norm_vs_time[time_index] = norm
        if norm > 1.001 * self.norm_vs_time[0]:
            logger.warning('Wavefunction norm ({}) has exceeded initial norm ({}) by more than .1% for {} {}'.format(norm, self.norm_vs_time[0], self.__class__.__name__, self.name))
        try:
            if norm > 1.001 * self.norm_vs_time[time_index - 1]:
                logger.warning('Wavefunction norm ({}) at time_index = {} has exceeded norm from previous time step ({}) by more than .1% for {} {}'.format(norm, time_index, self.norm_vs_time[time_index - 1], self.__class__.__name__, self.name))
        except IndexError:
            pass

        self.energy_expectation_value_vs_time_internal[time_index] = self.mesh.energy_expectation_value

        for gauge in self.spec.dipole_gauges:
            self.electric_dipole_moment_vs_time[gauge][time_index] = self.mesh.dipole_moment_expectation_value(gauge = gauge)

        for state in self.spec.test_states:
            self.inner_products_vs_time[state][time_index] = self.mesh.inner_product(self.mesh.get_g_for_state(state))

        self.electric_field_amplitude_vs_time[time_index] = self.spec.electric_potential.get_electric_field_amplitude(t = self.times[time_index])

        if 'l' in self.mesh.mesh_storage_method and self.spec.store_norm_by_l:
            norm_by_l = self.mesh.norm_by_l
            for sph_harm, l_norm in zip(self.spec.spherical_harmonics, norm_by_l):
                self.norm_by_harmonic_vs_time[sph_harm][time_index] = l_norm  # TODO: extend to non-l storage meshes

            largest_l = self.spec.spherical_harmonics[-1]
            norm_in_largest_l = self.norm_by_harmonic_vs_time[self.spec.spherical_harmonics[-1]][time_index]
            if norm_in_largest_l > self.norm_vs_time[self.time_index] / 1000:
                logger.warning('Wavefunction norm in largest angular momentum state is large (l = {}, norm = {}), consider increasing l bound'.format(largest_l, norm_in_largest_l))

        logger.debug('{} {} stored data for time index {}'.format(self.__class__.__name__, self.name, time_index))

    def run_simulation(self, store_intermediate_meshes = False):
        """
        Run the simulation by repeatedly evolving the mesh by the time step and recovering various data from it.

        :param only_end_data: only store data from the last time step (does not change memory/disk usage, only runtime)
        :param store_intermediate_meshes: store the mesh at every time step (very memory-intensive)
        :return: None
        """
        logger.info('Performing time evolution on {} ({})'.format(self.name, self.file_name))
        try:
            for animator in self.animators:
                animator.initialize(self)

            self.status = 'running'
            logger.debug("{} {} ({}) status set to 'running'".format(self.__class__.__name__, self.name, self.file_name))

            while True:
                self.store_data(self.time_index)

                for animator in self.animators:
                    if self.time_index == 0 or self.time_index == self.time_steps or self.time_index % animator.decimation == 0:
                        animator.send_frame_to_ffmpeg()

                if self.time_index == self.time_steps - 1:
                    break

                self.time_index += 1

                norm_diff_mask = self.mesh.evolve(self.times[self.time_index] - self.times[self.time_index - 1])  # evolve the mesh forward to the next time step
                self.norm_diff_mask_vs_time[self.time_index] = norm_diff_mask  # move to store data so it has the right index?

                logger.debug('{} {} ({}) evolved to time index {} / {} ({}%)'.format(self.__class__.__name__, self.name, self.file_name, self.time_index, self.time_steps - 1,
                                                                                     np.around(100 * (self.time_index + 1) / self.time_steps, 2)))

                if self.spec.checkpoints:
                    if (self.time_index + 1) % self.spec.checkpoint_every == 0:
                        self.save(target_dir = self.spec.checkpoint_dir, save_mesh = True)
                        logger.info('Checkpointed {} {} ({}) at time step {} / {}'.format(self.__class__.__name__, self.name, self.file_name, self.time_index + 1, self.time_steps))

            self.end_time = dt.datetime.now()
            self.elapsed_time = self.end_time - self.start_time

            self.status = 'finished'
            logger.debug("{} {} ({}) status set to 'finished'".format(self.__class__.__name__, self.name, self.file_name))
            logger.info('Finished performing time evolution on {} {} ({})'.format(self.__class__.__name__, self.name, self.file_name))
        except Exception as e:
            raise e
        finally:
            # make sure the animators get cleaned up if there's some kind of error during time evolution
            for animator in self.animators:
                animator.cleanup()

    def plot_wavefunction_vs_time(self, use_name = False, grayscale = False, log = False, x_scale = 'asec', **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)

        x_scale_unit = unit_names_to_values[x_scale]  # TODO: use the same names as in xy_plot()
        x_scale_name = unit_names_to_tex_strings[x_scale]

        grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [4, 1], hspace = 0.06)  # TODO: switch to fixed axis construction
        ax_overlaps = plt.subplot(grid_spec[0])
        ax_field = plt.subplot(grid_spec[1], sharex = ax_overlaps)

        if not isinstance(self.spec.electric_potential, potentials.NoPotentialEnergy):
            ax_field.plot(self.times / x_scale_unit, self.electric_field_amplitude_vs_time / atomic_electric_field, color = 'black', linewidth = 2)

        ax_overlaps.plot(self.times / x_scale_unit, self.norm_vs_time, label = r'$\left\langle \psi|\psi \right\rangle$', color = 'black', linewidth = 3)

        if grayscale:  # stackplot with two lines in grayscale (initial and non-initial)
            initial_overlap = [self.state_overlaps_vs_time[self.spec.initial_state]]
            non_initial_overlaps = [self.state_overlaps_vs_time[state] for state in self.spec.test_states if state != self.spec.initial_state]
            total_non_initial_overlaps = functools.reduce(np.add, non_initial_overlaps)
            overlaps = [initial_overlap, total_non_initial_overlaps]
            ax_overlaps.stackplot(self.times / x_scale_unit, *overlaps, alpha = 1,
                                  labels = [r'$\left| \left\langle \psi|\psi_{init} \right\rangle \right|^2$', r'$\left| \left\langle \psi|\psi_{n\leq5} \right\rangle \right|^2$'],
                                  colors = ['.3', '.5'])
        else:  # stackplot with all states broken out in full color
            overlaps = [self.state_overlaps_vs_time[state] for state in self.spec.test_states]
            num_colors = len(overlaps)
            ax_overlaps.set_prop_cycle(cycler('color', [plt.get_cmap('gist_rainbow')(n / num_colors) for n in range(num_colors)]))
            ax_overlaps.stackplot(self.times / x_scale_unit, *overlaps, alpha = 1, labels = [r'$\left| \left\langle \psi| {} \right\rangle \right|^2$'.format(state.tex_str) for state in self.spec.test_states])

        if log:
            ax_overlaps.set_yscale('log')
            min_overlap = np.min(self.state_overlaps_vs_time[self.spec.initial_state])
            ax_overlaps.set_ylim(bottom = max(1e-9, min_overlap * .1), top = 1.0)
            ax_overlaps.grid(True, which = 'both')
        else:
            ax_overlaps.set_ylim(0, 1.0)
            ax_overlaps.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax_overlaps.grid(True)
        ax_overlaps.set_xlim(self.spec.time_initial / x_scale_unit, self.spec.time_final / x_scale_unit)

        ax_field.grid(True)

        ax_field.set_xlabel('Time $t$ ({})'.format(x_scale_name), fontsize = 15)
        ax_overlaps.set_ylabel('Wavefunction Metric', fontsize = 15)
        ax_field.set_ylabel('E-Field (a.u.)', fontsize = 11)

        if grayscale:
            ax_overlaps.legend(loc = 'lower left', fontsize = 12)
        else:
            ax_overlaps.legend(bbox_to_anchor = (1.1, 1), loc = 'upper left', borderaxespad = 0., fontsize = 10, ncol = 1 + (len(self.spec.test_states) // 17))

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
        if log:
            postfix += '_log'
        prefix = self.file_name
        if use_name:
            prefix = self.name
        cp.utils.save_current_figure(name = prefix + '__wavefunction_vs_time{}'.format(postfix), **kwargs)

        plt.close()

    def plot_angular_momentum_vs_time(self, use_name = False, log = False, renormalize = False, **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)

        grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [4, 1], hspace = 0.06)
        ax_momentums = plt.subplot(grid_spec[0])
        ax_field = plt.subplot(grid_spec[1], sharex = ax_momentums)

        if not isinstance(self.spec.electric_potential, potentials.NoPotentialEnergy):
            ax_field.plot(self.times / asec, self.electric_field_amplitude_vs_time / atomic_electric_field, color = 'black', linewidth = 2)

        if renormalize:
            overlaps = [self.norm_by_harmonic_vs_time[sph_harm] / self.norm_vs_time for sph_harm in self.spec.spherical_harmonics]
            l_labels = [r'$\left| \left\langle \psi| {} \right\rangle \right|^2 / \left\langle \psi| \psi \right\rangle$'.format(sph_harm.tex_str) for sph_harm in self.spec.spherical_harmonics]
        else:
            overlaps = [self.norm_by_harmonic_vs_time[sph_harm] for sph_harm in self.spec.spherical_harmonics]
            l_labels = [r'$\left| \left\langle \psi| {} \right\rangle \right|^2$'.format(sph_harm.tex_str) for sph_harm in self.spec.spherical_harmonics]
        num_colors = len(overlaps)
        ax_momentums.set_prop_cycle(cycler('color', [plt.get_cmap('gist_rainbow')(n / num_colors) for n in range(num_colors)]))
        ax_momentums.stackplot(self.times / asec, *overlaps, alpha = 1, labels = l_labels)

        if log:
            ax_momentums.set_yscale('log')
            ax_momentums.set_ylim(top = 1.0)
            ax_momentums.grid(True, which = 'both')
        else:
            ax_momentums.set_ylim(0, 1.0)
            ax_momentums.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax_momentums.grid(True)
        ax_momentums.set_xlim(self.spec.time_initial / asec, self.spec.time_final / asec)

        ax_field.grid(True)

        ax_field.set_xlabel('Time $t$ (as)', fontsize = 15)
        y_label = r'$\left| \left\langle \psi | Y^l_0 \right\rangle \right|^2$'
        if renormalize:
            y_label += r'$/\left\langle\psi|\psi\right\rangle$'
        ax_momentums.set_ylabel(y_label, fontsize = 15)
        ax_field.set_ylabel('E-Field (a.u.)', fontsize = 11)

        ax_momentums.legend(bbox_to_anchor = (1.1, 1), loc = 'upper left', borderaxespad = 0., fontsize = 10, ncol = 1 + (len(self.spec.spherical_harmonics) // 17))

        ax_momentums.tick_params(labelright = True)
        ax_field.tick_params(labelright = True)
        ax_momentums.xaxis.tick_top()

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
        ax_momentums.tick_params(axis = 'both', which = 'major', labelsize = 10)

        postfix = ''
        if renormalize:
            postfix += '_renorm'
        prefix = self.file_name
        if use_name:
            prefix = self.name
        cp.utils.save_current_figure(name = prefix + '__angular_momentum_vs_time{}'.format(postfix), **kwargs)

        plt.close()

    def plot_dipole_moment_vs_time(self, gauge = 'length', use_name = False, **kwargs):
        if not use_name:
            prefix = self.file_name
        else:
            prefix = self.name
        cp.utils.xy_plot(prefix + '__dipole_moment_vs_time',
                         self.times, np.real(self.electric_dipole_moment_vs_time[gauge]),
                         x_scale = 'as', y_scale = 'atomic_electric_dipole',
                         x_label = 'Time $t$', y_label = 'Dipole Moment $d(t)$',
                         **kwargs)

    def dipole_moment_vs_frequency(self, gauge = 'length', first_time = None, last_time = None):
        logger.critical('ALERT: this method does not account for non-uniform time step!')

        if first_time is None:
            first_time_index, first_time = 0, self.times[0]
        else:
            first_time_index, first_time, _ = cp.utils.find_nearest_entry(self.times, first_time)
        if last_time is None:
            last_time_index, last_time = self.time_steps - 1, self.times[self.time_steps - 1]
        else:
            last_time_index, last_time, _ = cp.utils.find_nearest_entry(self.times, last_time)
        points = last_time_index - first_time_index
        frequency = nfft.fftshift(nfft.fftfreq(points, self.spec.time_step))
        dipole_moment = nfft.fftshift(nfft.fft(self.electric_dipole_moment_vs_time[gauge][first_time_index: last_time_index], norm = 'ortho'))

        return frequency, dipole_moment

    def plot_dipole_moment_vs_frequency(self, use_name = False, gauge = 'length', frequency_range = 10000 * THz, first_time = None, last_time = None, **kwargs):
        prefix = self.file_name
        if use_name:
            prefix = self.name
        frequency, dipole_moment = self.dipole_moment_vs_frequency(gauge = gauge, first_time = first_time, last_time = last_time)
        cp.utils.xy_plot(prefix + '__dipole_moment_vs_frequency',
                         frequency, np.abs(dipole_moment) ** 2,
                         x_scale = 'THz', y_scale = atomic_electric_dipole ** 2,
                         y_log_axis = True,
                         x_label = 'Frequency $f$', y_label = r'Dipole Moment $\left| d(\omega) \right|^2$ $\left( e^2 \, a_0^2 \right)$',
                         x_lower_limit = 0, x_upper_limit = frequency_range,
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
    def load(file_path, initialize_mesh = False, **kwargs):
        """Return a simulation loaded from the file_path. kwargs are for Beet.load."""
        sim = cp.core.Simulation.load(file_path, **kwargs)

        if initialize_mesh:
            sim.initialize_mesh()

        return sim
