import os

import compy as cp
from compy.units import *
import ionization as ion

import matplotlib as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

CMAP = plt.cm.inferno


def make_plots(spec):
    sim = ion.ElectricFieldSimulation(spec)
    units = ('nm', 'bohr_radius')
    for unit in units:
        sim.mesh.plot_g(name_postfix = '__' + unit, target_dir = OUT_DIR, colormap = CMAP, distance_unit = unit)
        # sim.mesh.plot_psi(name_postfix = '__' + unit, target_dir = OUT_DIR, colormap = CMAP, distance_unit = unit)
        print(sim.spec.mesh_type.__name__, sim.spec.initial_state, unit)


if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization') as logger:
        n = 5
        bound = 50
        angular_points = 100

        states = (ion.HydrogenBoundState(n, l) for n in range(n + 1) for l in range(n))

        specs = []

        line_potential = ion.FiniteSquareWell(potential_depth = 3 * eV, width = 1 * nm)
        for initial_state in ion.FiniteSquareWellState.all_states_of_well_from_parameters(3 * eV, 1 * nm, electron_mass):
            specs.append(ion.LineSpecification('line_mesh__{}'.format(initial_state.n),
                                               initial_state = initial_state,
                                               x_bound = 30 * nm))

        for initial_state in states:
            specs.append(ion.CylindricalSliceSpecification('cyl_slice__{}_{}'.format(initial_state.n, initial_state.l),
                                                           initial_state = initial_state,
                                                           z_bound = bound * bohr_radius, rho_bound = bound * bohr_radius))

            specs.append(ion.SphericalSliceSpecification('sph_slice__{}_{}'.format(initial_state.n, initial_state.l),
                                                         initial_state = initial_state,
                                                         r_bound = bound * bohr_radius, theta_points = angular_points))

            specs.append(ion.SphericalHarmonicSpecification('sph_harms__{}_{}'.format(initial_state.n, initial_state.l),
                                                            initial_state = initial_state,
                                                            r_bound = bound * bohr_radius, l_points = angular_points))

        cp.utils.multi_map(make_plots, specs, processes = 2)
