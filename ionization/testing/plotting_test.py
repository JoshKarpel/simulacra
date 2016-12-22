import os

import compy as cp
from compy.units import *
import ionization as ion

import matplotlib as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

CMAP = plt.cm.inferno


def make_plots(spec):
    with cp.utils.Timer() as t:
        sim = ion.ElectricFieldSimulation(spec)
        sim.mesh.plot_g(target_dir = OUT_DIR, colormap = CMAP)
        # sim.mesh.plot_psi(target_dir = OUT_DIR, colormap = CMAP)
    print(sim.name, t)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization') as logger:
        n = 4
        bound = 30
        angular_points = 5

        states = (ion.HydrogenBoundState(n, l) for n in range(n + 1) for l in range(n))

        specs = []

        for initial_state in states:
            # spec = ion.CylindricalSliceSpecification('cyl_slice__{}_{}'.format(initial_state.n, initial_state.l),
            #                                          initial_state = initial_state,
            #                                          z_bound = bound * bohr_radius, rho_bound = bound * bohr_radius)
            # specs.append(spec)
            #
            # spec = ion.SphericalSliceSpecification('sph_slice__{}_{}'.format(initial_state.n, initial_state.l),
            #                                        initial_state = initial_state,
            #                                        r_bound = bound * bohr_radius, theta_points = angular_points)
            # specs.append(spec)

            spec = ion.SphericalHarmonicSpecification('sph_harms__{}_{}'.format(initial_state.n, initial_state.l),
                                                      initial_state = initial_state,
                                                      r_bound = bound * bohr_radius, spherical_harmonics_max_l = angular_points)
            specs.append(spec)

        cp.utils.multi_map(make_plots, specs, processes = 3)
