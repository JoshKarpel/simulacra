import logging
import os

import numpy as np
import scipy.sparse as sparse

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        mass = electron_mass

        depth = 1.5 * eV
        width = 3 * nm

        z_0 = width * np.sqrt(2 * mass * depth) / hbar
        print('z_0 = {},   floor(z_0 / pi) = {}'.format(z_0, np.floor(z_0 / pi)))

        pot = ion.FiniteSquareWell(potential_depth = depth, width = width)
        init = ion.FiniteSquareWellState(well_depth = depth, well_width = width, mass = mass, n = 1)

        electric = ion.SineWave.from_photon_energy(1 * eV, amplitude = .01 * atomic_electric_field,
                                                   window = ion.SymmetricExponentialTimeWindow(window_time = 5 * fsec, window_width = 250 * asec, window_center = 10 * fsec))

        ani = [ion.animators.LineAnimator(postfix = '_full_2', target_dir = OUT_DIR, length = 60, renormalize = True)]

        sim = ion.LineSpecification('fsw',
                                    x_bound = 50 * nm, x_points = 2 ** 15,
                                    internal_potential = pot,
                                    electric_potential = electric,
                                    test_mass = mass,
                                    test_states = ion.FiniteSquareWellState.all_states_of_well(depth, width, mass),
                                    dipole_gauges = (),
                                    initial_state = init,
                                    time_initial = 0 * fsec, time_final = 50 * fsec, time_step = 5 * asec,
                                    mask = ion.RadialCosineMask(inner_radius = 40 * nm, outer_radius = 50 * nm),
                                    animators = ani
                                    ).to_simulation()

        print(sim.info())

        cp.utils.xy_plot('fsw_potential', sim.mesh.x_mesh, pot(distance = sim.mesh.x_mesh),
                         x_scale = 'nm', y_scale = 'eV',
                         target_dir = OUT_DIR)

        print('init norm', sim.mesh.norm)

        sim.mesh.plot_g(name_postfix = '_init', target_dir = OUT_DIR)

        sim.run_simulation()

        print(sim.info())
        print('norm', sim.mesh.norm)
        print('energy EV', sim.energy_expectation_value_vs_time_internal / eV)

        sim.mesh.plot_g(name_postfix = '_post', target_dir = OUT_DIR)
        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR, x_scale = 'fsec')
