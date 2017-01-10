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
        width = 3.1 * nm

        z_0 = width * np.sqrt(2 * mass * depth) / hbar
        print('z_0 = {},   floor(z_0 / pi) = {}'.format(z_0, np.floor(z_0 / pi)))

        pot = ion.FiniteSquareWell(potential_depth = depth, width = width)
        init = ion.FiniteSquareWellState(well_depth = depth, well_width = width, mass = mass, n = 1)

        electric = ion.SineWave.from_photon_energy(1 * eV, amplitude = .01 * atomic_electric_field)

        sim = ion.LineSpecification('fsw',
                                    x_bound = 10 * nm, x_points = 2 ** 14,
                                    internal_potential = pot,
                                    # electric_potential = electric,
                                    test_mass = mass,
                                    test_states = ion.FiniteSquareWellState.all_states_of_well(depth, width, mass),
                                    dipole_gauges = (),
                                    initial_state = init,
                                    time_initial = 0, time_final = 10 * fsec, time_step = 10 * asec,
                                    ).to_simulation()

        print(sim.info())

        cp.utils.xy_plot('fsw_potential', sim.mesh.x_mesh, pot(distance = sim.mesh.x_mesh),
                         x_scale = 'nm', y_scale = 'eV',
                         target_dir = OUT_DIR)

        sim.mesh.plot_g(name_postfix = '_init', target_dir = OUT_DIR)

        # # delta = 2 * bohr_radius
        # # sim.mesh.psi = np.exp(-0.5 * ((sim.mesh.x / delta) ** 2)).astype(np.complex128)
        #
        # print('init norm', sim.mesh.norm)
        #
        # sim.mesh.plot_g(name_postfix = '_init', target_dir = OUT_DIR)
        #
        # sim.run_simulation()
        #
        # print(sim.info())
        # print('norm', sim.mesh.norm)
        # print('energy EV', sim.energy_expectation_value_vs_time_internal / eV)
        # # print(pot.omega(init.mass), init.omega)
        # # print('period: {} fs'.format(uround(init.period, fsec, 3)))
        #
        # sim.mesh.plot_g(name_postfix = '_post', target_dir = OUT_DIR)
        # sim.plot_wavefunction_vs_time(target_dir = OUT_DIR, x_scale = 'fsec')
        #
        # # print(sim.mesh.wavenumbers)
        # # print(sim.mesh.free_evolution_prefactor)
