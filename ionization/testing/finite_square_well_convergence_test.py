import logging
import os

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


def run(spec):
    sim = spec.to_simulation()

    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.INFO,
                         file_logs = True, file_dir = OUT_DIR, file_name = sim.name, file_level = logging.INFO, file_mode = 'w') as logger:
        logger.info('Predicted initial energy: {} eV'.format(uround(sim.spec.initial_state.energy, eV, 10)))
        logger.info('Measured initial energy: {} eV'.format(uround(sim.spec.initial_state.energy, eV, 10)))

        logger.info(sim.info())
        sim.run_simulation()
        # sim.save(target_dir = OUT_DIR)
        logger.info(sim.info())

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR, x_scale = 'asec')

        cp.utils.xy_plot(sim.name + '__energy_vs_time',
                         sim.times, sim.energy_expectation_value_vs_time_internal,
                         x_label = '$t$', x_scale = 'asec', y_label = r'$E(t)$', y_scale = 'eV',
                         target_dir = OUT_DIR)

        cp.utils.xy_plot(sim.name + '__energy_vs_time__error',
                         sim.times, np.abs(sim.spec.initial_state.energy - sim.energy_expectation_value_vs_time_internal),
                         x_label = '$t$', x_scale = 'asec', y_label = r'$E(t) - E(t=0)$', y_scale = 'eV',
                         target_dir = OUT_DIR)

        cp.utils.xy_plot(sim.name + '__energy_vs_time__error_log',
                         sim.times, np.abs(sim.spec.initial_state.energy - sim.energy_expectation_value_vs_time_internal) / np.abs(sim.spec.initial_state.energy),
                         x_label = '$t$', x_scale = 'asec', y_label = r'$\frac{E(t) - E(t=0)}{E(t=0}}$', y_log_axis = True,
                         target_dir = OUT_DIR)

        return sim


if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.INFO) as logger:
        mass = electron_mass

        x_bounds = [500, 1000, 2000]
        x_points_exps = [16, 18, 20]
        time_bound = 10000
        dt = 1

        depth = 21.02
        width = 1.2632 * 2 * bohr_radius / nm

        # depth = 1.5
        # width = 1

        fsw_potential = ion.FiniteSquareWell(potential_depth = depth * eV, width = width * nm)
        init = ion.FiniteSquareWellState.from_square_well_potential(fsw_potential, mass, n = 1)

        test_states = ion.FiniteSquareWellState.all_states_of_well_from_well(fsw_potential, mass)

        specs = []

        for x_bound in x_bounds:
            for x_points_exp in x_points_exps:
                identifier = 'fsw__w={}nm_d={}eV__bound={}nm_points={}_time={}as_step={}as'.format(round(depth, 2), round(width, 2), x_bound, x_points_exp, time_bound, dt)

                specs.append(ion.LineSpecification(identifier,
                                                   x_bound = x_bound * nm, x_points = 2 ** x_points_exp,
                                                   internal_potential = fsw_potential,
                                                   test_mass = mass,
                                                   test_states = test_states,
                                                   dipole_gauges = (),
                                                   initial_state = init,
                                                   time_initial = 0, time_final = time_bound * asec, time_step = dt * asec,
                                                   ))

        sims = cp.utils.multi_map(run, specs, processes = 2)
