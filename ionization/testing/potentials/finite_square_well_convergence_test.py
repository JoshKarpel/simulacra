import logging
import os
import itertools as it

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


def run(spec):
    sim = spec.to_simulation()

    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG,
                         file_logs = True, file_dir = OUT_DIR, file_name = sim.name, file_level = logging.INFO, file_mode = 'w') as logger:
        logger.info('Predicted initial energy: {} eV'.format(uround(sim.spec.initial_state.energy, eV, 10)))
        logger.info('Measured initial energy: {} eV'.format(uround(sim.spec.initial_state.energy, eV, 10)))

        cp.utils.xy_plot(sim.name + '__fft_pre_k',
                         sim.mesh.wavenumbers / twopi, np.abs(sim.mesh.fft(sim.mesh.g_mesh)) ** 2,
                         x_scale = 'per_nm', x_label = r'Wavenumber $k = \frac{1}{\lambda}$',
                         target_dir = OUT_DIR)

        cp.utils.xy_plot(sim.name + '__fft_pre_E',
                         sim.mesh.energies, np.abs(sim.mesh.fft(sim.mesh.g_mesh)) ** 2,
                         x_scale = 'eV', x_label = r'Energy $E = \frac{\hbar^2 k^2}{2\mu}$',
                         target_dir = OUT_DIR)

        logger.info(sim.info())
        sim.run_simulation()
        # sim.save(target_dir = OUT_DIR)
        logger.info(sim.info())

        cp.utils.xy_plot(sim.name + '__fft_post_k',
                         sim.mesh.wavenumbers / twopi, np.abs(sim.mesh.fft(sim.mesh.g_mesh)) ** 2,
                         x_scale = 'per_nm', x_label = r'Wavenumber $k = \frac{1}{\lambda}$',
                         target_dir = OUT_DIR)

        cp.utils.xy_plot(sim.name + '__fft_post_E',
                         sim.mesh.energies, np.abs(sim.mesh.fft(sim.mesh.g_mesh)) ** 2,
                         x_scale = 'eV', x_label = r'Energy $E = \frac{\hbar^2 k^2}{2\mu}$',
                         target_dir = OUT_DIR)

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR, x_scale = 'asec')

        cp.utils.xy_plot(sim.name + '__energy_vs_time',
                         sim.times, sim.energy_expectation_value_vs_time_internal,
                         x_label = '$t$', x_scale = 'asec', y_label = r'$E(t)$', y_scale = 'eV',
                         target_dir = OUT_DIR)

        cp.utils.xy_plot(sim.name + '__energy_vs_time__error',
                         sim.times, np.abs(sim.spec.initial_state.energy - sim.energy_expectation_value_vs_time_internal),
                         x_label = '$t$', x_scale = 'asec', y_label = r'$E(t) - E_0$', y_scale = 'eV',
                         target_dir = OUT_DIR)

        cp.utils.xy_plot(sim.name + '__energy_vs_time__ratio',
                         sim.times, np.abs(sim.spec.initial_state.energy / sim.energy_expectation_value_vs_time_internal),
                         x_label = '$t$', x_scale = 'asec', y_label = r'$\left| \frac{E(t)}{E_0} \right|$',
                         target_dir = OUT_DIR)

        cp.utils.xy_plot(sim.name + '__energy_vs_time__ratio_log',
                         sim.times, np.abs(sim.spec.initial_state.energy / sim.energy_expectation_value_vs_time_internal),
                         x_label = '$t$', x_scale = 'asec', y_label = r'$\left| \frac{E(t)}{E_0} \right|$', y_log_axis = True,
                         target_dir = OUT_DIR)

        return sim


if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        prefix = cp.utils.now_string()

        mass = electron_mass

        x_bounds = [2000]
        x_points_exps = [15, 16, 17, 18, 19, 20]
        time_bound = 500
        dt = 1

        depth = 21.02
        width = 1.2632 * 2 * bohr_radius / nm

        # depth = 1.5
        # width = 1

        fsw_potential = ion.FiniteSquareWell(potential_depth = depth * eV, width = width * nm)
        init = ion.FiniteSquareWellState.from_potential(fsw_potential, mass, n = 1)

        test_states = ion.FiniteSquareWellState.all_states_of_well_from_well(fsw_potential, mass)

        # electric_field = ion.Rectangle(start_time = 100 * asec, end_time = 150 * asec, amplitude = 1 * atomic_electric_field)

        specs = []

        for x_bound, x_points_exp in it.product(x_bounds, x_points_exps):
            identifier = prefix + '__fsw__w={}nm_d={}eV__bound={}nm_points={}_time={}as_step={}as'.format(round(depth, 2), round(width, 2), x_bound, x_points_exp, time_bound, dt)

            specs.append(ion.LineSpecification(identifier,
                                               x_bound = x_bound * nm, x_points = 2 ** x_points_exp,
                                               internal_potential = fsw_potential,
                                               test_mass = mass,
                                               test_states = test_states,
                                               dipole_gauges = (),
                                               initial_state = init,
                                               time_initial = 0, time_final = time_bound * asec, time_step = dt * asec,
                                               fft_cutoff_energy = 500 * eV
                                               ))

        sims = cp.utils.multi_map(run, specs, processes = 2)

        times = sims[0].times
        energy_expectation_value = [sim.energy_expectation_value_vs_time_internal for sim in sims]
        energy_expectation_value_error = [np.abs(sim.energy_expectation_value_vs_time_internal - sim.spec.initial_state.energy) for sim in sims]
        energy_expectation_value_ratio = [np.abs(sim.energy_expectation_value_vs_time_internal / sim.spec.initial_state.energy) for sim in sims]
        labels = ['bound = {} nm, points = $2^{{{}}}$'.format(uround(sim.spec.x_bound, 'nm'), int(np.log2(sim.spec.x_points))) for sim in sims]

        cp.utils.xy_plot(prefix + '__combined__energy_vs_time',
                         times, *energy_expectation_value,
                         line_labels = labels,
                         x_label = '$t$', x_scale = 'asec', y_label = r'$E(t)$', y_scale = 'eV',
                         target_dir = OUT_DIR)

        cp.utils.xy_plot(prefix + '__combined__energy_vs_time__error',
                         times, *energy_expectation_value_error,
                         line_labels = labels,
                         x_label = '$t$', x_scale = 'asec', y_label = r'$E(t) - E_0$', y_scale = 'eV',
                         target_dir = OUT_DIR)

        cp.utils.xy_plot(prefix + '__combined__energy_vs_time__ratio',
                         times, *energy_expectation_value_ratio,
                         line_labels = labels,
                         x_label = '$t$', x_scale = 'asec', y_label = r'$\left| \frac{E(t)}{E_0} \right|$',
                         target_dir = OUT_DIR)

        cp.utils.xy_plot(prefix + '__combined__energy_vs_time__ratio_log',
                         times, *energy_expectation_value_ratio,
                         line_labels = labels,
                         x_label = '$t$', x_scale = 'asec', y_label = r'$\left| \frac{E(t)}{E_0} \right|$', y_log_axis = True,
                         target_dir = OUT_DIR)
