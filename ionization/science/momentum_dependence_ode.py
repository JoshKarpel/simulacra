import os
import logging
import collections as co
from copy import deepcopy

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)

if __name__ == '__main__':
    with log as logger:
        a_0 = 1

        pw = 200
        bound = 20

        times = np.linspace(-bound * pw * asec, bound * pw * asec, 1e5)
        dt = np.abs(times[1] - times[0])

        ###########
        # ELECTRIC FIELD
        ###########

        electric_field = ion.SincPulse(pw * asec, .1 * Jcm2, phase = 'cos', dc_correction_time = bound * pw * asec)
        electric_field_amplitude_vs_time = electric_field.get_electric_field_amplitude(times)
        # print('e field', electric_field_amplitude_vs_time)
        # integral_of_electric_field_amplitude_vs_time = electric_field.get_total_electric_field_numeric(times)

        cp.utils.xy_plot('electric_field_vs_time',
                         times, electric_field_amplitude_vs_time,
                         x_scale = 'asec', y_scale = 'AEF',
                         target_dir = OUT_DIR)

        ############
        # TRYING TO DETERMINE A AND DELTA
        ############

        # assume \ket{k} is a free space wavefunction

        depth = 1 * eV
        width = 1 * nm
        finite_square_well = ion.FiniteSquareWell(potential_depth = depth, width = width)
        square_well_states = list(ion.FiniteSquareWellState.all_states_of_well_from_parameters(depth, width, electron_mass))

        animators = [
            ion.animators.LineAnimator(postfix = '', target_dir = OUT_DIR, plot_limit = 40 * nm, length = 60, renormalize = False, metrics = ('norm', 'initial_state_overlap')),
            ion.animators.LineAnimator(postfix = '_renormalized', target_dir = OUT_DIR, plot_limit = 40 * nm, length = 60, renormalize = True, metrics = ('norm', 'initial_state_overlap'))
        ]
        sim = ion.LineSpecification('sim',
                                    time_initial = -bound * pw * asec, time_final = (bound * pw + 20000) * asec, time_step = 5 * asec,
                                    x_bound = 50 * nm, x_points = 2 ** 15,
                                    internal_potential = finite_square_well,
                                    electric_potential = electric_field,
                                    test_states = square_well_states,
                                    dipole_gauges = (),
                                    initial_state = square_well_states[0],
                                    mask = ion.RadialCosineMask(inner_radius = 40 * nm, outer_radius = 50 * nm),
                                    animators = animators
                                    ).to_simulation()

        wavenumbers = (twopi / nm) * np.linspace(-10, 10, 1000)
        plane_waves = [ion.OneDFreeParticle(k, mass = electron_mass) for k in wavenumbers]
        dk = np.abs(plane_waves[1].wavenumber - plane_waves[0].wavenumber)

        matrix_element_by_k = co.OrderedDict((k, dk * (np.abs(sim.mesh.inner_product(sim.mesh.get_g_for_state(k), sim.mesh.x_mesh * sim.mesh.g_mesh)) ** 2)) for k in plane_waves)
        matrix_element_vs_k = np.array(list(matrix_element_by_k.values()))

        cp.utils.xy_plot('matrix_element_vs_k',
                         wavenumbers, matrix_element_vs_k,
                         x_scale = twopi / nm, x_label = r'Wavenumber $k$ ($2\pi/\mathrm{nm}$)',
                         y_label = r'$\left| \left\langle \alpha \right| \widehat{x} \left| k \right\rangle \right|^2 \, dk$',
                         # y_lower_limit = 0, y_upper_limit = 1,
                         target_dir = OUT_DIR)

        matrix_element_by_wk = co.OrderedDict((k.energy / hbar, (np.abs(sim.mesh.inner_product(sim.mesh.get_g_for_state(k), sim.mesh.x_mesh * sim.mesh.g_mesh)) ** 2) / np.sqrt(k.energy / hbar)) for k in plane_waves)
        wk = np.array(list(matrix_element_by_wk.keys()))
        matrix_element_vs_wk = np.array(list(matrix_element_by_wk.values()))

        integral = np.sum(matrix_element_vs_wk[:-1] * np.abs(np.diff(wk)))
        print(integral / (twopi * 250 * THz))

        cp.utils.xy_plot('matrix_element_vs_wk',
                         wk, matrix_element_vs_wk,
                         x_scale = twopi * THz, x_label = r'Angular Frequency $\omega_k$ ($2\pi \, \times \, \mathrm{THz}$)',
                         y_label = r'$\frac{\left| \left\langle \alpha \right| \widehat{x} \left| k \right\rangle \right|^2}{\sqrt{\omega_k}}$',
                         x_lower_limit = -twopi * 100 * THz, x_upper_limit = twopi * 1000 * THz,
                         # y_lower_limit = 0, y_upper_limit = 1,
                         target_dir = OUT_DIR)

        delta = twopi * 250 * THz
        A = integral / delta
        prefactor = -((electron_charge / hbar) ** 2) * np.sqrt(electron_mass / 2) * A * delta

        print(sim.info())
        sim.run_simulation()
        print(sim.info())
        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)

        ############
        # RK4 TIME EVOLUTION
        ############

        a = np.zeros(np.shape(times), dtype = np.complex128) * np.NaN
        a[0] = a_0

        for time_index, current_time in enumerate(times[:-1]):
            print(time_index, current_time / asec, a[time_index], np.abs(a[time_index]) ** 2)

            current_electric_field = electric_field_amplitude_vs_time[time_index]
            midpoint_electric_field = (electric_field_amplitude_vs_time[time_index] + electric_field_amplitude_vs_time[time_index + 1]) / 2
            next_electric_field = electric_field_amplitude_vs_time[time_index + 1]

            time_difference_term = delta * (current_time - times[:time_index]) / 2
            integral_through_previous_step = electric_field_amplitude_vs_time[:time_index] * a[:time_index] * np.exp(-1j * time_difference_term) * cp.math.sinc(time_difference_term)
            integral_through_previous_step = prefactor * electric_field_amplitude_vs_time[time_index] * np.sum(integral_through_previous_step) * dt

            current_step = prefactor * electric_field_amplitude_vs_time[time_index] * electric_field_amplitude_vs_time[time_index] * a[time_index] * dt

            k_1 = integral_through_previous_step + current_step

            time_difference_term_from_midpoint = delta * ((current_time + dt / 2) - times[:time_index + 1]) / 2
            integral_through_previous_step_from_midpoint = electric_field_amplitude_vs_time[:time_index + 1] * a[:time_index + 1] * np.exp(-1j * time_difference_term_from_midpoint) * cp.math.sinc(time_difference_term_from_midpoint)
            integral_through_previous_step_from_midpoint = prefactor * midpoint_electric_field * np.sum(integral_through_previous_step_from_midpoint) * dt

            midpoint_contribution = prefactor * (midpoint_electric_field ** 2) * (a[time_index] + (dt * k_1 / 2))

            k_2 = integral_through_previous_step_from_midpoint + midpoint_contribution

            midpoint_contribution = prefactor * (midpoint_electric_field ** 2) * (a[time_index] + (dt * k_2 / 2))

            k_3 = integral_through_previous_step_from_midpoint + midpoint_contribution

            time_difference_term_from_next = delta * (times[time_index + 1] - times[:time_index + 1]) / 2
            integral_through_previous_step_from_next = electric_field_amplitude_vs_time[:time_index + 1] * a[:time_index + 1] * np.exp(-1j * time_difference_term_from_next) * cp.math.sinc(time_difference_term_from_next)
            integral_through_previous_step_from_next = prefactor * next_electric_field * np.sum(integral_through_previous_step_from_next) * dt

            next_contribution = prefactor * (next_electric_field ** 2) * (a[time_index] + (dt * k_3))

            k_4 = integral_through_previous_step_from_next + next_contribution

            a[time_index + 1] = a[time_index] + (dt * (k_1 + (2 * k_2) * (2 * k_3) + k_4) / 6)

            # print()

        cp.utils.xy_plot('a_vs_t', times, np.abs(a) ** 2,
                         x_scale = 'asec', x_label = r'$t$',
                         target_dir = OUT_DIR)
