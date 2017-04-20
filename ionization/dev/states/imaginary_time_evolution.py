import logging
import os

import numpy as np

import compy as cp
import ionization as ion
import plots
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        test_states = [ion.HydrogenBoundState(n, l) for n in range(6) for l in range(n)]
        spec_kwargs = {'r_bound': 100 * bohr_radius,
                       'r_points': 400,
                       'l_bound': 100,
                       'initial_state': ion.HydrogenBoundState(1, 0),
                       'time_initial': 0 * asec,
                       'time_final': 200 * asec,
                       'time_step': 1 * asec,
                       'test_states': test_states,
                       # 'electric_potential': ion.Rectangle(50 * asec, 100 * asec, amplitude = 1 * atomic_electric_field)
                       }

        differences = []
        fractional_differences = []
        labels = []

        pre_post_differences = []
        pre_post_fractional_differences = []

        im_time_steps = [1, 10, 50, 100, 250, 500, 1000, 1500, 2000]

        for im_time_step in im_time_steps:
            sim = ion.SphericalHarmonicSpecification('imag_{}'.format(im_time_step), **spec_kwargs, find_numerical_ground_state = True, imaginary_time_evolution_steps = im_time_step).to_simulation()

            g_discrete = sim.mesh.g_mesh[0, :] / sim.mesh.norm
            g_analytic = sim.spec.initial_state.radial_function(sim.mesh.r) * sim.mesh.r / sim.mesh.norm

            difference = np.abs(g_discrete - g_analytic)
            differences.append(difference)
            fractional_difference = np.abs((g_discrete - g_analytic) / g_analytic)
            fractional_differences.append(fractional_difference)
            labels.append(str(im_time_step))

            g_pre = g_discrete

            sim.mesh.g_mesh[0, :] = g_discrete
            sim.run_simulation()

            g_post = sim.mesh.g_mesh[0, :]

            pre_post_difference = np.abs(g_pre) ** 2 - np.abs(g_post) ** 2
            pre_post_fractional_difference = (np.abs(g_pre) ** 2 - np.abs(g_post) ** 2) / np.abs(g_pre) ** 2

            pre_post_differences.append(pre_post_difference)
            pre_post_fractional_differences.append(pre_post_fractional_difference)

            # cp.utils.xy_plot('g_fractional_difference_log_{}'.format(im_time_step),
            #                  imag.mesh.r, fractional_difference,
            #                  x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| \frac{g_{\mathrm{discrete}} - g_{\mathrm{analytic}}}{g_{\mathrm{analytic}}} \right|$',
            #                  y_log_axis = True, x_log_axis = False,
            #                  target_dir = OUT_DIR)
            #
            # cp.utils.xy_plot('g_fractional_difference_log_log_{}'.format(im_time_step),
            #                  imag.mesh.r, fractional_difference,
            #                  x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| \frac{g_{\mathrm{discrete}} - g_{\mathrm{analytic}}}{g_{\mathrm{analytic}}} \right|$',
            #                  y_log_axis = True, x_log_axis = True,
            #                  target_dir = OUT_DIR)

        plots.xy_plot('g_difference_lin_log',
                      sim.mesh.r, *differences,
                      line_labels = labels,
                      x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{discrete}} - g_{\mathrm{analytic}} \right|$',
                      y_log_axis = False, x_log_axis = True,
                      target_dir = OUT_DIR)

        plots.xy_plot('g_difference_log',
                      sim.mesh.r, *differences,
                      line_labels = labels,
                      x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{discrete}} - g_{\mathrm{analytic}} \right|$',
                      y_log_axis = True, x_log_axis = False,
                      target_dir = OUT_DIR)

        plots.xy_plot('g_difference_log_log',
                      sim.mesh.r, *differences,
                      line_labels = labels,
                      x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{discrete}} - g_{\mathrm{analytic}} \right|$',
                      y_log_axis = True, x_log_axis = True,
                      target_dir = OUT_DIR)

        plots.xy_plot('g_fractional_difference_log',
                      sim.mesh.r, *fractional_differences,
                      line_labels = labels,
                      x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| \frac{g_{\mathrm{discrete}} - g_{\mathrm{analytic}}}{g_{\mathrm{analytic}}} \right|$',
                      y_log_axis = True, x_log_axis = False,
                      target_dir = OUT_DIR)

        plots.xy_plot('g_fractional_difference_log_log',
                      sim.mesh.r, *fractional_differences,
                      line_labels = labels,
                      x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| \frac{g_{\mathrm{discrete}} - g_{\mathrm{analytic}}}{g_{\mathrm{analytic}}} \right|$',
                      y_log_axis = True, x_log_axis = True,
                      target_dir = OUT_DIR)

        ## COMPARE TO POST-EVOLUTION STATE
        plots.xy_plot('evolved_g_difference_lin_log',
                      sim.mesh.r, *pre_post_differences,
                      line_labels = labels,
                      x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{pre}} \right|^2  - \left|g_{\mathrm{post}} \right|^2$',
                      y_log_axis = False, x_log_axis = True,
                      target_dir = OUT_DIR)

        plots.xy_plot('evolved_g_difference_log',
                      sim.mesh.r, *pre_post_differences,
                      line_labels = labels,
                      x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{pre}} \right|^2  - \left|g_{\mathrm{post}} \right|^2$',
                      y_log_axis = True, x_log_axis = False,
                      target_dir = OUT_DIR)

        plots.xy_plot('evolved_g_difference_log_log',
                      sim.mesh.r, *pre_post_differences,
                      line_labels = labels,
                      x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{pre}} \right|^2  - \left|g_{\mathrm{post}} \right|^2$',
                      y_log_axis = True, x_log_axis = True,
                      target_dir = OUT_DIR)

        plots.xy_plot('evolved_g_fractional_difference_log',
                      sim.mesh.r, *pre_post_fractional_differences,
                      line_labels = labels,
                      x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\frac{ \left| g_{\mathrm{pre}} \right|^2  - \left|g_{\mathrm{post}} \right|^2 }{\left| g_{\mathrm{pre}} \right|^2}$',
                      y_log_axis = True, x_log_axis = False,
                      target_dir = OUT_DIR)

        plots.xy_plot('evolved_g_fractional_difference_log_log',
                      sim.mesh.r, *pre_post_fractional_differences,
                      line_labels = labels,
                      x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\frac{ \left| g_{\mathrm{pre}} \right|^2  - \left|g_{\mathrm{post}} \right|^2 }{\left| g_{\mathrm{pre}} \right|^2}$',
                      y_log_axis = True, x_log_axis = True,
                      target_dir = OUT_DIR)

        # with open(os.path.join(OUT_DIR, 'results.txt'), mode = 'w') as f:
        #     print('OVERLAPS BEFORE EVOLUTION', file = f)
        #     print('Norm = ', '  |  '.join('({}) {}'.format(sim.name, sim.mesh.norm) for sim in sims), file = f)
        #     for test_state in test_states:
        #         s = '{}g> = '.format(test_state.bra)
        #         s += '  |  '.join('({}) {}'.format(sim.name, sim.mesh.state_overlap(test_state)) for sim in sims)
        #         print(s, file = f)
        #
        #     for sim in sims:
        #         sim.run_simulation()
        #         sim.plot_wavefunction_vs_time(target_dir = OUT_DIR, log = True)
        #
        #     print('OVERLAPS AFTER EVOLUTION', file = f)
        #     print('Norm = ', '  |  '.join('({}) {}'.format(sim.name, sim.mesh.norm) for sim in sims), file = f)
        #     for test_state in test_states:
        #         s = '{}g> = '.format(test_state.bra)
        #         s += '  |  '.join('({}) {}'.format(sim.name, sim.mesh.state_overlap(test_state)) for sim in sims)
        #         print(s, file = f)
