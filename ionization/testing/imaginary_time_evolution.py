import logging
import os

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        test_states = [ion.HydrogenBoundState(n, l) for n in range(6) for l in range(n)]
        spec_kwargs = {'r_bound': 100 * bohr_radius,
                       'r_points': 400,
                       'l_points': 100,
                       'initial_state': ion.HydrogenBoundState(1, 0),
                       'time_initial': 0 * asec,
                       'time_final': 200 * asec,
                       'time_step': 1 * asec,
                       'test_states': test_states,
                       'electric_potential': ion.Rectangle(50 * asec, 100 * asec, amplitude = 1 * atomic_electric_field)
                       }

        # im_time_steps = 1000

        # sims = [ion.SphericalHarmonicSpecification('no_imag', **spec_kwargs, find_numerical_ground_state = False).to_simulation(),
        #         ion.SphericalHarmonicSpecification('imag', **spec_kwargs, find_numerical_ground_state = True, imaginary_time_evolution_steps = im_time_steps).to_simulation()
        #         ]
        #
        # imag = sims[1]
        # g_discrete = imag.mesh.g_mesh[0, :] / imag.mesh.norm
        # g_analytic = imag.spec.initial_state.radial_function(imag.mesh.r) * imag.mesh.r
        # cp.utils.xy_plot('g_abs_comparison',
        #                  imag.mesh.r, np.abs(g_discrete) ** 2, np.abs(g_analytic) ** 2,
        #                  line_labels = (r'$g_{\mathrm{discrete}}', r'g_{\mathrm{analytic}}'),
        #                  x_scale = 'bohr_radius', x_label = r'$r$', y_label = r'$\left|g\right|^2$',
        #                  target_dir = OUT_DIR)
        #
        # cp.utils.xy_plot('g_abs_difference',
        #                  imag.mesh.r,
        #                  np.abs(g_discrete) ** 2 - np.abs(g_analytic) ** 2,
        #                  x_scale = 'bohr_radius', x_label = r'$r$', y_label = r'$\left|g_{\mathrm{discrete}}\right|^2 - \left|g_{\mathrm{analytic}}\right|^2$',
        #                  y_log_axis = False,
        #                  target_dir = OUT_DIR)
        #
        # cp.utils.xy_plot('g_difference_log',
        #                  imag.mesh.r, np.abs(g_discrete - g_analytic),
        #                  x_scale = 'bohr_radius', x_label = r'$r$', y_label = r'$\left|g_{\mathrm{discrete}} - g_{\mathrm{analytic}}\right|^2$',
        #                  y_log_axis = True, y_lower_limit = 1e-8,
        #                  target_dir = OUT_DIR)

        differences = []
        fractional_differences = []
        labels = []

        for im_time_step in [1, 10, 50, 100, 250, 500, 1000, 1500, 2000]:
            imag = ion.SphericalHarmonicSpecification('imag_{}'.format(im_time_step), **spec_kwargs, find_numerical_ground_state = True, imaginary_time_evolution_steps = im_time_step).to_simulation()

            g_discrete = imag.mesh.g_mesh[0, :] / imag.mesh.norm
            g_analytic = imag.spec.initial_state.radial_function(imag.mesh.r) * imag.mesh.r

            difference = np.abs(g_discrete - g_analytic)
            differences.append(difference)
            fractional_difference = np.abs((g_discrete - g_analytic) / g_analytic)
            fractional_differences.append(fractional_difference)
            labels.append(str(im_time_step))

            # cp.utils.xy_plot('g_fractional_difference_log_{}'.format(im_time_step),
            #                  imag.mesh.r, fractional_difference,
            #                  x_scale = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| \frac{g_{\mathrm{discrete}} - g_{\mathrm{analytic}}}{g_{\mathrm{analytic}}} \right|$',
            #                  y_log_axis = True, x_log_axis = False,
            #                  target_dir = OUT_DIR)
            #
            # cp.utils.xy_plot('g_fractional_difference_log_log_{}'.format(im_time_step),
            #                  imag.mesh.r, fractional_difference,
            #                  x_scale = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| \frac{g_{\mathrm{discrete}} - g_{\mathrm{analytic}}}{g_{\mathrm{analytic}}} \right|$',
            #                  y_log_axis = True, x_log_axis = True,
            #                  target_dir = OUT_DIR)

        cp.utils.xy_plot('g_difference_lin_log',
                         imag.mesh.r, *differences,
                         line_labels = labels,
                         x_scale = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{discrete}} - g_{\mathrm{analytic}} \right|$',
                         y_log_axis = False, x_log_axis = True,
                         target_dir = OUT_DIR)

        cp.utils.xy_plot('g_difference_log',
                         imag.mesh.r, *differences,
                         line_labels = labels,
                         x_scale = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{discrete}} - g_{\mathrm{analytic}} \right|$',
                         y_log_axis = True, x_log_axis = False,
                         target_dir = OUT_DIR)

        cp.utils.xy_plot('g_difference_log_log',
                         imag.mesh.r, *differences,
                         line_labels = labels,
                         x_scale = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{discrete}} - g_{\mathrm{analytic}} \right|$',
                         y_log_axis = True, x_log_axis = True,
                         target_dir = OUT_DIR)

        cp.utils.xy_plot('g_fractional_difference_log',
                         imag.mesh.r, *fractional_differences,
                         line_labels = labels,
                         x_scale = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| \frac{g_{\mathrm{discrete}} - g_{\mathrm{analytic}}}{g_{\mathrm{analytic}}} \right|$',
                         y_log_axis = True, x_log_axis = False,
                         target_dir = OUT_DIR)

        cp.utils.xy_plot('g_fractional_difference_log_log',
                         imag.mesh.r, *fractional_differences,
                         line_labels = labels,
                         x_scale = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| \frac{g_{\mathrm{discrete}} - g_{\mathrm{analytic}}}{g_{\mathrm{analytic}}} \right|$',
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
