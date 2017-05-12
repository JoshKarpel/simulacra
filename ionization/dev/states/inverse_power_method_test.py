import logging
import os

import simulacra as si
import numpy as np

import scipy.sparse.linalg as sparsealg
from tqdm import tqdm
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        test_states = [ion.HydrogenBoundState(n, l) for n in range(6) for l in range(n)]
        spec_kwargs = {'r_bound': 100 * bohr_radius,
                       'r_points': 400,
                       'l_bound': 100,
                       'initial_state': ion.HydrogenBoundState(1, 0),
                       'time_initial': 0 * asec,
                       'time_final': 1000 * asec,
                       'time_step': 1 * asec,
                       'test_states': test_states,
                       # 'electric_potential': ion.Rectangle(50 * asec, 100 * asec, amplitude = 1 * atomic_electric_field)
                       }

        steps = [1, 10, 50, 100, 500, 1000, 2000]
        # steps = [1, 10, 50, 100, 250, 500, 1000, 1500, 2000]

        differences = []
        fractional_differences = []
        labels = []

        pre_post_differences = []
        pre_post_fractional_differences = []

        pre_post_norm_differences = []

        for step in steps:
            sim = ion.SphericalHarmonicSpecification('ipm_{}'.format(step), **spec_kwargs).to_simulation()

            g_analytic = sim.spec.initial_state.radial_function(sim.mesh.r) * sim.mesh.r / np.sqrt(sim.mesh.norm)  # analytic mesh reference

            h = sim.mesh._get_internal_hamiltonian_matrix_operator_single_l(l = 0)

            h.data[1] -= sim.spec.initial_state.energy
            h = h.tocsc()

            with si.utils.BlockTimer() as t:
                h_inv = sparsealg.inv(h)
            print(t)

            g_discrete = sim.mesh.g_mesh[0, :] / sim.mesh.norm

            for ii in tqdm(range(step)):
                # g_discrete = cy.tdma(h, g_discrete)

                g_discrete = h_inv.dot(g_discrete)

                g_discrete /= np.sqrt(np.sum(sim.mesh.inner_product_multiplier * np.abs(g_discrete) ** 2))

            # si.utils.xy_plot(sim.name + 'g_comparison',
            #                  sim.mesh.r,
            #                  np.abs(g_analytic) ** 2, np.abs(g_discrete) ** 2,
            #                  x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g \right|^2$',
            #                  target_dir = OUT_DIR)

            difference = np.abs(g_discrete - g_analytic)
            fractional_difference = np.abs((g_discrete - g_analytic) / g_analytic)

            differences.append(difference)
            fractional_differences.append(fractional_difference)
            labels.append(str(step))

            g_pre = g_discrete

            sim.mesh.g_mesh[0, :] = g_discrete
            norm_pre = sim.mesh.norm
            sim.run_simulation()

            g_post = sim.mesh.g_mesh[0, :]

            pre_post_difference = np.abs(g_pre) ** 2 - np.abs(g_post) ** 2
            pre_post_fractional_difference = (np.abs(g_pre) ** 2 - np.abs(g_post) ** 2) / np.abs(g_pre) ** 2

            pre_post_differences.append(pre_post_difference)
            pre_post_fractional_differences.append(pre_post_fractional_difference)

            pre_post_norm_difference = np.abs(norm_pre - sim.mesh.norm)
            pre_post_norm_differences.append(pre_post_norm_difference)

            # si.utils.xy_plot('g_difference_{}'.format(step),
            #                  sim.mesh.r, difference,
            #                  x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{discrete}} - g_{\mathrm{analytic}} \right|$',
            #                  y_log_axis = False, x_log_axis = False,
            #                  target_dir = OUT_DIR)
            #
            # si.utils.xy_plot('g_difference_log_lin_{}'.format(step),
            #                  sim.mesh.r, difference,
            #                  x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{discrete}} - g_{\mathrm{analytic}} \right|$',
            #                  y_log_axis = False, x_log_axis = True,
            #                  target_dir = OUT_DIR)
            #
            # si.utils.xy_plot('g_difference_log_{}'.format(step),
            #                  sim.mesh.r, difference,
            #                  x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{discrete}} - g_{\mathrm{analytic}} \right|$',
            #                  y_log_axis = True, x_log_axis = False,
            #                  target_dir = OUT_DIR)
            #
            # si.utils.xy_plot('g_fractional_difference_log_{}'.format(step),
            #                  sim.mesh.r, fractional_difference,
            #                  x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| \frac{g_{\mathrm{discrete}} - g_{\mathrm{analytic}}}{g_{\mathrm{analytic}}} \right|$',
            #                  y_log_axis = True, x_log_axis = False,
            #                  target_dir = OUT_DIR)
            #
            # si.utils.xy_plot('g_fractional_difference_log_log_{}'.format(step),
            #                  sim.mesh.r, fractional_difference,
            #                  x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| \frac{g_{\mathrm{discrete}} - g_{\mathrm{analytic}}}{g_{\mathrm{analytic}}} \right|$',
            #                  y_log_axis = True, x_log_axis = True,
            #                  target_dir = OUT_DIR)

        # COMPARE TO ANALYTIC STATE
        si.plots.xy_plot('g_difference_lin_log',
                         sim.mesh.r, *differences,
                         line_labels = labels,
                         x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{discrete}} - g_{\mathrm{analytic}} \right|$',
                         y_log_axis = False, x_log_axis = True,
                         target_dir = OUT_DIR)

        si.plots.xy_plot('g_difference_log',
                         sim.mesh.r, *differences,
                         line_labels = labels,
                         x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{discrete}} - g_{\mathrm{analytic}} \right|$',
                         y_log_axis = True, x_log_axis = False,
                         target_dir = OUT_DIR)

        si.plots.xy_plot('g_difference_log_log',
                         sim.mesh.r, *differences,
                         line_labels = labels,
                         x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{discrete}} - g_{\mathrm{analytic}} \right|$',
                         y_log_axis = True, x_log_axis = True,
                         target_dir = OUT_DIR)

        si.plots.xy_plot('g_fractional_difference_log',
                         sim.mesh.r, *fractional_differences,
                         line_labels = labels,
                         x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| \frac{g_{\mathrm{discrete}} - g_{\mathrm{analytic}}}{g_{\mathrm{analytic}}} \right|$',
                         y_log_axis = True, x_log_axis = False,
                         target_dir = OUT_DIR)

        si.plots.xy_plot('g_fractional_difference_log_log',
                         sim.mesh.r, *fractional_differences,
                         line_labels = labels,
                         x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| \frac{g_{\mathrm{discrete}} - g_{\mathrm{analytic}}}{g_{\mathrm{analytic}}} \right|$',
                         y_log_axis = True, x_log_axis = True,
                         target_dir = OUT_DIR)

        ## COMPARE TO POST-EVOLUTION STATE
        si.plots.xy_plot('evolved_g_difference_lin_log',
                         sim.mesh.r, *pre_post_differences,
                         line_labels = labels,
                         x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{pre}} \right|^2  - \left|g_{\mathrm{post}} \right|^2$',
                         y_log_axis = False, x_log_axis = True,
                         target_dir = OUT_DIR)

        si.plots.xy_plot('evolved_g_difference_log',
                         sim.mesh.r, *pre_post_differences,
                         line_labels = labels,
                         x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{pre}} \right|^2  - \left|g_{\mathrm{post}} \right|^2$',
                         y_log_axis = True, x_log_axis = False,
                         target_dir = OUT_DIR)

        si.plots.xy_plot('evolved_g_difference_log_log',
                         sim.mesh.r, *pre_post_differences,
                         line_labels = labels,
                         x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g_{\mathrm{pre}} \right|^2  - \left|g_{\mathrm{post}} \right|^2$',
                         y_log_axis = True, x_log_axis = True,
                         target_dir = OUT_DIR)

        si.plots.xy_plot('evolved_g_fractional_difference_log',
                         sim.mesh.r, *pre_post_fractional_differences,
                         line_labels = labels,
                         x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\frac{ \left| g_{\mathrm{pre}} \right|^2  - \left|g_{\mathrm{post}} \right|^2 }{\left| g_{\mathrm{pre}} \right|^2}$',
                         y_log_axis = True, x_log_axis = False,
                         target_dir = OUT_DIR)

        si.plots.xy_plot('evolved_g_fractional_difference_log_log',
                         sim.mesh.r, *pre_post_fractional_differences,
                         line_labels = labels,
                         x_unit = 'bohr_radius', x_label = r'$r$', y_label = r'$\frac{ \left| g_{\mathrm{pre}} \right|^2  - \left|g_{\mathrm{post}} \right|^2 }{\left| g_{\mathrm{pre}} \right|^2}$',
                         y_log_axis = True, x_log_axis = True,
                         target_dir = OUT_DIR)

        print(steps)
        print(pre_post_norm_differences)

        si.plots.xy_plot('norm_diff', steps, pre_post_norm_differences,
                         y_label = r'|initial norm - final norm|',
                         target_dir = OUT_DIR)

        si.plots.xy_plot('norm_diff_log_log', steps, pre_post_norm_differences,
                         y_label = r'|initial norm - final norm|',
                         y_log_axis = True, x_log_axis = True,
                         target_dir = OUT_DIR)
