import logging
import os

import numpy as np
import scipy.sparse.linalg as sparsealg

import compy as cp
import compy.cy as cy
import ionization as ion
import plots
from ionization import integrodiff as ide
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


def run(spec):
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        sim = spec.to_simulation()

        logger.debug(sim.info())
        sim.run_simulation()
        logger.debug(sim.info())

        # sim.plot_solution(target_dir = OUT_DIR,
        #                   y_axis_label = r'$   \left| a_{\alpha}(t) \right|^2  $',
        #                   f_axis_label = r'${}(t)$'.format(str_efield),
        #                   f_scale = 'AEF')

        return sim


if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        pw = 105
        amp = 5
        t_bound = pw + 10
        electric_field = ion.Rectangle(start_time = -pw * asec, end_time = pw * asec, amplitude = 1 * atomic_electric_field)
        OUT_DIR = os.path.join(OUT_DIR, '{}_pw={}as_amp={}aef'.format(electric_field.__class__.__name__, pw, amp))

        # pw = 50
        # flu = 1
        # t_bound = pw * 30
        # phase = pi / 2
        # electric_field = ion.SincPulse(pulse_width = pw * asec, fluence = flu * Jcm2, phase = phase)
        # OUT_DIR = os.path.join(OUT_DIR, '{}_pw={}as_flu={}jcm2_phase={}__bound={}'.format(electric_field.__class__.__name__, pw, flu, round(phase, 3), round(t_bound / pw)))

        q = electron_charge
        m = electron_mass_reduced
        L = bohr_radius

        tau_alpha = 4 * m * (L ** 2) / hbar
        prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

        # dt = 2

        int_methods = ['simpson']
        # int_methods = ['trapezoid', 'simpson']
        evol_methods = ['FE', 'BE', 'RK4', 'TRAP']

        ark4 = ide.AdaptiveIntegroDifferentialEquationSpecification('int={}__evol={}'.format('simpson', 'ARK4'),
                                                                    time_initial = -t_bound * asec, time_final = t_bound * asec, time_step = 1 * asec,
                                                                    prefactor = prefactor,
                                                                    f = electric_field.get_electric_field_amplitude,
                                                                    kernel = ide.gaussian_kernel, kernel_kwargs = dict(tau_alpha = tau_alpha),
                                                                    evolution_method = 'ARK4',
                                                                    integration_method = 'simpson',
                                                                    ).to_simulation()
        ark4.run_simulation()

        plots.xy_plot('time_step',
                      ark4.times,
                      ark4.time_steps_list,
                      x_axis_label = r'Time $t$', x_scale = 'asec',
                      y_axis_label = r'Time Step $\Delta t$', y_scale = 'asec',
                      y_log_axis = True,
                      target_dir = OUT_DIR,
                      )

        for dt in [5, 2, 1, .5, .1, .05]:
            specs = []

            for evol_method in evol_methods:
                for int_method in int_methods:
                    specs.append(ide.IntegroDifferentialEquationSpecification('int={}__evol={}'.format(int_method, evol_method),
                                                                              time_initial = -t_bound * asec, time_final = t_bound * asec, time_step = dt * asec,
                                                                              prefactor = prefactor,
                                                                              electric_potential = electric_field.get_electric_field_amplitude,
                                                                              kernel = ide.gaussian_kernel, kernel_kwargs = dict(tau_alpha = tau_alpha),
                                                                              evolution_method = evol_method,
                                                                              integration_method = int_method,
                                                                              ))

            results = cp.utils.multi_map(run, specs, processes = 5)

            t = results[0].times
            # y = [np.abs(r.y) ** 2 for r in results]

            y = [cp.utils.downsample(ark4.times, t, np.abs(ark4.y) ** 2)]
            for r in results:
                y.append(np.abs(r.y) ** 2)

            labels = [ark4.name] + list(r.name for r in results)

            plt_kwargs = dict(
                line_labels = labels,
                target_dir = OUT_DIR,
            )

            plots.xy_plot('dt={}as__compare'.format(dt),
                          t,
                          *y,
                          x_label = r'Time $t$', x_scale = 'asec', y_label = r'$   \left| a_{\alpha}(t) \right|^2  $',
                          **plt_kwargs, y_upper_limit = 1, y_lower_limit = 0,
                          )

            plots.xy_plot('dt={}as__compare_log'.format(dt),
                          t,
                          *y,
                          x_label = r'Time $t$', x_scale = 'asec', y_label = r'$   \left| a_{\alpha}(t) \right|^2  $',
                          y_log_axis = True, y_upper_limit = 1, y_lower_limit = 1e-2,
                          **plt_kwargs
                          )

        print('ark4 min step (as):', uround(ark4.minimum_time_step, 'asec', 5))
