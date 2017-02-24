import logging
import os

import numpy as np
import scipy.sparse.linalg as sparsealg

import compy as cp
import compy.cy as cy
import ionization as ion
from ionization import integrodiff as ide
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


def run(spec):
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        sim = spec.to_simulation()

        logger.debug(sim.info())
        sim.run_simulation()
        logger.debug(sim.info())

        sim.plot_solution(target_dir = OUT_DIR,
                          y_axis_label = r'$   \left| a_{\alpha}(t) \right|^2  $',
                          f_axis_label = r'${}(t)$'.format(str_efield),
                          f_scale = 'AEF')

        return sim


if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        pw = 50
        flu = 20

        electric_field = ion.SincPulse(pulse_width = pw * asec, fluence = flu * Jcm2)

        q = electron_charge
        m = electron_mass_reduced
        L = bohr_radius

        tau_alpha = 4 * m * (L ** 2) / hbar
        prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

        dt = 1
        t_bound = 1000

        int_methods = ['trapezoid', 'simpson']
        evol_methods = ['FE', 'BE', 'RK4', 'TRAP']

        specs = []

        for evol_method in evol_methods:
            for int_method in int_methods:
                specs.append(ide.BoundStateIntegroDifferentialEquationSpecification('int={}__evol={}'.format(int_method, evol_method),
                                                                                    time_initial = -t_bound * asec, time_final = t_bound * asec, time_step = dt * asec,
                                                                                    prefactor = prefactor,
                                                                                    f = electric_field.get_electric_field_amplitude,
                                                                                    kernel = ide.gaussian_kernel, kernel_kwargs = dict(tau_alpha = tau_alpha),
                                                                                    evolution_method = evol_method,
                                                                                    integration_method = int_method,
                                                                                    ))

        results = cp.utils.multi_map(run, specs, processes = 2)

        t = results[0].times
        y = [np.abs(r.y) ** 2 for r in results]
        labels = list(r.name for r in results)

        plt_kwargs = dict(
            line_labels = labels,
            target_dir = OUT_DIR,
        )

        cp.utils.xy_plot('compare',
                         t,
                         *y,
                         x_label = r'Time $t$', x_scale = 'asec', y_label = r'$   \left| a_{\alpha}(t) \right|^2  $',
                         **plt_kwargs
                         )

        cp.utils.xy_plot('compare_log',
                         t,
                         *y,
                         x_label = r'Time $t$', x_scale = 'asec', y_label = r'$   \left| a_{\alpha}(t) \right|^2  $',
                         y_log_axis = True,
                         **plt_kwargs
                         )
