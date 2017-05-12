import itertools as it
import logging
import os

import simulacra as si
import numpy as np

from simulacra.units import *

import ionization as ion
from ionization import integrodiff as ide


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = si.utils.LogManager('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO)


def run(spec):
    with log as logger:
        sim = spec.to_simulation()

        sim.run_simulation()

        return sim


if __name__ == '__main__':
    with log as logger:
        pw = 100 * asec
        t_bound = 10

        L = bohr_radius
        m = electron_mass
        q = electron_charge

        sinc = ion.SincPulse(pulse_width = pw)
        efield = ion.GaussianPulse(pulse_width = pw, fluence = 20 * Jcm2, omega_carrier = sinc.omega_carrier)
        # efield = ion.SincPulse(pulse_width = pw, fluence = 1 * Jcm2)

        # efield = ion.Rectangle(start_time = -100 * asec, end_time = 100 * asec, amplitude = .01 * atomic_electric_field)

        prefactor = - ((q / m) ** 2) / (4 * (L ** 2))
        tau_alpha = 2 * m * (L ** 2) / hbar

        specs = []
        for method in ('FE', 'BE', 'TRAP', 'RK4'):
            specs.append(ide.VelocityGaugeIntegroDifferentialEquationSpecification(method,
                                                                                   time_initial = - t_bound * pw, time_final = t_bound * pw, time_step = 1 * asec,
                                                                                   electric_potential = efield,
                                                                                   prefactor = prefactor,
                                                                                   kernel = ide.velocity_guassian_kernel, kernel_kwargs = dict(tau_alpha = tau_alpha, width = L),
                                                                                   evolution_method = method,
                                                                                   ))

        plt_kwargs = dict(
            target_dir = OUT_DIR,
        )

        results = si.utils.multi_map(run, specs, processes = 4)

        for r in results:
            print(r.info())
            r.plot_fields_vs_time(**plt_kwargs)
            r.plot_a_vs_time(**plt_kwargs)

        for log, rel in it.product((True, False), repeat = 2):
            plot_name = 'comparison'
            if log:
                plot_name += '__log'

            if rel:
                plot_name += '__rel'
                y = [(np.abs(r.a) ** 2) / (np.abs(results[-1].a) ** 2) for r in results]
                y_lab = r'$ \left| a(t) \right|^2 / \left| a_{\mathrm{RK4}}(t) \right|^2$'
            else:
                y = [np.abs(r.a) ** 2 for r in results]
                y_lab = r'$ \left| a(t) \right|^2 $'

            si.plots.xy_plot(plot_name,
                             results[0].times,
                             *y,
                             line_labels = [r.name for r in results],
                             x_label = r'Time $t$', x_unit = 'asec',
                             y_label = y_lab,
                             y_log_axis = log,
                             **plt_kwargs)
