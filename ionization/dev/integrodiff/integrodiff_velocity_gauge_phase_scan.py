import logging
import os

from tqdm import tqdm

import numpy as np
import scipy.integrate as integrate

import compy as cp
import ionization as ion
from ionization import integrodiff as ide
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG)


def run(spec):
    with log as logger:
        sim = spec.to_simulation()
        sim.run_simulation()
        logger.info(sim.info())

        # sim.plot_a_vs_time(target_dir = OUT_DIR)

    return sim


if __name__ == '__main__':
    with log as logger:
        # pulse_widths = np.linspace(50, 600, 50) * asec
        pw = 400 * asec
        phases = np.linspace(0, twopi, 200)
        t_bound = 5

        flu = 10 * Jcm2

        L = bohr_radius
        m = electron_mass
        q = electron_charge

        prefactor = -((q / m) ** 2) / (4 * (L ** 2))
        tau_alpha = 2 * m * (L ** 2) / hbar

        specs = []
        for phase in phases:
            reference_sinc = ion.SincPulse(pulse_width = pw)
            efield = ion.GaussianPulse(pulse_width = pw, fluence = flu, phase = phase, omega_carrier = reference_sinc.omega_carrier,
                                       window = ion.SymmetricExponentialTimeWindow(window_time = (t_bound * .9) * pw, window_width = .2 * pw))

            specs.append(ide.VelocityGaugeIntegroDifferentialEquationSpecification(f'{efield.__class__.__name__}_pw={uround(pw, asec, 3)}as_flu={uround(flu, Jcm2, 3)}Jcm2_phase={uround(phase)}',
                                                                                   time_initial = - t_bound * pw, time_final = t_bound * pw, time_step = .5 * asec,
                                                                                   electric_potential = efield,
                                                                                   prefactor = prefactor,
                                                                                   kernel = ide.velocity_guassian_kernel, kernel_kwargs = dict(tau_alpha = tau_alpha, width = L),
                                                                                   evolution_method = 'TRAP',
                                                                                   pulse_width = pw,
                                                                                   phase = phase,
                                                                                   flu = flu,
                                                                                   ))

        plt_kwargs = dict(
            target_dir = OUT_DIR,
        )

        results = cp.utils.multi_map(run, specs, processes = 4)

        for log in (True, False):
            if not log:
                y_lower_limit = 0
            else:
                y_lower_limit = None

            cp.plots.xy_plot(f'ionization_vs_phase__pw={uround(pw, asec, 3)}as_flu={uround(flu, Jcm2, 3)}Jcm2__log={log}',
                             [r.spec.phase for r in results],
                             [np.abs(r.a[-1]) ** 2 for r in results],
                             x_label = r'CEP $\varphi$ ($\pi$)', x_unit = 'rad',
                             y_label = r'$  \left| a_{\mathrm{final}} \right|^2  $', y_log_axis = log, y_upper_limit = 1, y_lower_limit = y_lower_limit,
                             **plt_kwargs)

            cp.plots.xy_plot(f'ionization_vs_phase__pw={uround(pw, asec, 3)}as_flu={uround(flu, Jcm2, 3)}Jcm2__log={log}__rel',
                             [r.spec.phase for r in results],
                             [(np.abs(r.a[-1]) ** 2) / (np.abs(results[0].a[-1]) ** 2) for r in results],
                             x_label = r'CEP $\varphi$ ($\pi$)', x_unit = 'rad',
                             y_label = r'$  \left| a_{\mathrm{final}}(\varphi) \right|^2 / \left| a_{\mathrm{final}}(\varphi = 0) \right|^2  $', y_log_axis = log,
                             **plt_kwargs)
