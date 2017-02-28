import logging
import os

import numpy as np
import scipy.interpolate as interp

import compy as cp
import compy.cy as cy
import ionization as ion
from ionization import integrodiff as ide
from compy.units import *

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


def run(spec):
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        sim = spec.to_simulation()

        logger.debug(sim.info())
        sim.run_simulation()
        logger.debug(sim.info())

        logger.info('{} took {} seconds for {} steps, {} computations'.format(sim.name, sim.elapsed_time.total_seconds(), sim.time_steps, sim.computed_time_steps))

        # sim.plot_solution(target_dir = spec.out_dir,
        #                   y_axis_label = r'$   \left| a_{\alpha}(t) \right|^2  $',
        #                   f_axis_label = r'${}(t)$'.format(str_efield),
        #                   f_scale = 'AEF')

        return sim


if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        l = 1

        q = electron_charge
        m = electron_mass_reduced
        L = l * bohr_radius

        tau_alpha = 4 * m * (L ** 2) / hbar
        prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

        t_bound_per_pw = 10
        eps = 1e-3
        # eps_on = 'y'
        eps_on = 'dydt'

        # max_dt = 10
        # method_str = 'max_dt={}as__TperPW={}__eps={}_on_{}'.format(max_dt, t_bound_per_pw, eps, eps_on)

        min_dt_per_pw = 20
        method_str = 'min_dt_per_pw={}__TperPW={}__eps={}_on_{}'.format(min_dt_per_pw, t_bound_per_pw, eps, eps_on)

        # pulse_widths = np.array()
        # pulse_widths = np.array([140, 142.5, 145, 147.5, 150], dtype = np.float64)
        # pulse_widths = np.array([50, 100, 150, 200, 250, 300, tau_alpha / asec, 1.5 * tau_alpha / asec], dtype = np.float64)
        pulse_widths = np.array([50, 100, 150, 200, 250, 300, 400, 600, 800], dtype = np.float64)

        phases = np.linspace(0, pi, 100)

        # flu = 5
        for flu in [.1, .5, 1, 5, 10, 20]:
            physics_str = 'gaussian__lambda={}br_flu={}__pw={}asto{}as__{}pws'.format(
                l,
                round(flu, 3),
                round(np.min(pulse_widths), 1),
                round(np.max(pulse_widths), 1),
                len(pulse_widths),
            )

            specs = []

            for pw in pulse_widths:
                for phase in phases:
                    sinc = ion.SincPulse(pulse_width = pw * asec, fluence = flu * Jcm2, phase = phase)

                    electric_field = ion.GaussianPulse(pulse_width = pw * asec, fluence = flu * Jcm2, phase = phase, omega_carrier = sinc.omega_carrier,
                                                       window = ion.SymmetricExponentialTimeWindow(window_time = (t_bound_per_pw - 1) * pw * asec, window_width = .5 * pw * asec))

                    specs.append(ide.AdaptiveIntegroDifferentialEquationSpecification('flu={}jcm2_pw={}as_phi={}'.format(round(flu, 3), round(pw, 3), round(phase, 3)),
                                                                                      time_initial = -t_bound_per_pw * pw * asec, time_final = t_bound_per_pw * pw * asec,
                                                                                      time_step = .1 * asec,
                                                                                      # maximum_time_step = max_dt * asec,
                                                                                      maximum_time_step = (pw / min_dt_per_pw) * asec,
                                                                                      prefactor = prefactor,
                                                                                      f = electric_field.get_electric_field_amplitude,
                                                                                      kernel = ide.gaussian_kernel, kernel_kwargs = dict(tau_alpha = tau_alpha),
                                                                                      pulse_width = pw * asec,
                                                                                      phase = phase,
                                                                                      out_dir = OUT_DIR,
                                                                                      ))

            results = cp.utils.multi_map(run, specs, processes = 3)

            a_alpha_final = {pw: dict() for pw in pulse_widths * asec}
            for r in results:
                a_alpha_final[r.spec.pulse_width][r.spec.phase] = r.y[-1]

            pw_labels = list(r'$\tau$ = {} $\mathrm{{as}}$'.format(uround(pw, asec, 3)) for pw in sorted(a_alpha_final))
            y = [cp.utils.dict_to_arrays(d) for pw, d in sorted(a_alpha_final.items())]
            x = y[0][0]
            y = [np.abs(x[1]) ** 2 for x in y]

            IvPhase_kwargs = dict(
                x_label = r'Carrier-Envelope Phase $\varphi / \pi$',
                x_scale = pi,
                y_label = r'$   \left|    a_{\alpha}  ( t_{ \mathrm{final} } )    \right|^2  $',
                line_labels = pw_labels,
                target_dir = OUT_DIR,
            )

            cp.utils.xy_plot(method_str + '__' + physics_str + '__IvPhase',
                             phases, *y,
                             **IvPhase_kwargs,
                             y_lower_limit = 0, y_upper_limit = 1,
                             )

            cp.utils.xy_plot(method_str + '__' + physics_str + '__IvPhase_log',
                             phases, *y,
                             y_log_axis = True,
                             **IvPhase_kwargs,
                             y_upper_limit = 1,
                             )

            IvPhaseRel_kwargs = dict(
                x_label = r'Carrier-Envelope Phase $\varphi / \pi$',
                x_scale = pi,
                y_label = r'$   \left|    a^{\varphi}_{\alpha}  ( t_{ \mathrm{final} } )    \right|^2 /  \left|    a^{0}_{\alpha}  ( t_{ \mathrm{final} } )    \right|^2  $',
                line_labels = pw_labels,
                target_dir = OUT_DIR,
            )

            y_rel = [yy / yy[0] for yy in y]

            cp.utils.xy_plot(method_str + '__' + physics_str + '__IvPhase_rel',
                             phases, *y_rel,
                             **IvPhaseRel_kwargs,
                             # y_lower_limit = 0, y_upper_limit = 1,
                             )

            cp.utils.xy_plot(method_str + '__' + physics_str + '__IvPhase_rel_log',
                             phases, *y_rel,
                             y_log_axis = True,
                             **IvPhaseRel_kwargs,
                             # y_upper_limit = 1,
                             )
