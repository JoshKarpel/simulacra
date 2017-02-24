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

        sim.plot_solution(target_dir = spec.out_dir,
                          y_axis_label = r'$   \left| a_{\alpha}(t) \right|^2  $',
                          f_axis_label = r'${}(t)$'.format(str_efield),
                          f_scale = 'AEF')

        return sim


if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        q = electron_charge
        m = electron_mass_reduced
        L = bohr_radius

        tau_alpha = 4 * m * (L ** 2) / hbar
        prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

        pulse_widths = np.linspace(50, 700, 100)

        # phases = [0, pi / 2]
        # phases = np.array([0, 1, 2, 3]) * pi / 4
        phases = np.array([0, 1, 2, 3, 4]) * pi / 8

        flu = 10

        t_bound_per_pw = 10

        dt = 1

        out_dir_extra = 'flu={}__dt={}as_TperPW={}__pw={}asto{}as_{}pws_phases={}'.format(round(flu, 3), dt,
                                                                                          t_bound_per_pw,
                                                                                          round(np.min(pulse_widths), 1),
                                                                                          round(np.max(pulse_widths), 1),
                                                                                          len(pulse_widths),
                                                                                          len(phases))

        OUT_DIR = os.path.join(OUT_DIR, out_dir_extra)

        specs = []
        for pw in pulse_widths:
            for phase in phases:
                electric_field = ion.SincPulse(pulse_width = pw * asec, fluence = flu * Jcm2, phase = phase,
                                               window = ion.SymmetricExponentialTimeWindow(window_time = (t_bound_per_pw - 1) * pw * asec, window_width = .5 * pw * asec))

                specs.append(ide.BoundStateIntegroDifferentialEquationSpecification('flu={}jcm2_pw={}as_phi={}'.format(round(flu, 3), round(pw, 3), round(phase, 3)),
                                                                                    time_initial = -t_bound_per_pw * pw * asec, time_final = t_bound_per_pw * pw * asec, time_step = dt * asec,
                                                                                    prefactor = prefactor,
                                                                                    f = electric_field.get_electric_field_amplitude,
                                                                                    kernel = ide.gaussian_kernel, kernel_kwargs = dict(tau_alpha = tau_alpha),
                                                                                    pulse_width = pw * asec,
                                                                                    phase = phase,
                                                                                    out_dir = OUT_DIR,
                                                                                    ))

        results = cp.utils.multi_map(run, specs, processes = 3)

        a_alpha_final = {phase: dict() for phase in phases}
        for r in results:
            a_alpha_final[r.spec.phase][r.spec.pulse_width] = r.y[-1]

        phase_labels = list(r'$\varphi = {}\pi$'.format(round(phase / pi, 3)) for phase in sorted(a_alpha_final))
        y = [cp.utils.dict_to_arrays(d) for phase, d in sorted(a_alpha_final.items())]
        x = y[0][0]
        y = [np.abs(x[1]) ** 2 for x in y]

        ivpw_kwargs = dict(
            x_label = r'Pulse Width $\tau$', x_scale = 'asec',
            y_label = r'$   \left|    a_{\alpha}  ( t_{ \mathrm{final} } )    \right|^2  $',
            line_labels = phase_labels,
            vlines = [tau_alpha],
            target_dir = OUT_DIR,
        )

        cp.utils.xy_plot(out_dir_extra + '__ivpw',
                         pulse_widths * asec, *y,
                         **ivpw_kwargs,
                         y_lower_limit = 0, y_upper_limit = 1,
                         )

        cp.utils.xy_plot(out_dir_extra + '__ivpw_log',
                         pulse_widths * asec, *y,
                         y_log_axis = True,
                         **ivpw_kwargs,
                         y_upper_limit = 1,
                         )
