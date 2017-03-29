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
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        sim = spec.to_simulation()

        logger.debug(sim.info())
        sim.run_simulation()
        logger.debug(sim.info())

        logger.info('{} took {} seconds for {} steps, {} computations'.format(sim.name, sim.elapsed_time.total_seconds(), sim.time_steps, sim.computed_time_steps))

        sim.plot_solution(target_dir = spec.out_dir,
                          y_axis_label = r'$   \left| a_{\alpha}(t) \right|^2  $',
                          field_axis_label = r'${}(t)$'.format(str_efield),
                          field_scale = 'AEF')

        cp.utils.xy_plot(sim.name + '_RI',
                         sim.times,
                         np.real(sim.y), np.imag(sim.y), np.abs(sim.y), np.angle(sim.y),
                         line_labels = ('Real', 'Imag', 'Abs', 'Arg'),
                         target_dir = OUT_DIR)

        return sim


if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        l = 1

        q = electron_charge
        m = electron_mass_reduced
        L = l * bohr_radius

        tau_alpha = 4 * m * (L ** 2) / hbar
        prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

        t_bound_per_pw = 30
        eps = 1e-3
        # eps_on = 'y'
        eps_on = 'dydt'

        # max_dt = 10
        # method_str = 'max_dt={}as__TperPW={}__eps={}_on_{}'.format(max_dt, t_bound_per_pw, eps, eps_on)

        min_dt_per_pw = 30
        method_str = 'min_dt_per_pw={}__TperPW={}__eps={}_on_{}'.format(min_dt_per_pw, t_bound_per_pw, eps, eps_on)

        # pulse_widths = np.array()
        # pulse_widths = np.array([140, 142.5, 145, 147.5, 150], dtype = np.float64)
        # pulse_widths = np.array([50, 100, 150, 200, 250, 300, tau_alpha / asec, 1.5 * tau_alpha / asec], dtype = np.float64)

        # flu = 5
        flu = .62
        pw = 367
        phi = .65 * pi
        phases = [phi, pi + phi, pi - phi, - phi]

        physics_str = 'flu={}jcm2_pw={}as_phi={}pi__eps={}'.format(flu, pw, round(phi / pi, 3), eps)

        specs = []

        for phase in phases:
            electric_field = ion.SincPulse(pulse_width = pw * asec, fluence = flu * Jcm2, phase = phase,
                                           window = ion.SymmetricExponentialTimeWindow(window_time = (t_bound_per_pw - 1) * pw * asec, window_width = .5 * pw * asec))

            specs.append(ide.AdaptiveIntegroDifferentialEquationSpecification('flu={}jcm2_pw={}as_phi={}pi'.format(round(flu, 3), round(pw, 3), round(phase / pi, 3)),
                                                                              time_initial = -t_bound_per_pw * pw * asec, time_final = t_bound_per_pw * pw * asec,
                                                                              time_step = .1 * asec,
                                                                              error_on = eps_on, eps = eps,
                                                                              # maximum_time_step = max_dt * asec,
                                                                              maximum_time_step = (pw / min_dt_per_pw) * asec,
                                                                              minimum_time_step = 1e-3 * asec,
                                                                              prefactor = prefactor,
                                                                              f = electric_field.get_electric_field_amplitude,
                                                                              kernel = ide.gaussian_kernel, kernel_kwargs = dict(tau_alpha = tau_alpha),
                                                                              pulse_width = pw * asec,
                                                                              phase = phase,
                                                                              out_dir = OUT_DIR,
                                                                              ))

        results = cp.utils.multi_map(run, specs, processes = 2)

        with open(os.path.join(OUT_DIR, physics_str + '.txt'), mode = 'w') as f:
            for r in results:
                print(r.spec.phase / pi, r.y[-1], np.abs(r.y[-1]) ** 2, file = f)
