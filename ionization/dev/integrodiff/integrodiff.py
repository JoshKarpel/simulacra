import logging
import os

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *
from ionization import integrodiff as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        # electric_field = ion.Rectangle(start_time = -500 * asec, end_time = 500 * asec, amplitude = 1 * atomic_electric_field)
        electric_field = ion.SincPulse(pulse_width = 100 * asec, fluence = 1 * Jcm2)

        q = electron_charge
        m = electron_mass_reduced
        L = bohr_radius

        tau_alpha = 4 * m * (L ** 2) / hbar
        prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

        dt = 20
        t_bound = 1000

        spec = ide.IntegroDifferentialEquationSpecification('ide_test__{}__dt={}as'.format(electric_field.__class__.__name__, dt),
                                                            time_initial = -t_bound * asec, time_final = t_bound * asec, time_step = dt * asec,
                                                            prefactor = prefactor,
                                                            electric_potential = electric_field.get_electric_field_amplitude,
                                                            kernel = ide.gaussian_kernel, kernel_kwargs = dict(tau_alpha = tau_alpha)
                                                            )

        sim = spec.to_simulation()

        sim.run_simulation()

        # print(sim.y)
        # print(np.abs(sim.y) ** 2)
        # print('tau alpha (as)', tau_alpha / asec)

        sim.plot_a_vs_time(target_dir = OUT_DIR,
                           y_axis_label = r'$   \left| a_{\alpha}(t) \right|^2  $',
                           field_axis_label = r'${}(t)$'.format(str_efield),
                           field_scale = 'AEF')
