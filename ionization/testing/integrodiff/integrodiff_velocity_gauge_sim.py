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

        return sim


if __name__ == '__main__':
    with log as logger:
        pw = 100 * asec
        t_bound = 10

        L = 1 * bohr_radius
        m = electron_mass
        q = electron_charge

        sinc = ion.SincPulse(pulse_width = pw)
        efield = ion.GaussianPulse(pulse_width = pw, fluence = 1 * Jcm2, omega_carrier = sinc.omega_carrier)
        # efield = ion.SincPulse(pulse_width = pw, fluence = 1 * Jcm2)

        # efield = ion.Rectangle(start_time = -100 * asec, end_time = 100 * asec, amplitude = .01 * atomic_electric_field)

        prefactor = - ((q / m) ** 2) / (4 * (L ** 2))
        tau_alpha = 2 * m * (L ** 2) / hbar

        specs = []
        for method in ('FE', 'BE', 'TRAP'):
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

        results = cp.utils.multi_map(run, specs, processes = 3)

        for r in results:
            print(r.name, r.a)
            r.plot_fields_vs_time(**plt_kwargs)
            r.plot_a_vs_time(**plt_kwargs)
