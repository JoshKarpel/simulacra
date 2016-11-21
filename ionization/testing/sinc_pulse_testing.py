import os
import logging

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)


def run(spec):
    OUT_DIR_mod = os.path.join(OUT_DIR, spec.out_dir_mod)

    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG,
                         file_logs = True, file_dir = OUT_DIR_mod, file_name = spec.name, file_level = logging.DEBUG) as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR_mod)
        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR_mod, log_metrics = True, name_postfix = '_log')


if __name__ == '__main__':
    with log as logger:
        bound = 100
        r_points = bound * 4
        l_points = 50

        out_dir_mod = 'bounds_r={}_l={}'.format(bound, l_points)

        n = 1
        l = 0

        initial_state = ion.BoundState(n, l, 0)

        pulse_widths = [10, 50, 100, 200, 300, 500, 1000, 2000]
        fluences = [1, 5, 10, 20]

        specs = []
        for pulse_width in pulse_widths:
            for fluence in fluences:
                pw = pulse_width * asec
                flu = fluence * J / (cm ** 2)

                t_init = -20 * pw
                t_final = -t_init

                window = ion.potentials.SymmetricExponentialWindow(window_time = t_init + (3 * pw), window_width = pw / 3)
                e_field_sin = ion.potentials.SincPulse(pulse_width = pw, fluence = flu, phase = 'sin', window = window)
                e_field_cos = ion.potentials.SincPulse(pulse_width = pw, fluence = flu, phase = 'cos', window = window)

                mask = ion.potentials.RadialCosineMask(on_radius = (bound - 25) * bohr_radius, off_radius = bound * bohr_radius)

                spec = ion.SphericalHarmonicSpecification('{}_{}__cos_pw={}_flu={}'.format(n, l, pulse_width, fluence),
                                                          time_initial = t_init, time_final = t_final,
                                                          r_points = r_points,
                                                          r_bound = bound * bohr_radius,
                                                          l_points = l_points,
                                                          initial_state = initial_state,
                                                          electric_potential = e_field_cos,
                                                          evolution_method = 'CN',
                                                          out_dir_mod = out_dir_mod)
                specs.append(spec)

                spec = ion.SphericalHarmonicSpecification('{}_{}__sin_pw={}_flu={}'.format(n, l, pulse_width, fluence),
                                                          time_initial = t_init, time_final = t_final,
                                                          r_points = r_points,
                                                          r_bound = bound * bohr_radius,
                                                          l_points = l_points,
                                                          initial_state = initial_state,
                                                          electric_potential = e_field_sin,
                                                          evolution_method = 'CN',
                                                          out_dir_mod = out_dir_mod)
                specs.append(spec)

        cp.utils.multi_map(run, specs, processes = 4)
