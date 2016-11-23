import os
import logging

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.INFO, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)


def run(spec):
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.INFO,
                         file_logs = True, file_dir = spec.out_dir_mod, file_name = spec.name, file_level = logging.DEBUG, file_mode = 'w') as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        sim.plot_wavefunction_vs_time(target_dir = spec.out_dir_mod)
        sim.plot_wavefunction_vs_time(target_dir = spec.out_dir_mod, log = True, name_postfix = '_log')
        sim.plot_angular_momentum_vs_time(target_dir = spec.out_dir_mod)
        sim.plot_angular_momentum_vs_time(target_dir = spec.out_dir_mod, log = True, name_postfix = '_log')
        sim.plot_angular_momentum_vs_time(target_dir = spec.out_dir_mod, renormalize = True, name_postfix = '_renorm')
        sim.plot_angular_momentum_vs_time(target_dir = spec.out_dir_mod, renormalize = True, log = True, name_postfix = '_log_renorm')


if __name__ == '__main__':
    with log as logger:
        bound = 200
        points_per_r = 4
        r_points = bound * points_per_r
        l_points = 100
        dt = 1

        initial_states = [ion.BoundState(1, 0), ion.BoundState(2, 0), ion.BoundState(2, 1)]

        pulse_widths = [10, 50, 100, 250, 500, 1000, 1500, 2000]
        fluences = [1, 10, 20]

        specs = []
        for initial_state in initial_states:
            for pulse_width in pulse_widths:
                for fluence in fluences:
                    t_step = dt * asec
                    if pulse_width < 60:
                        t_step /= 10
                    if pulse_width > 550:
                        t_step *= 2.5

                    pw = pulse_width * asec
                    flu = fluence * J / (cm ** 2)

                    t_init = -20 * pw
                    t_final = -t_init

                    window = ion.potentials.SymmetricExponentialWindow(window_time = t_init + (2 * pw), window_width = pw / 3)
                    e_field_sin = ion.potentials.SincPulse(pulse_width = pw, fluence = flu, phase = 'sin', window = window)
                    e_field_cos = ion.potentials.SincPulse(pulse_width = pw, fluence = flu, phase = 'cos', window = window)

                    mask = ion.potentials.RadialCosineMask(inner_radius = (bound - 25) * bohr_radius, outer_radius = bound * bohr_radius)

                    out_dir_add = 'r={}at{}_l={}__n={}_l={}__flu={}'.format(bound, points_per_r, l_points, initial_state.n, initial_state.l, fluence)
                    out_dir_mod = os.path.join(OUT_DIR, out_dir_add)

                    animators = [ion.animators.SphericalHarmonicAnimator(target_dir = out_dir_mod, postfix = '_{}'.format(bound)),
                                 ion.animators.SphericalHarmonicAnimator(target_dir = out_dir_mod, plot_limit = 30 * bohr_radius, postfix = '_30'),
                                 ion.animators.SphericalHarmonicAnimator(target_dir = out_dir_mod, plot_limit = 100 * bohr_radius, postfix = '_100')
                                 ]

                    spec = ion.SphericalHarmonicSpecification('pw={}__cos'.format(pulse_width),
                                                              time_initial = t_init, time_final = t_final, time_step = t_step,
                                                              r_points = r_points,
                                                              r_bound = bound * bohr_radius,
                                                              l_points = l_points,
                                                              initial_state = initial_state,
                                                              electric_potential = e_field_cos,
                                                              evolution_method = 'CN',
                                                              mask = mask,
                                                              animators = animators,
                                                              out_dir_mod = out_dir_mod)
                    specs.append(spec)

                    spec = ion.SphericalHarmonicSpecification('pw={}__sin'.format(pulse_width),
                                                              time_initial = t_init, time_final = t_final, time_step = t_step,
                                                              r_points = r_points,
                                                              r_bound = bound * bohr_radius,
                                                              l_points = l_points,
                                                              initial_state = initial_state,
                                                              electric_potential = e_field_sin,
                                                              evolution_method = 'CN',
                                                              mask = mask,
                                                              animators = animators,
                                                              out_dir_mod = out_dir_mod)
                    specs.append(spec)

        cp.utils.multi_map(run, specs, processes = 6)
