import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = si.utils.LogManager('compy', 'ionization', stdout_level = logging.DEBUG, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)


def run(spec):
    with si.utils.LogManager('compy', 'ionization', stdout_level = logging.DEBUG,
                             file_logs = True, file_dir = spec.out_dir_mod, file_name = spec.name, file_level = logging.DEBUG, file_mode = 'w') as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        sim.plot_wavefunction_vs_time(target_dir = spec.out_dir_mod)
        sim.plot_wavefunction_vs_time(target_dir = spec.out_dir_mod, log = True, name_postfix = '_log')
        sim.plot_angular_momentum_vs_time(target_dir = spec.out_dir_mod)
        sim.plot_angular_momentum_vs_time(target_dir = spec.out_dir_mod, log = True, name_postfix = '_log')


if __name__ == '__main__':
    with log as logger:
        bound = 200
        r_points = bound * 4
        l_bound = 128
        dt = 1

        initial_states = [ion.HydrogenBoundState(1, 0), ion.HydrogenBoundState(2, 0), ion.HydrogenBoundState(2, 1)]

        pulse_widths = [10, 50, 100, 250, 500, 1000, 1500, 2000]
        fluences = [.1, 1, 5, 10, 20]

        specs = []
        for initial_state in initial_states:
            for pulse_width in pulse_widths:
                for fluence in fluences:
                    t_step = dt * asec
                    if pulse_width < 60:
                        t_step /= 10

                    pw = pulse_width * asec
                    flu = fluence * J / (cm ** 2)

                    t_init = -20 * pw
                    t_final = -t_init

                    window = ion.SymmetricExponentialTimeWindow(window_time = t_init + (2 * pw), window_width = pw / 3)
                    e_field_sin = ion.SincPulse(pulse_width = pw, fluence = flu, phase = 'sin', window = window)
                    e_field_cos = ion.SincPulse(pulse_width = pw, fluence = flu, phase = 'cos', window = window)

                    mask = ion.RadialCosineMask(inner_radius = (bound - 25) * bohr_radius, outer_radius = bound * bohr_radius)

                    out_dir_add = 'bounds_r={}_l={}__n={}_l={}__flu={}'.format(bound, l_bound, initial_state.n, initial_state.l, fluence)
                    out_dir_mod = os.path.join(OUT_DIR, out_dir_add)

                    animators = [ion.animators.SphericalHarmonicAnimator(target_dir = out_dir_mod),
                                 ion.animators.SphericalHarmonicAnimator(target_dir = out_dir_mod, plot_limit = 30 * bohr_radius, postfix = '_lim=30'),
                                 ion.animators.SphericalHarmonicAnimator(target_dir = out_dir_mod, plot_limit = 100 * bohr_radius, postfix = '_lim=100'),
                                 ]

                    spec = ion.SphericalHarmonicSpecification('pw={}__cos'.format(pulse_width),
                                                              time_initial = t_init, time_final = t_final, time_step = t_step,
                                                              r_points = r_points,
                                                              r_bound = bound * bohr_radius,
                                                              l_bound = l_bound,
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
                                                              l_bound = l_bound,
                                                              initial_state = initial_state,
                                                              electric_potential = e_field_sin,
                                                              evolution_method = 'CN',
                                                              mask = mask,
                                                              animators = animators,
                                                              out_dir_mod = out_dir_mod)
                    specs.append(spec)

        si.utils.multi_map(run, specs, processes = 6)
