import logging
import os
from copy import deepcopy

import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = si.utils.LogManager('compy', 'ionization', stdout_level = logging.INFO)


def run(spec):
    with si.utils.LogManager('compy', 'ionization',
                             stdout_logs = True, stdout_level = logging.INFO) as logger:
        try:
            sim = spec.to_simulation()

            logger.info(sim.info())
            sim.run_simulation()
            logger.info(sim.info())
        except Exception as e:
            logger.exception(e)
            raise e


if __name__ == '__main__':
    with log as logger:
        bound = 80
        points_per_r = 4
        r_points = bound * points_per_r
        l_bound = 100
        dt = 1

        initial_states = [ion.HydrogenBoundState(1, 0), ion.HydrogenBoundState(2, 0), ion.HydrogenBoundState(2, 1)]

        test_states = tuple(ion.HydrogenBoundState(n, l) for n in range(3 + 1) for l in range(n))

        pulse_widths = [50, 100, 200, 300, 500, 800]

        fluences = [1, 5, 20]

        specs = []
        for initial_state in initial_states:
            for pulse_width in pulse_widths:
                for fluence in fluences:
                    t_step = dt * asec
                    if pulse_width < 40:
                        t_step *= .5
                    if pulse_width > 350:
                        t_step *= 4

                    pw = pulse_width * asec
                    flu = fluence * J / (cm ** 2)

                    t_init = -20 * pw
                    t_final = -t_init

                    window = ion.potentials.SymmetricExponentialTimeWindow(window_time = t_init + (2 * pw), window_width = pw / 2)
                    e_field_sin = ion.potentials.SincPulse(pulse_width = pw, fluence = flu, phase = 'sin', window = window)
                    e_field_cos = ion.potentials.SincPulse(pulse_width = pw, fluence = flu, phase = 'cos', window = window)

                    mask = ion.potentials.RadialCosineMask(inner_radius = .8 * bound * bohr_radius, outer_radius = bound * bohr_radius)

                    out_dir_add = 'r={}at{}_l={}__n={}_l={}__flu={}'.format(bound, points_per_r, l_bound, initial_state.n, initial_state.l, fluence)
                    out_dir_mod = os.path.join(OUT_DIR, out_dir_add)

                    animator_kwargs = {'target_dir': out_dir_mod,
                                       'distance_unit': 'nm',
                                       'length': 60,
                                       'top_right_axis_manager_type': ion.animators.TestStateStackplot}

                    animators = [
                        ion.animators.SphericalHarmonicAnimator(postfix = '__{}br'.format(bound), **animator_kwargs),
                        # ion.animators.SphericalHarmonicAnimator(plot_limit = 30 * bohr_radius, postfix = '__30br', **animator_kwargs),
                        ion.animators.SphericalHarmonicAnimator(plot_limit = 50 * bohr_radius, postfix = '__50br', **animator_kwargs),
                        # ion.animators.SphericalHarmonicAnimator(plot_limit = 100 * bohr_radius, postfix = '__100br', **animator_kwargs)
                    ]

                    base_kwargs = {
                        'time_initial': t_init,
                        'time_final': t_final,
                        'time_step': t_step,
                        'r_points': r_points,
                        'r_bound': bound * bohr_radius,
                        'l_bound': l_bound,
                        'initial_state': initial_state,
                        'test_states': test_states,
                        'mask': mask,
                        'animators': animators,
                        'out_dir_mod': out_dir_mod,
                    }

                    specs.append(ion.SphericalHarmonicSpecification('pw={}__cos'.format(pulse_width),
                                                                    electric_potential = e_field_cos,
                                                                    **deepcopy(base_kwargs)))

                    specs.append(ion.SphericalHarmonicSpecification('pw={}__sin'.format(pulse_width),
                                                                    electric_potential = e_field_sin,
                                                                    **deepcopy(base_kwargs)))

        si.utils.multi_map(run, specs, processes = 2)
