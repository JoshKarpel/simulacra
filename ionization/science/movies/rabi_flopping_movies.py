import os
import logging
from copy import deepcopy

import numpy as np

import simulacra as si
import ionization as ion
from simulacra.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = si.utils.LogManager('compy', 'ionization', stdout_level = logging.DEBUG, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)


def run(spec):
    with si.utils.LogManager('compy', 'ionization',
                             stdout_logs = True, stdout_level = logging.INFO,
                             file_logs = True, file_dir = spec.out_dir, file_name = spec.name, file_level = logging.INFO, file_mode = 'w') as logger:
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
        state_a = ion.HydrogenBoundState(1, 0)
        state_b = ion.HydrogenBoundState(2, 1)

        amplitudes = [.001, .005, .01, .1]
        cycles = [1, 3, 5]
        dt = 1

        specs = []

        for amplitude in amplitudes:
            for cycle in cycles:
                animator_kwargs = {'target_dir': OUT_DIR,
                                   'distance_unit': 'nm',
                                   'top_right_axis_manager_type': ion.animators.TestStateStackplot}

                animators = [
                    ion.animators.SphericalHarmonicAnimator(postfix = 'full', length = 30, **animator_kwargs),
                    ion.animators.SphericalHarmonicAnimator(postfix = 'zoom_20_short', plot_limit = 20 * bohr_radius, length = 60, **animator_kwargs),
                    ion.animators.SphericalHarmonicAnimator(postfix = 'zoom_20_long', plot_limit = 20 * bohr_radius, length = 120, **animator_kwargs),
                ]

                ###

                electric_field = ion.SineWave.from_photon_energy(np.abs(state_a.energy - state_b.energy), amplitude = amplitude * atomic_electric_field)

                bound = 50
                ppbr = 20

                spec_kwargs = {'r_bound': bound * bohr_radius,
                               'r_points': bound * ppbr,
                               'l_bound': max(state_a.l, state_b.l) + 10,
                               'initial_state': state_a,
                               'test_states': (state_a, state_b),
                               'time_initial': 0 * asec,
                               'time_step': dt * asec,
                               'mask': ion.RadialCosineMask(inner_radius = 30 * bohr_radius, outer_radius = 50 * bohr_radius),
                               'animators': animators,
                               'out_dir': OUT_DIR
                               }

                dummy = ion.SphericalHarmonicSpecification('dummy', **spec_kwargs).to_simulation()

                matrix_element = np.abs(dummy.mesh.dipole_moment_expectation_value(mesh_a = dummy.mesh.get_g_for_state(state_b)))

                rabi_frequency = amplitude * atomic_electric_field * matrix_element / hbar / twopi
                rabi_time = 1 / rabi_frequency

                specs.append(ion.SphericalHarmonicSpecification('rabi_{}_{}_to_{}_{}__amp={}aef__{}cycles__bound={}br__ppbr={}__dt={}as'.format(state_a.n, state_a.l, state_b.n, state_b.l, amplitude, cycle, bound, ppbr, dt),
                                                                time_final = cycle * rabi_time,
                                                                electric_potential = electric_field,
                                                                rabi_frequency = rabi_frequency,
                                                                rabi_time = rabi_time,
                                                                **spec_kwargs, ))

        si.utils.multi_map(run, specs, processes = 3)
