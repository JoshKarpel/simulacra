import os
import logging
from copy import deepcopy

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)

if __name__ == '__main__':
    with log as logger:
        state_a = ion.HydrogenBoundState(1, 0)
        state_b = ion.HydrogenBoundState(2, 1)

        amplitude = .001
        cycles = 5
        dt = 1

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
                       'l_points': max(state_a.l, state_b.l) + 10,
                       'initial_state': state_a,
                       'test_states': (state_a, state_b),
                       'time_initial': 0 * asec,
                       'time_step': dt * asec,
                       'mask': ion.RadialCosineMask(inner_radius = 30 * bohr_radius, outer_radius = 50 * bohr_radius),
                       'animators': animators
                       }

        dummy = ion.SphericalHarmonicSpecification('dummy', **spec_kwargs).to_simulation()

        matrix_element = np.abs(dummy.mesh.dipole_moment_expectation_value(mesh_a = dummy.mesh.get_g_for_state(state_b)))

        matrix_element_theory = ((2 ** 7.5) / (3 ** 5)) * bohr_radius * proton_charge

        print(matrix_element, matrix_element_theory)

        rabi_frequency = amplitude * atomic_electric_field * matrix_element / hbar / twopi
        rabi_time = 1 / rabi_frequency

        sim = ion.SphericalHarmonicSpecification('rabi_{}_{}_to_{}_{}__amp={}aef__{}cycles__bound={}br__ppbr={}__dt={}as'.format(state_a.n, state_a.l, state_b.n, state_b.l, amplitude, cycles, bound, ppbr, dt),
                                                 **spec_kwargs,
                                                 time_final = cycles * rabi_time,
                                                 # time_final = 200 * asec,
                                                 electric_potential = electric_field).to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)
