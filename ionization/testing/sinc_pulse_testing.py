import os
import logging

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logger = cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)


def run(spec):
    with logger as log:
        sim = ion.ElectricFieldSimulation(spec)

        log.info(sim.info())
        sim.run_simulation()
        log.info(sim.info())

        sim.plot_wavefunction_vs_time(save = True, target_dir = spec.extra_args['out_dir'])


if __name__ == '__main__':
    with logger as log:
        # n_max = 4

        bound = 50
        points = 2 ** 10
        angular_points = 2 ** 5

        pulse_width = 100 * asec

        t_init = -8 * pulse_width
        t_final = -t_init

        window = ion.potentials.LinearRampWindow(ramp_on_time = t_init + (200 * asec), ramp_time = 4 * pulse_width)
        # e_field = ion.potentials.SineWave(omega = twopi / (pulse_width), amplitude = 0.5 * atomic_electric_field, window = window)
        e_field = ion.potentials.SincPulse(pulse_width, amplitude = .5 * atomic_electric_field)

        # specs = []

        n = 1
        l = 0

        initial_state = ion.BoundState(n, l, 0)

        lag_sph_harm_spec = ion.SphericalHarmonicSpecification('{}_{}__lag_sph_harm'.format(n, l), time_initial = t_init, time_final = t_final,
                                                               mesh_type = ion.LagrangianSphericalHarmonicMesh,
                                                               r_points = points,
                                                               r_bound = bound * bohr_radius,
                                                               spherical_harmonics_max_l = angular_points - 1,
                                                               initial_state = initial_state,
                                                               electric_potential = e_field,
                                                               out_dir = OUT_DIR)
        # specs.append(lag_sph_harm_spec)

        sim = ion.ElectricFieldSimulation(lag_sph_harm_spec)
        log.info(sim)
        sim.run_simulation()
        log.info(sim)

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)
        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR, log_metrics = True)
