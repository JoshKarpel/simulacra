import logging
import os

import numpy as np
import scipy.sparse as sparse

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


def run_sim(spec):
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG,
                         file_logs = True, file_name = spec.name, file_dir = OUT_DIR, file_mode = 'w') as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())

        sim.run_simulation()

        laser_frequency = sim.spec.electric_potential.frequency
        laser_period = sim.spec.electric_potential.period

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)
        sim.plot_dipole_moment_vs_time(target_dir = OUT_DIR)
        sim.plot_dipole_moment_vs_frequency(target_dir = OUT_DIR,
                                            frequency_range = laser_frequency * 40,
                                            vlines = (laser_frequency * n for n in range(45) if n % 2 != 0),
                                            first_time = 5 * laser_period, last_time = 10 * laser_period)

        sim.save(target_dir = OUT_DIR, save_mesh = False)

        logger.info(sim.info())


if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        bound = 1
        # points = 2 ** 8
        points = bound * 3
        # angular_points = 2 ** 6
        angular_points = 4

        laser_frequency = c / (1064 * nm)
        laser_period = 1 / laser_frequency

        t_init = 0
        t_final = 10 * laser_period
        t_step = laser_period / 800

        # external_potential = ion.Rectangle(start_time = 40 * asec, end_time = 80 * asec, amplitude = .1 * atomic_electric_field)
        window = ion.LinearRampWindow(ramp_on_time = 0, ramp_time = 5 * laser_period)

        base_amplitude = 1.2e10 * V / m
        # amplitudes = np.sqrt(np.linspace(1, 10, 2)) * base_amplitude
        amplitudes = [base_amplitude]

        specs = []

        for amplitude in amplitudes:
            external_potential = ion.SineWave(twopi * laser_frequency, amplitude = amplitude,
                                              window = window)
            internal_potential = ion.Coulomb() + ion.RadialImaginary(center = bound * bohr_radius, width = 20 * bohr_radius, decay_time = 30 * asec)

            sph_spec = ion.SphericalHarmonicSpecification('CN__dipole__{}x{}_amp={}'.format(points, angular_points, uround(amplitude, atomic_electric_field, 3)),
                                                          r_bound = bound * bohr_radius, r_points = points,
                                                          l_points = angular_points,
                                                          time_initial = t_init, time_final = t_final, time_step = t_step,
                                                          internal_potential = internal_potential,
                                                          electric_potential = external_potential,
                                                          do_imag_ev = False)
            specs.append(sph_spec)

            sph_spec = ion.SphericalHarmonicSpecification('SO__dipole__{}x{}_amp={}'.format(points, angular_points, uround(amplitude, atomic_electric_field, 3)),
                                                          r_bound = bound * bohr_radius, r_points = points,
                                                          l_points = angular_points,
                                                          time_initial = t_init, time_final = t_final, time_step = t_step,
                                                          internal_potential = internal_potential,
                                                          electric_potential = external_potential,
                                                          do_imag_ev = False,
                                                          evolution_method = 'SO')
            specs.append(sph_spec)

        cp.utils.multi_map(run_sim, specs, processes = 2)
