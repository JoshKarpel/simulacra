import logging
import os

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


def run_sim(spec):
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG,
                         file_logs = True, file_name = spec.name, file_dir = OUT_DIR, file_mode = 'w') as logger:
        sim = ion.ElectricFieldSimulation(spec)

        logger.info(sim.info())

        sim.run_simulation()

        laser_frequency = sim.spec.electric_potential.frequency
        laser_period = sim.spec.electric_potential.period

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)
        sim.plot_dipole_moment_vs_time(target_dir = OUT_DIR)
        sim.plot_dipole_moment_vs_frequency(target_dir = OUT_DIR,
                                            frequency_range = laser_frequency * 22,
                                            vlines = (laser_frequency * n for n in range(22) if n % 2 != 0),
                                            first_time = 15 * laser_period, last_time = 20 * laser_period)

        sim.save(target_dir = OUT_DIR, save_mesh = False)

        logger.info(sim.info())


if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        bound = 225
        # points = 2 ** 8
        points = bound * 20
        # angular_points = 2 ** 6
        angular_points = 48

        laser_frequency = c / (1064 * nm)
        laser_period = 1 / laser_frequency

        t_init = 0
        t_final = 20 * laser_period
        t_step = laser_period / 800

        # external_potential = ion.potentials.Rectangle(start_time = 40 * asec, end_time = 80 * asec, amplitude = .1 * atomic_electric_field)
        window = ion.potentials.LinearRampWindow(ramp_on_time = 0, ramp_time = 5 * laser_period)

        base_amplitude = 1.2e10 * V / m
        amplitudes = np.sqrt(np.linspace(1, 10, 5)) * base_amplitude

        specs = []

        for amplitude in amplitudes:
            external_potential = ion.potentials.SineWave(twopi * laser_frequency, amplitude = amplitude,
                                                         window = window)
            internal_potential = ion.potentials.NuclearPotential() + ion.potentials.RadialImaginaryPotential(center = bound * bohr_radius)

            cyl_spec = ion.CylindricalSliceSpecification('cyl_dipole__{}x{}_amp={}'.format(points, round(points / 2), uround(amplitude, atomic_electric_field, 3)),
                                                         z_bound = bound * bohr_radius, z_points = points,
                                                         rho_bound = bound * bohr_radius, rho_points = points / 2,
                                                         time_initial = t_init, time_final = t_final, time_step = t_step,
                                                         internal_potential = internal_potential,
                                                         electric_potential = external_potential)
            # specs.append(cyl_spec)

            sph_spec = ion.SphericalSliceSpecification('sph_dipole__{}x{}_amp={}'.format(points, angular_points, uround(amplitude, atomic_electric_field, 3)),
                                                       r_bound = bound * bohr_radius, r_points = points,
                                                       theta_points = angular_points,
                                                       time_initial = t_init, time_final = t_final, time_step = t_step,
                                                       internal_potential = internal_potential,
                                                       electric_potential = external_potential)
            specs.append(sph_spec)

        cp.utils.multi_map(run_sim, specs, processes = 5)
