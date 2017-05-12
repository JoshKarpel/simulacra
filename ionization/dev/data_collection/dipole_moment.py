import logging
import os

import simulacra as si
import numpy as np
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
# OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME + '_masked')


def run_sim(spec):
    with si.utils.LogManager('compy', 'ionization', stdout_level = logging.DEBUG,
                             file_logs = True, file_name = spec.name, file_dir = OUT_DIR, file_mode = 'w') as logger:
        # sim = ion.ElectricFieldSimulation(spec)
        sim = spec.to_simulation()

        ###
        if sim.spec.do_imag_ev:
            for _ in range(100):
                sim.mesh.evolve(-1j * asec)

            logger.info(sim.mesh.norm)

            sim.mesh.g_mesh /= np.sqrt(sim.mesh.norm)

            logger.info(sim.mesh.norm)
            logger.info(sim.mesh.state_overlap(ion.HydrogenBoundState(1)))
            logger.info(sim.mesh.state_overlap(ion.HydrogenBoundState(2, 0)))
            logger.info(sim.mesh.state_overlap(ion.HydrogenBoundState(2, 1)))
            logger.info(sim.mesh.state_overlap(ion.HydrogenBoundState(3, 0)))
            logger.info(sim.mesh.state_overlap(ion.HydrogenBoundState(3, 1)))
            logger.info(sim.mesh.state_overlap(ion.HydrogenBoundState(3, 2)))
        ###

        logger.info(sim.info())

        sim.run_simulation()

        laser_frequency = sim.spec.electric_potential.frequency
        laser_period = sim.spec.electric_potential.period

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)
        sim.plot_dipole_moment_vs_time(target_dir = OUT_DIR)

        for p, q in [(10, 15), (10, 20), (10, 25), (10, 30),
                     (15, 20), (15, 25), (15, 30),
                     (20, 25), (20, 30),
                     (25, 30)]:
            sim.plot_dipole_moment_vs_frequency(target_dir = OUT_DIR,
                                                frequency_range = laser_frequency * 40,
                                                vlines = (laser_frequency * n for n in range(45) if n % 2 != 0),
                                                first_time = p * laser_period, last_time = q * laser_period,
                                                name_postfix = '__{}to{}'.format(p, q))

        sim.save(target_dir = OUT_DIR, save_mesh = False)

        logger.info(sim.info())


if __name__ == '__main__':
    with si.utils.LogManager('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        bound = 225
        # points = 2 ** 8
        points = bound * 4
        # angular_points = 2 ** 6
        angular_points = 49

        laser_frequency = c / (1064 * nm)
        laser_period = 1 / laser_frequency

        t_init = 0
        t_final = 30 * laser_period
        t_step = laser_period / 800

        # external_potential = ion.Rectangle(start_time = 40 * asec, end_time = 80 * asec, amplitude = .1 * atomic_electric_field)
        window = ion.LinearRampTimeWindow(ramp_on_time = 0, ramp_time = 5 * laser_period)

        base_amplitude = 1.2e10 * V / m
        amplitudes = np.sqrt(np.linspace(1, 10, 2)) * base_amplitude

        specs = []

        for amplitude in amplitudes:
            external_potential = ion.SineWave(twopi * laser_frequency, amplitude = amplitude, window = window)
            # internal_potential = ion.Coulomb() + ion.RadialImaginary(center = bound * bohr_radius, width = 20 * bohr_radius, decay_time = 30 * asec)
            internal_potential = ion.Coulomb()

            # mask = None
            mask = ion.RadialCosineMask(inner_radius = (bound - 50 * bohr_radius), outer_radius = bound * bohr_radius, smoothness = 8)

            sph_spec = ion.SphericalHarmonicSpecification('CN__dipole__{}x{}_amp={}'.format(points, angular_points, uround(amplitude, atomic_electric_field, 3)),
                                                          r_bound = bound * bohr_radius, r_points = points,
                                                          l_bound = angular_points,
                                                          time_initial = t_init, time_final = t_final, time_step = t_step,
                                                          internal_potential = internal_potential,
                                                          electric_potential = external_potential,
                                                          mask = mask,
                                                          do_imag_ev = False)
            specs.append(sph_spec)

            sph_spec = ion.SphericalHarmonicSpecification('CN_I__dipole__{}x{}_amp={}'.format(points, angular_points, uround(amplitude, atomic_electric_field, 3)),
                                                          r_bound = bound * bohr_radius, r_points = points,
                                                          l_bound = angular_points,
                                                          time_initial = t_init, time_final = t_final, time_step = t_step,
                                                          internal_potential = internal_potential,
                                                          electric_potential = external_potential,
                                                          mask = mask,
                                                          do_imag_ev = True)
            specs.append(sph_spec)

            sph_spec = ion.SphericalHarmonicSpecification('SO__dipole__{}x{}_amp={}'.format(points, angular_points, uround(amplitude, atomic_electric_field, 3)),
                                                          r_bound = bound * bohr_radius, r_points = points,
                                                          l_bound = angular_points,
                                                          time_initial = t_init, time_final = t_final, time_step = t_step,
                                                          internal_potential = internal_potential,
                                                          electric_potential = external_potential,
                                                          evolution_method = 'SO',
                                                          mask = mask,
                                                          do_imag_ev = False)
            specs.append(sph_spec)

            sph_spec = ion.SphericalHarmonicSpecification('SO_I__dipole__{}x{}_amp={}'.format(points, angular_points, uround(amplitude, atomic_electric_field, 3)),
                                                          r_bound = bound * bohr_radius, r_points = points,
                                                          l_bound = angular_points,
                                                          time_initial = t_init, time_final = t_final, time_step = t_step,
                                                          internal_potential = internal_potential,
                                                          electric_potential = external_potential,
                                                          evolution_method = 'SO',
                                                          mask = mask,
                                                          do_imag_ev = True)
            specs.append(sph_spec)

        si.utils.multi_map(run_sim, specs, processes = 3)
