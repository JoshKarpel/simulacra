import logging
import os

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        bound = 30
        points = 2 ** 7

        laser_frequency = c / (1064 * nm)
        laser_period = 1 / laser_frequency

        t_init = 0
        t_final = 20 * laser_period
        t_step = laser_period / 800

        # external_potential = ion.potentials.Rectangle(start_time = 40 * asec, end_time = 80 * asec, amplitude = .1 * atomic_electric_field)
        window = ion.potentials.LinearRampWindow(ramp_on_time = 0, ramp_time = 5 * laser_period)

        amplitude = 3 * 1.2e10 * V / m
        print(amplitude / atomic_electric_field)

        external_potential = ion.potentials.SineWave(twopi * laser_frequency, amplitude = amplitude,
                                                     window = window)

        spec = ion.CylindricalSliceSpecification('dipole',
                                                 z_bound = bound * bohr_radius, z_points = points,
                                                 rho_bound = bound * bohr_radius, rho_points = points / 2,
                                                 time_initial = t_init, time_final = t_final, time_step = t_step,
                                                 electric_potential = external_potential)

        sim = ion.ElectricFieldSimulation(spec)

        print(sim.info())

        sim.run_simulation()

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)
        sim.plot_dipole_moment_vs_time(target_dir = OUT_DIR)
        sim.plot_dipole_moment_vs_frequency(target_dir = OUT_DIR,
                                            frequency_range = laser_frequency * 22,
                                            vlines = (laser_frequency * n for n in range(22) if n % 2 != 0))

        print(sim.info())
