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
        points = 2 ** 10

        laser_frequency = c / (1064 * nm)
        laser_period = 1 / laser_frequency

        t_init = 0
        t_final = 20 * laser_period
        t_step = 5 * asec

        # external_potential = ion.potentials.Rectangle(start_time = 40 * asec, end_time = 80 * asec, amplitude = .1 * atomic_electric_field)
        window = ion.potentials.LinearRampWindow(ramp_on_time = 0, ramp_time = 5 * laser_period)

        external_potential = ion.potentials.SineWave(twopi * laser_frequency, amplitude = .01 * atomic_electric_field,
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

        print(sim.info())
