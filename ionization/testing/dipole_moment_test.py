import logging
import os

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        bound = 30
        points = 2 ** 10

        print(points, points / 2)

        t_init = 0
        t_final = 5 * fsec
        t_step = 5 * asec

        # external_potential = ion.potentials.Rectangle(start_time = 40 * asec, end_time = 80 * asec, amplitude = .1 * atomic_electric_field)
        external_potential = ion.potentials.SineWave(twopi * (c / (1064 * nm)), amplitude = .01 * atomic_electric_field)

        spec = ion.CylindricalSliceSpecification('dipole',
                                                 z_bound = bound * bohr_radius, z_points = points,
                                                 rho_bound = bound * bohr_radius, rho_points = points / 2,
                                                 time_initial = t_init, time_final = t_final, time_step = t_step,
                                                 electric_potential = external_potential)

        sim = ion.ElectricFieldSimulation(spec)

        sim.run_simulation()

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)
        sim.plot_dipole_moment_vs_time(target_dir = OUT_DIR)
