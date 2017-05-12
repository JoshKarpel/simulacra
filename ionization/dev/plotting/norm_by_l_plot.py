import logging
import os

import numpy as np
import scipy.sparse as sparse

import compy as cp
import ionization as ion
from units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.LogManager('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        bound = 50
        # points = 2 ** 8
        points = bound * 4
        # angular_points = 2 ** 6
        angular_points = 25

        laser_frequency = c / (1064 * nm)
        laser_period = 1 / laser_frequency

        # laser_frequency = 1 / (50 * asec)
        # laser_period = 1 / laser_frequency

        t_init = 0
        t_final = 5 * laser_period
        t_step = laser_period / 800

        # external_potential = ion.Rectangle(start_time = 40 * asec, end_time = 80 * asec, amplitude = .1 * atomic_electric_field)
        window = ion.LinearRampTimeWindow(ramp_on_time = 0, ramp_time = 1 * laser_period)
        amplitude = 1 * atomic_electric_field
        external_potential = ion.SineWave(twopi * laser_frequency, amplitude = amplitude,
                                          window = window)
        internal_potential = ion.Coulomb() + ion.RadialImaginary(center = bound * bohr_radius, width = 20 * bohr_radius, decay_time = 30 * asec)

        sph_spec = ion.SphericalHarmonicSpecification('test',
                                                      r_bound = bound * bohr_radius, r_points = points,
                                                      l_bound = angular_points,
                                                      time_initial = t_init, time_final = t_final, time_step = t_step,
                                                      internal_potential = internal_potential,
                                                      electric_potential = external_potential)

        sim = sph_spec.to_simulation()

        print(sim.mesh.norm_by_l)

        sim.run_simulation()
        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)
        sim.plot_angular_momentum_vs_time(target_dir = OUT_DIR)
