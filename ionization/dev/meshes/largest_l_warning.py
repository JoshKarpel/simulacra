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
    with cp.utils.LogManager('compy', 'ionization', stdout_level = logging.DEBUG,
                             file_logs = True, file_name = spec.name, file_dir = OUT_DIR, file_mode = 'w') as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())

        sim.run_simulation()

        logger.info(sim.info())


if __name__ == '__main__':
    with cp.utils.LogManager('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        bound = 50
        # points = 2 ** 8
        points = bound * 4
        # angular_points = 2 ** 6
        angular_points = 100

        external_potential = ion.SineWave.from_photon_energy(20 * eV, amplitude = 1 * atomic_electric_field)
        internal_potential = ion.Coulomb()

        sph_spec = ion.SphericalHarmonicSpecification('test',
                                                      r_bound = bound * bohr_radius, r_points = points,
                                                      l_bound = angular_points,
                                                      time_initial = 0, time_final = 1000 * asec, time_step = 1 * asec,
                                                      internal_potential = internal_potential,
                                                      electric_potential = external_potential)

        sim = sph_spec.to_simulation()

        logger.info(sim.info())

        sim.run_simulation()

        logger.info(sim.info())
