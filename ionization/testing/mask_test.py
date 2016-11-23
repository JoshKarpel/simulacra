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
        mask = ion.potentials.RadialCosineMask(inner_radius = 40 * bohr_radius, outer_radius = 49 * bohr_radius)
        sim = ion.SphericalHarmonicSpecification('mask',
                                                 r_bound = 50 * bohr_radius, r_points = 900,
                                                 l_points = 50,
                                                 mask = mask).to_simulation()

        sim.run_simulation()

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)
