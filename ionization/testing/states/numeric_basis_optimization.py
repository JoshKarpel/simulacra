import logging
import os

from tqdm import tqdm

import numpy as np
import scipy.sparse.linalg as sparsealg

import compy as cp
import compy.cy as cy
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG, file_logs = False, file_mode = 'w', file_dir = OUT_DIR, file_name = 'log') as logger:
        bound = 100
        points_per_bohr_radius = 16
        l_max = 10

        sim = ion.SphericalHarmonicSpecification('eig',
                                                 r_bound = bound * bohr_radius,
                                                 r_points = bound * points_per_bohr_radius,
                                                 l_bound = 50,
                                                 use_numeric_eigenstates_as_basis = True,
                                                 numeric_eigenstate_energy_max = 500 * eV,
                                                 numeric_eigenstate_l_max = l_max,
                                                 ).to_simulation()

        print('max?:', ((bound * points_per_bohr_radius) - 2) * (l_max + 1))
        print('total:', len(sim.spec.test_states))
        bound = len(list(sim.bound_states))
        free = len(list(sim.free_states))
        print('{} free + {} bound = {} total'.format(free, bound, free + bound))
