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
        points_per_bohr_radius = 4

        spec_kwargs = {'r_bound': bound * bohr_radius,
                       'r_points': bound * points_per_bohr_radius,
                       'l_points': 100,
                       'initial_state': ion.HydrogenBoundState(1, 0),
                       'time_initial': 0 * asec,
                       'time_final': 200 * asec,
                       'time_step': 1 * asec,
                       'use_numeric_eigenstates_as_basis': True,
                       'numeric_eigenstate_energy_max': 5 * eV,
                       'numeric_eigenstate_l_max': 5,
                       }

        analytic_test_states = [ion.HydrogenBoundState(n, l) for n in range(6) for l in range(n)]

        sim = ion.SphericalHarmonicSpecification('eig', **spec_kwargs).to_simulation()

        print('norm', sim.mesh.norm())

        # print(sim.mesh.numeric_basis)

        for state in sim.mesh.numeric_basis:
            print(state == state.analytic_state)
            print(state is state.analytic_state)
            print(hash(state), hash(state.analytic_state))

        print('ANALYTIC OVERLAP WITH INITIAL STATE')
        for state in analytic_test_states:
            print(state, sim.mesh.state_overlap(a = state, b = sim.mesh.g_mesh))

        print('\n', '+_' * 100, '\n')

        print('NUMERIC OVERLAP WITH INITIAL STATE')
        for state in sim.mesh.numeric_basis:
            print(state, sim.mesh.state_overlap(a = state, b = sim.mesh.g_mesh))
