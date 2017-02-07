import logging
import os
import contextlib

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
    n_max = 5
    test_states = [ion.HydrogenBoundState(n, l) for n in range(n_max + 1) for l in range(n)]
    # test_states += [ion.HydrogenCoulombState(energy = e * eV, l = l) for e in range(0, 10) for l in range(10)]

    l_max = 10
    bound = 200
    points_per_bohr_radius = 4

    spec_kwargs = {'r_bound': bound * bohr_radius,
                   'r_points': bound * points_per_bohr_radius,
                   'l_points': 100,
                   'initial_state': ion.HydrogenBoundState(1, 0),
                   'time_initial': -1000 * asec,
                   'time_final': 1000 * asec,
                   'time_step': 1 * asec,
                   'test_states': test_states,
                   }

    OUT_DIR = os.path.join(OUT_DIR, 'bound={}_ppbr={}'.format(bound, points_per_bohr_radius))
    sim = ion.SphericalHarmonicSpecification('eig', **spec_kwargs).to_simulation()

    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG, file_logs = True, file_mode = 'w', file_dir = OUT_DIR, file_name = 'log') as logger:

        # CONSTRUCT EIGENBASIS
        numerical_basis = {}

        for l in range(l_max + 1):
            logger.info('working on l = {}'.format(l))
            h = sim.mesh._get_internal_hamiltonian_matrix_operator_single_l(l = l)
            with cp.utils.Timer() as t:
                eigenvalues, eigenvectors = sparsealg.eigsh(h, k = h.shape[0] - 2, which = 'SA')
            logger.info('Matrix diagonalization took: {}'.format(t))

            for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T):
                energy = uround(eigenvalue, eV, 5)  # for str representation
                eigenvector /= np.sqrt(sim.mesh.inner_product_multiplier * np.sum(np.abs(eigenvector) ** 2))  # normalize

                if energy > 0:
                    state = ion.HydrogenCoulombState(energy = eigenvalue, l = l)
                    name = 'g__{}eV_l={}'.format(np.abs(energy), l)
                    # OUT_DIR_tmp = os.path.join(OUT_DIR, 'free_{}'.format(int((energy // 100) * 100)))
                    # logger.info('Numerical Eigenstate E = {}, l = {} -> {}'.format(energy, l, state))
                else:
                    n_guess = round(np.sqrt(rydberg / np.abs(eigenvalue)))
                    if n_guess == 0:
                        n_guess = 1
                    state = ion.HydrogenBoundState(n = n_guess, l = l)
                    # name = 'g__n={}_l={}'.format(state.n, state.l, np.abs(energy), l)
                    # OUT_DIR_tmp = os.path.join(OUT_DIR, 'bound')
                    frac_diff = np.abs((state.energy - eigenvalue) / state.energy)
                    # logger.info('Numerical Eigenstate E = {}, l = {} -> {}. Predicted n: {} -> {}. Fractional Difference: {}.'.format(energy, l, state, np.sqrt(rydberg / np.abs(eigenvalue)), n_guess, frac_diff))

                numerical_basis[state] = eigenvector

        # for state, mesh in numerical_basis.items():
        #     print(state, mesh)
        print(len(numerical_basis))


        # with open(os.path.join(OUT_DIR, 'overlaps.txt'), mode = 'w') as f:
        #     with contextlib.redirect_stdout(f):
        #         numeric_bound_states = []
        #         numeric_free_states = []
        #         for state in numerical_basis:
        #             if state.energy < 0:
        #                 numeric_bound_states.append(state)
        #             else:
        #                 numeric_free_states.append(state)
        #
        #         print('NUMERIC BOUND STATES:')
        #         for state in sorted(numeric_bound_states, key = lambda x: x.tuple):
        #             print(state)
        #
        #         print('+-' * 100)
        #
        #         print('NUMERIC FREE STATES:')
        #         for state in sorted(numeric_free_states, key = lambda x: x.tuple):
        #             print(state)
        #
        #         for analytic_state in test_states:
        #             print('+-' * 100)
        #             overlaps = {}
        #
        #             analytic_mesh = sim.mesh.get_radial_g_for_state(analytic_state)
        #             analytic_mesh /= np.sqrt(sim.mesh.inner_product_multiplier * np.sum(np.abs(analytic_mesh) ** 2))
        #             conj_analytic_mesh = np.conj(analytic_mesh)
        #
        #             for numeric_state, numeric_mesh in numerical_basis.items():
        #                 # overlaps[numeric_state] = sim.mesh.state_overlap(a = analytic_mesh, b = numeric_mesh)
        #                 overlaps[numeric_state] = np.abs(np.sum(conj_analytic_mesh * numeric_mesh) * sim.mesh.inner_product_multiplier) ** 2 if analytic_state.l == numeric_state.l else 0
        #
        #             for numeric_state, overlap in sorted(overlaps.items(), key = lambda x: -x[1]):
        #                 if overlap != 0:
        #                     print('<A|N>  |{}{}|^2 = {}'.format(analytic_state.bra, numeric_state.ket, overlap))

        def overlap_with_l(l, mesh, sim):
            return np.abs(np.sum(sim.mesh.g_mesh[l, :] * mesh) * sim.mesh.inner_product_multiplier) ** 2


        electric_field = ion.SineWave.from_photon_energy(rydberg + 5 * eV, amplitude = .25 * atomic_electric_field)
        sim = ion.SphericalHarmonicSpecification('test', electric_potential = electric_field, **spec_kwargs).to_simulation()

        sim.run_simulation()
        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)
        # sim.plot_wavefunction_vs_time(target_dir = OUT_DIR, log = True)
        print(sim.info())

        with open(os.path.join(OUT_DIR, 'overlaps_post.txt'), mode = 'w') as f:
            with contextlib.redirect_stdout(f):
                overlaps = {}
                for numeric_state, numeric_mesh in numerical_basis.items():
                    overlaps[numeric_state] = overlap_with_l(numeric_state.l, numeric_mesh, sim)
                    # print(numeric_state.ket, overlap_with_l(numeric_state.l, numeric_mesh, sim))

                for numeric_state, overlap in sorted(overlaps.items(), key = lambda x: -x[1]):
                    if overlap != 0:
                        print('<A|N>  |<Psi{}|^2 = {}'.format(numeric_state.ket, overlap))
