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
    # test_states = [ion.HydrogenBoundState(n, l) for n in range(6) for l in range(n)]

    l_points = 5
    bound = 100
    points_per_bohr_radius = 8

    spec_kwargs = {'r_bound': bound * bohr_radius,
                   'r_points': bound * points_per_bohr_radius,
                   'l_points': 100,
                   'initial_state': ion.HydrogenBoundState(1, 0),
                   'time_initial': 0 * asec,
                   'time_final': 200 * asec,
                   'time_step': 1 * asec,
                   }

    OUT_DIR = os.path.join(OUT_DIR, 'bound={}_ppbr={}'.format(bound, points_per_bohr_radius))
    sim = ion.SphericalHarmonicSpecification('eig', **spec_kwargs).to_simulation()

    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO, file_logs = True, file_mode = 'w', file_dir = OUT_DIR, file_name = 'log') as logger:
        for l in range(l_points):
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
                    OUT_DIR_tmp = os.path.join(OUT_DIR, 'free_{}'.format(int((energy // 100) * 100)))
                    logger.info('Numerical Eigenstate E = {}, l = {} -> {}'.format(energy, l, state))
                else:
                    n_guess = round(np.sqrt(rydberg / np.abs(eigenvalue)))
                    if n_guess == 0:
                        n_guess = 1
                    state = ion.HydrogenBoundState(n = n_guess, l = l)
                    name = 'g__n={}_l={}'.format(state.n, state.l, np.abs(energy), l)
                    OUT_DIR_tmp = os.path.join(OUT_DIR, 'bound')
                    frac_diff = np.abs((state.energy - eigenvalue) / state.energy)
                    logger.info('Numerical Eigenstate E = {}, l = {} -> {}. Predicted n: {} -> {}. Fractional Difference: {}.'.format(energy, l, state, np.sqrt(rydberg / np.abs(eigenvalue)), n_guess, frac_diff))

                g_analytic = state.radial_function(sim.mesh.r) * sim.mesh.r  # get R(r)
                g_analytic /= np.sqrt(sim.mesh.inner_product_multiplier * np.sum(np.abs(g_analytic) ** 2))  # normalize

                y = [np.abs(eigenvector) ** 2, np.abs(g_analytic) ** 2]
                labels = ['Numeric, $E = {}$ eV, $\ell = {}$'.format(energy, l), r'${}$, $E = {}$ eV'.format(state.tex_str, uround(state.energy, eV, 5))]

                cp.utils.xy_plot(name,
                                 sim.mesh.r, *y,
                                 line_labels = labels,
                                 x_scale = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g \right|^2$', title = 'Energy = {} eV, $\ell$ = {}'.format(energy, l),
                                 target_dir = OUT_DIR_tmp)
