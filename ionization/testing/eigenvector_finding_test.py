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
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        # test_states = [ion.HydrogenBoundState(n, l) for n in range(6) for l in range(n)]

        l = 2
        bound = 100
        points_per_bohr_radius = 4

        spec_kwargs = {'r_bound': bound * bohr_radius,
                       'r_points': bound * points_per_bohr_radius,
                       'l_points': 100,
                       'initial_state': ion.HydrogenBoundState(1, 0),
                       'time_initial': 0 * asec,
                       'time_final': 200 * asec,
                       'time_step': 1 * asec,
                       # 'test_states': test_states,
                       # 'electric_potential': ion.Rectangle(50 * asec, 100 * asec, amplitude = 1 * atomic_electric_field)
                       }

        OUT_DIR = os.path.join(OUT_DIR, 'bound={}_ppbr={}'.format(bound, points_per_bohr_radius))

        differences = []
        fractional_differences = []
        labels = []

        pre_post_differences = []
        pre_post_fractional_differences = []

        print(l)
        sim = ion.SphericalHarmonicSpecification('eig', **spec_kwargs).to_simulation()

        g_analytic = sim.spec.initial_state.radial_function(sim.mesh.r) * sim.mesh.r / sim.mesh.norm  # analytic mesh reference

        h = sim.mesh._get_internal_hamiltonian_matrix_operator_single_l(l = l)
        # dr = sim.mesh.delta_r / bohr_radius
        # print('correction', dr)
        # h.data[1][0] *= (1 + (dr * (1 + dr) / 8))

        # h.data[1] -= sim.spec.initial_state.energy
        # h = h.tocsc()

        with cp.utils.Timer() as t:
            eigenvalues, eigenvectors = sparsealg.eigsh(h, k = h.shape[0] - 2, which = 'SA')
        print(t)

        # print(np.shape(h))
        # print(len(eigenvalues), len(eigenvectors))

        # print(eigenvalues / eV)
        # print(eigenvalues[:10] / eV)

        # analytic_energies = np.array(list(ion.HydrogenBoundState(n = n, l = 0).energy for n in range(1, 20)))
        # print(analytic_energies[:10] / eV)
        # print(1 - eigenvalues[:5] / analytic_energies[:5])

        # g_discrete = eigenvectors[:, 0] / np.sqrt(sim.mesh.inner_product_multiplier * np.sum(np.abs(eigenvectors[:, 0]) ** 2))

        for eigenvalue, eigenvector in tqdm(zip(eigenvalues, eigenvectors.T), total = len(eigenvalues)):
            energy = uround(eigenvalue, eV, 5)
            s = 'p' if energy > 0 else 'm'
            name = 'g_{}{}eV_l={}'.format(s, np.abs(energy), l)
            eigenvector /= np.sqrt(sim.mesh.inner_product_multiplier * np.sum(np.abs(eigenvector) ** 2))
            cp.utils.xy_plot(name,
                             sim.mesh.r, np.abs(eigenvector) ** 2,
                             x_scale = 'bohr_radius', x_label = r'$r$', y_label = r'$\left| g \right|^2$', title = 'Energy = {} eV'.format(energy),
                             target_dir = OUT_DIR)
