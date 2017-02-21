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
        bound = 70
        points_per_bohr_radius = 4

        spec_kwargs = dict(
            r_bound = bound * bohr_radius,
            r_points = bound * points_per_bohr_radius,
            l_points = 70,
            initial_state = ion.HydrogenBoundState(1, 0),
            time_initial = 0 * asec,
            time_final = 500 * asec,
            time_step = 1 * asec,
            use_numeric_eigenstates_as_basis = True,
            numeric_eigenstate_energy_max = 200 * eV,
            numeric_eigenstate_l_max = 50,
            electric_potential = ion.SineWave.from_photon_energy(rydberg + 5 * eV, amplitude = .5 * atomic_electric_field),
            mask = ion.RadialCosineMask(inner_radius = .8 * bound * bohr_radius, outer_radius = bound * bohr_radius),
        )

        analytic_test_states = [ion.HydrogenBoundState(n, l) for n in range(6) for l in range(n)]

        sims = [
            ion.SphericalHarmonicSpecification('eig_1', store_data_every = 1, **spec_kwargs).to_simulation(),
            ion.SphericalHarmonicSpecification('eig_5', store_data_every = 5, **spec_kwargs).to_simulation(),
            ion.SphericalHarmonicSpecification('eig_10', store_data_every = 10, **spec_kwargs).to_simulation(),
        ]

        for sim in sims:
            sim.run_simulation()
            sim.save(target_dir = OUT_DIR, save_mesh = False)

            plot_kwargs = dict(
                target_dir = OUT_DIR,
                img_format = 'pdf',
                bound_state_max_n = 4,
            )

            sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__no_grouping')
            sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__no_grouping__collapsed_l',
                                          collapse_bound_state_angular_momentums = True)

            grouped_states, group_labels = sim.group_free_states_by_continuous_attr('energy', divisions = 12, cutoff_value = 150 * eV, label_unit = 'eV')
            sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__energy',
                                          grouped_free_states = grouped_states, group_labels = group_labels)
            sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__energy__collapsed_l',
                                          collapse_bound_state_angular_momentums = True,
                                          grouped_free_states = grouped_states, group_labels = group_labels)

            grouped_states, group_labels = sim.group_free_states_by_discrete_attr('l', cutoff_value = 20)
            sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__l',
                                          grouped_free_states = grouped_states, group_labels = group_labels)
            sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__l__collapsed_l',
                                          collapse_bound_state_angular_momentums = True,
                                          grouped_free_states = grouped_states, group_labels = group_labels)
