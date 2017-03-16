import logging
import os

from tqdm import tqdm

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG, file_logs = False, file_mode = 'w', file_dir = OUT_DIR, file_name = 'log') as logger:
        bound = 100
        points_per_bohr_radius = 4

        t_bound = 1000

        window = ion.SymmetricExponentialTimeWindow(window_time = .9 * t_bound * asec, window_width = 10 * asec)

        # efield = ion.SineWave.from_photon_energy(rydberg + 20 * eV, amplitude = .05 * atomic_electric_field,
        #                                                          window = ion.SymmetricExponentialTimeWindow(window_time = .9 * t_bound * asec, window_width = 10 * asec))

        efield = ion.SineWave.from_photon_energy(rydberg + 20 * eV, amplitude = .05 * atomic_electric_field, phase = pi,
                                                 window = ion.SymmetricExponentialTimeWindow(window_time = .9 * t_bound * asec, window_width = 10 * asec))
        #
        # efield += ion.SineWave.from_photon_energy(rydberg + 30 * eV, amplitude = .05 * atomic_electric_field,
        #                                           window = ion.SymmetricExponentialTimeWindow(window_time = .9 * t_bound * asec, window_width = 10 * asec))

        # efield = ion.SineWave(twopi * (c / (800 * nm)), amplitude = .01 * atomic_electric_field,
        #                       window = window)

        spec_kwargs = dict(
            r_bound = bound * bohr_radius,
            r_points = bound * points_per_bohr_radius,
            l_points = 50,
            initial_state = ion.HydrogenBoundState(1, 0),
            time_initial = -t_bound * asec,
            time_final = t_bound * asec,
            time_step = 1 * asec,
            use_numeric_eigenstates_as_basis = True,
            numeric_eigenstate_energy_max = 50 * eV,
            numeric_eigenstate_l_max = 20,
            electric_potential = efield,
            electric_potential_dc_correction = True,
            mask = ion.RadialCosineMask(inner_radius = .8 * bound * bohr_radius, outer_radius = bound * bohr_radius),
            store_data_every = 25,
        )

        sim = ion.SphericalHarmonicSpecification('PWTest_phase=pi', **spec_kwargs).to_simulation()

        sim.run_simulation()
        print(sim.info())

        sim.mesh.plot_g(target_dir = OUT_DIR)
        sim.mesh.plot_g(target_dir = OUT_DIR, name_postfix = '_25', plot_limit = 25 * bohr_radius)

        plot_kwargs = dict(
            target_dir = OUT_DIR,
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

        sim.plot_energy_spectrum(**plot_kwargs,
                                 energy_upper_limit = 50 * eV, states = 'all',
                                 group_angular_momentum = False)
        sim.plot_energy_spectrum(**plot_kwargs,
                                 states = 'bound',
                                 group_angular_momentum = False)
        sim.plot_energy_spectrum(**plot_kwargs,
                                 energy_upper_limit = 50 * eV, states = 'free',
                                 bins = 25,
                                 group_angular_momentum = False)

        sim.plot_energy_spectrum(**plot_kwargs,
                                 energy_upper_limit = 50 * eV, states = 'all',
                                 angular_momentum_cutoff = 10)
        sim.plot_energy_spectrum(**plot_kwargs,
                                 states = 'bound',
                                 angular_momentum_cutoff = 10)
        sim.plot_energy_spectrum(**plot_kwargs,
                                 bins = 25,
                                 energy_upper_limit = 50 * eV, states = 'free',
                                 angular_momentum_cutoff = 10)

        spectrum_kwargs = dict(
            target_dir = OUT_DIR,
            r_points = 500,
        )

        for log in (True, False):
            thetas, wavenumbers, along_z = sim.mesh.inner_product_with_plane_waves(thetas = [0], wavenumbers = np.linspace(.1, 50, 500) * per_nm,
                                                                                   g_mesh = sim.mesh.get_g_with_states_removed(sim.bound_states))

            cp.utils.xy_plot('along_theta=0__log={}__wavenumber'.format(log),
                             wavenumbers[0],
                             np.abs(along_z[0]) ** 2,
                             x_scale = 'per_nm', x_label = 'Wavenumber $k$',
                             y_log_axis = log,
                             target_dir = OUT_DIR)

            cp.utils.xy_plot('along_theta=0__log={}__momentum'.format(log),
                             wavenumbers[0] * hbar,
                             np.abs(along_z[0]) ** 2,
                             x_scale = 'atomic_momentum', x_label = 'Momentum $p$',
                             y_log_axis = log,
                             target_dir = OUT_DIR)

            sim.mesh.plot_electron_momentum_spectrum(r_type = 'energy', r_scale = 'eV', r_lower_lim = .1 * eV, r_upper_lim = 50 * eV,
                                                     log = log,
                                                     **spectrum_kwargs)
            sim.mesh.plot_electron_momentum_spectrum(r_type = 'wavenumber',
                                                     r_upper_lim = 40 * per_nm,
                                                     log = log,
                                                     **spectrum_kwargs)
            sim.mesh.plot_electron_momentum_spectrum(r_type = 'momentum', r_scale = 'atomic_momentum', r_lower_lim = .01 * atomic_momentum, r_upper_lim = 2.5 * atomic_momentum,
                                                     log = log,
                                                     **spectrum_kwargs)

            sim.mesh.plot_electron_momentum_spectrum(r_type = 'energy', r_scale = 'eV', r_lower_lim = .1 * eV, r_upper_lim = 50 * eV,
                                                     log = log,
                                                     g_mesh = sim.mesh.get_g_with_states_removed(sim.bound_states),
                                                     name_postfix = '__bound_removed',
                                                     **spectrum_kwargs)
            sim.mesh.plot_electron_momentum_spectrum(r_type = 'wavenumber',
                                                     r_upper_lim = 40 * per_nm,
                                                     log = log,
                                                     g_mesh = sim.mesh.get_g_with_states_removed(sim.bound_states),
                                                     name_postfix = '__bound_removed',
                                                     **spectrum_kwargs)
            sim.mesh.plot_electron_momentum_spectrum(r_type = 'momentum', r_scale = 'atomic_momentum', r_lower_lim = .01 * atomic_momentum, r_upper_lim = 2.5 * atomic_momentum,
                                                     log = log,
                                                     g_mesh = sim.mesh.get_g_with_states_removed(sim.bound_states),
                                                     name_postfix = '__bound_removed',
                                                     **spectrum_kwargs)
