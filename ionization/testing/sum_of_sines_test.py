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
        bound = 200
        points_per_bohr_radius = 4

        t_bound = 1000

        window = ion.SymmetricExponentialTimeWindow(window_time = .9 * t_bound * asec, window_width = 10 * asec)

        # efield = ion.SineWave.from_photon_energy(rydberg + 20 * eV, amplitude = .05 * atomic_electric_field,
        #                                                          window = ion.SymmetricExponentialTimeWindow(window_time = .9 * t_bound * asec, window_width = 10 * asec))

        # efield = ion.SineWave.from_photon_energy(rydberg + 20 * eV, amplitude = .05 * atomic_electric_field,
        #                                          window = ion.SymmetricExponentialTimeWindow(window_time = .9 * t_bound * asec, window_width = 10 * asec))
        #
        # efield += ion.SineWave.from_photon_energy(rydberg + 30 * eV, amplitude = .05 * atomic_electric_field,
        #                                           window = ion.SymmetricExponentialTimeWindow(window_time = .9 * t_bound * asec, window_width = 10 * asec))

        efield = ion.SumOfSinesPulse(pulse_width = 200 * asec, pulse_frequency_ratio = 5, number_of_modes = 71, fluence = .1 * Jcm2,
                                     window = window)

        spec_kwargs = dict(
            r_bound = bound * bohr_radius,
            r_points = bound * points_per_bohr_radius,
            l_points = 200,
            initial_state = ion.HydrogenBoundState(1, 0),
            time_initial = -t_bound * asec,
            time_final = t_bound * asec,
            time_step = 1 * asec,
            use_numeric_eigenstates_as_basis = True,
            numeric_eigenstate_energy_max = 100 * eV,
            numeric_eigenstate_l_max = 50,
            electric_potential = efield,
            electric_potential_dc_correction = True,
            mask = ion.RadialCosineMask(inner_radius = .8 * bound * bohr_radius, outer_radius = bound * bohr_radius),
            store_data_every = 25,
        )

        sim = ion.SphericalHarmonicSpecification('PWTest', **spec_kwargs).to_simulation()

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

        # 1d tests along various theta first

        # thetas = np.array([0, .5, 1, 1.5, 2])
        # energies = np.linspace(.01, 50, 100) * eV
        # wavenumbers = ion.electron_wavenumber_from_energy(energies)
        # inner_products = np.zeros(len(wavenumbers), dtype = np.complex128) * np.NaN
        #
        # for theta in thetas:
        #     for ii, k in enumerate(wavenumbers):
        #         inner_products[ii] = sim.mesh.inner_product_with_plane_wave(k, theta * pi)
        #
        #     cp.utils.xy_plot('plane_wave_overlaps__theta={}pi'.format(theta),
        #                      wavenumbers,
        #                      np.abs(inner_products) ** 2,
        #                      x_scale = 'per_nm',
        #                      target_dir = OUT_DIR)

        # then do 2d plot for all theta

        spectrum_kwargs = dict(
            target_dir = OUT_DIR,
            r_points = 500,
        )

        for log in (True, False):
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

            # unit_value, unit_name = unit_value_and_name_from_unit('per_nm')
            #
            # thetas = np.linspace(0, twopi, 100)
            # wavenumbers = np.linspace(50, .1, 100) * per_nm
            #
            # # theta_mesh, wavenumber_mesh = np.meshgrid(thetas, wavenumbers, indexing = 'ij')
            #
            # # inner_product_mesh = np.zeros(np.shape(theta_mesh), dtype = np.complex128)
            #
            # # with cp.utils.Timer() as t:
            # #     for ii, theta in enumerate(thetas):
            # #         for jj, wavenumber in enumerate(wavenumbers):
            # #             print(ii, jj)
            # #             inner_product_mesh[ii, jj] = sim.mesh.inner_product_with_plane_wave(wavenumber, theta)
            # # print(t)
            #
            # # with cp.utils.Timer() as t:
            # #     for jj, wavenumber in enumerate(wavenumbers):
            # #         print(jj)
            # #         inner_product_mesh[:, jj] = sim.mesh.inner_product_with_plane_waves_k(wavenumber, thetas)
            # # print(t)
            #
            # # with cp.utils.Timer() as t:
            # #     for ii, theta in enumerate(thetas):
            # #         print(ii)
            # #         inner_product_mesh[ii, :] = sim.mesh.inner_product_with_plane_waves_theta(theta, wavenumbers)
            # # print(t)
            #
            # with cp.utils.Timer() as t:
            #     theta_mesh, wavenumber_mesh, inner_product_mesh = sim.mesh.inner_product_with_plane_waves(thetas, wavenumbers)
            # print(t)
            #
            # print(np.sum(np.abs(inner_product_mesh) ** 2))
            #
            # print('dtheta', thetas[1] - thetas[0])
            # print('dk', wavenumbers[1] - wavenumbers[0])
            #
            # d_theta = np.abs(thetas[1] - thetas[0])
            # d_k = np.abs(wavenumbers[1] - wavenumbers[0])
            # print('norm = ', sim.mesh.norm())
            # print('norm?', pi * d_theta * d_k * np.sum(np.abs(np.sin(theta_mesh)) * (wavenumber_mesh ** 2) * (np.abs(inner_product_mesh) ** 2)))
            #
            # fig = cp.utils.get_figure('full', aspect_ratio = 1)
            # fig.set_tight_layout(True)
            #
            # axis = plt.subplot(111, projection = 'polar')
            # axis.set_theta_zero_location('N')
            # axis.set_theta_direction('clockwise')
            #
            # plt.set_cmap(plt.cm.get_cmap('viridis'))
            #
            # color_mesh = axis.pcolormesh(theta_mesh,
            #                              wavenumber_mesh / unit_value,
            #                              np.abs(inner_product_mesh) ** 2,
            #                              shading = 'gouraud')
            #
            # # make a colorbar
            # cbar_axis = fig.add_axes([1.01, .1, .04, .8])  # add a new axis for the cbar so that the old axis can stay square
            # cbar = plt.colorbar(mappable = color_mesh, cax = cbar_axis)
            # cbar.ax.tick_params(labelsize = 10)
            #
            # axis.grid(True, color = ion.COLORMESH_GRID_COLOR, linestyle = ':', linewidth = .5)  # change grid color to make it show up against the colormesh
            # angle_labels = ['{}\u00b0'.format(s) for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
            # axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)
            #
            # axis.tick_params(axis = 'both', which = 'major', labelsize = 10)  # increase size of tick labels
            # axis.tick_params(axis = 'y', which = 'major', colors = ion.COLORMESH_GRID_COLOR, pad = 3)  # make r ticks a color that shows up against the colormesh
            # axis.tick_params(axis = 'both', which = 'both', length = 0)
            #
            # axis.set_rlabel_position(80)
            #
            # max_yticks = 5
            # yloc = plt.MaxNLocator(max_yticks, symmetric = False, prune = 'both')
            # axis.yaxis.set_major_locator(yloc)
            #
            # fig.canvas.draw()  # must draw early to modify the axis text
            #
            # tick_labels = axis.get_yticklabels()
            # for t in tick_labels:
            #     t.set_text(t.get_text() + r'${}$'.format(unit_name))
            # axis.set_yticklabels(tick_labels)
            #
            # axis.set_rmax(np.nanmax(wavenumbers) / unit_value)
            #
            # cp.utils.save_current_figure(name = 'circle_momentum_spectrum', target_dir = OUT_DIR)
            #
            # plt.close()

            # or should I keep it as kx, kz?

            # or both as options...
