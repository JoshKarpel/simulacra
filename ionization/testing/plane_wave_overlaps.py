import logging
import os

from tqdm import tqdm

import numpy as np
import scipy.sparse.linalg as sparsealg
import matplotlib.pyplot as plt

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
            time_final = 300 * asec,
            time_step = 1 * asec,
            use_numeric_eigenstates_as_basis = True,
            numeric_eigenstate_energy_max = 50 * eV,
            numeric_eigenstate_l_max = 20,
            electric_potential = ion.SineWave.from_photon_energy(rydberg + 5 * eV, amplitude = .5 * atomic_electric_field),
            mask = ion.RadialCosineMask(inner_radius = .8 * bound * bohr_radius, outer_radius = bound * bohr_radius),
            store_data_every = 25,
        )

        sim = ion.SphericalHarmonicSpecification('PWTest', **spec_kwargs).to_simulation()

        sim.run_simulation()
        print(sim.info())

        plot_kwargs = dict(
            target_dir = OUT_DIR,
            bound_state_max_n = 4,
        )

        # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__no_grouping')
        # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__no_grouping__collapsed_l',
        #                               collapse_bound_state_angular_momentums = True)
        #
        # grouped_states, group_labels = sim.group_free_states_by_continuous_attr('energy', divisions = 12, cutoff_value = 150 * eV, label_unit = 'eV')
        # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__energy',
        #                               grouped_free_states = grouped_states, group_labels = group_labels)
        # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__energy__collapsed_l',
        #                               collapse_bound_state_angular_momentums = True,
        #                               grouped_free_states = grouped_states, group_labels = group_labels)
        #
        # grouped_states, group_labels = sim.group_free_states_by_discrete_attr('l', cutoff_value = 20)
        # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__l',
        #                               grouped_free_states = grouped_states, group_labels = group_labels)
        # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__l__collapsed_l',
        #                               collapse_bound_state_angular_momentums = True,
        #                               grouped_free_states = grouped_states, group_labels = group_labels)
        #
        # for log in (False, True):
        #     plot_kwargs['log'] = log
        #
        #     sim.plot_energy_spectrum(**plot_kwargs,
        #                              energy_upper_limit = 50 * eV, states = 'all',
        #                              group_angular_momentum = False)
        #     sim.plot_energy_spectrum(**plot_kwargs,
        #                              states = 'bound',
        #                              group_angular_momentum = False)
        #     sim.plot_energy_spectrum(**plot_kwargs,
        #                              energy_upper_limit = 50 * eV, states = 'free',
        #                              bins = 25,
        #                              group_angular_momentum = False)
        #
        #     sim.plot_energy_spectrum(**plot_kwargs,
        #                              energy_upper_limit = 50 * eV, states = 'all',
        #                              angular_momentum_cutoff = 10)
        #     sim.plot_energy_spectrum(**plot_kwargs,
        #                              states = 'bound',
        #                              angular_momentum_cutoff = 10)
        #     sim.plot_energy_spectrum(**plot_kwargs,
        #                              bins = 25,
        #                              energy_upper_limit = 50 * eV, states = 'free',
        #                              angular_momentum_cutoff = 10)

        # 1d tests along various theta first

        thetas = np.array([0, .5, 1, 1.5, 2])
        energies = np.linspace(.01, 50, 100) * eV
        wavenumbers = ion.electron_wavenumber_from_energy(energies)
        inner_products = np.zeros(len(wavenumbers), dtype = np.complex128) * np.NaN

        for theta in thetas:
            for ii, k in enumerate(wavenumbers):
                inner_products[ii] = sim.mesh.inner_product_with_plane_wave(k, theta * pi)

            cp.utils.xy_plot('plane_wave_overlaps__theta={}pi'.format(theta),
                             wavenumbers,
                             np.abs(inner_products) ** 2,
                             x_scale = 'per_nm',
                             target_dir = OUT_DIR)

        # then do 2d plot for all theta

        unit_value, unit_name = unit_value_and_name_from_unit('per_nm')

        thetas = np.linspace(0, twopi, 100)

        theta_mesh, wavenumber_mesh = np.meshgrid(thetas, wavenumbers, indexing = 'ij')

        inner_product_mesh = np.zeros(np.shape(theta_mesh), dtype = np.complex128)
        for ii, theta in enumerate(thetas):
            for jj, wavenumber in enumerate(wavenumbers):
                print(ii, jj)
                inner_product_mesh[ii, jj] = sim.mesh.inner_product_with_plane_wave(wavenumber, theta)

        fig = cp.utils.get_figure('full')
        fig.set_tight_layout(True)

        axis = plt.subplot(111, projection = 'polar')
        axis.set_theta_zero_location('N')
        axis.set_theta_direction('clockwise')

        plt.set_cmap(plt.cm.get_cmap('viridis'))

        color_mesh = axis.pcolormesh(theta_mesh,
                                     wavenumber_mesh / unit_value,
                                     np.abs(inner_product_mesh) ** 2,
                                     shading = 'gouraud')

        # make a colorbar
        cbar_axis = fig.add_axes([1.01, .1, .04, .8])  # add a new axis for the cbar so that the old axis can stay square
        cbar = plt.colorbar(mappable = color_mesh, cax = cbar_axis)
        cbar.ax.tick_params(labelsize = 10)

        axis.grid(True, color = ion.COLORMESH_GRID_COLOR, linestyle = ':', linewidth = .5)  # change grid color to make it show up against the colormesh
        angle_labels = ['{}\u00b0'.format(s) for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
        axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)

        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)  # increase size of tick labels
        axis.tick_params(axis = 'y', which = 'major', colors = ion.COLORMESH_GRID_COLOR, pad = 3)  # make r ticks a color that shows up against the colormesh
        axis.tick_params(axis = 'both', which = 'both', length = 0)

        axis.set_rlabel_position(80)

        max_yticks = 5
        yloc = plt.MaxNLocator(max_yticks, symmetric = False, prune = 'both')
        axis.yaxis.set_major_locator(yloc)

        fig.canvas.draw()  # must draw early to modify the axis text

        tick_labels = axis.get_yticklabels()
        for t in tick_labels:
            t.set_text(t.get_text() + r'${}$'.format(unit_name))
        axis.set_yticklabels(tick_labels)

        axis.set_rmax(np.nanmax(wavenumbers) / unit_value)

        cp.utils.save_current_figure(name = 'circle_momentum_spectrum', target_dir = OUT_DIR)

        plt.close()

        # or should I keep it as kx, kz?

        # or both as options...
