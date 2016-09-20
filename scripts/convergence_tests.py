import os
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

import compy as cp
from compy.units import *
import compy.quantum.hydrogenic as hyd


# def plot_2d(mesh, x_points, y_points, title = 'title'):
#     fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
#     fig.set_tight_layout(True)
#     axis = plt.subplot(111)
#
#     color_mesh = axis.pcolormesh(y_points, x_points,
#                                  mesh,
#                                  shading = 'gouraud', cmap = plt.cm.viridis)
#
#     axis.set_xlabel(r'$\rho$ points', fontsize = 15)
#     axis.set_ylabel(r'$z$ points', fontsize = 15)
#
#     title = axis.set_title(title, fontsize = 15)
#     title.set_y(1.05)  # move title up a bit
#
#     # make a colorbar
#     cbar = fig.colorbar(mappable = color_mesh, ax = axis)
#     cbar.ax.tick_params(labelsize = 10)
#
#     axis.axis('tight')  # removes blank space between color mesh and axes
#
#     axis.grid(True, color = 'pink', linestyle = ':')  # change grid color to make it show up against the colormesh
#
#     axis.tick_params(labelright = True, labeltop = True)  # ticks on all sides
#     axis.tick_params(axis = 'both', which = 'major', labelsize = 10)  # increase size of tick labels
#     axis.tick_params(axis = 'both', which = 'both', length = 0)
#
#     # set upper and lower y ticks to not display to avoid collisions with the x ticks at the edges
#     y_ticks = axis.yaxis.get_major_ticks()
#     y_ticks[0].label1.set_visible(False)
#     y_ticks[0].label2.set_visible(False)
#     y_ticks[-1].label1.set_visible(False)
#     y_ticks[-1].label2.set_visible(False)
#
#
# def cylindrical_slice_2d_z_rho_points(z_max_power, z_tests, rho_max_power, rho_tests, state):
#     z_powers = np.linspace(1, z_max_power + 1, z_tests)
#     z_points = 2 ** z_powers
#     rho_powers = np.linspace(1, rho_max_power + 1, rho_tests)
#     rho_points = 2 ** rho_powers
#
#     print(z_points)
#     print(rho_points)
#
#     norm = np.zeros(shape = (len(z_points), len(rho_points)))
#     energy = np.zeros(shape = (len(z_points), len(rho_points)))
#
#     print(energy)
#
#     for z, z_point in enumerate(z_points):
#         for rho, rho_point in enumerate(rho_points):
#             spec = hyd.CylindricalSliceSpecification('test',
#                                                      initial_state = state,
#                                                      z_points = z_point, rho_points = rho_point)
#
#             sim = hyd.ElectricFieldSimulation(spec)
#
#             norm[z, rho] = sim.mesh.norm
#             energy[z, rho] = sim.mesh.energy_expectation_value
#
#     print(norm)
#     print(energy / eV)
#
#     plot_2d(1 - norm, z_powers, rho_powers)
#     cp.utils.save_current_figure(name = 'norm', target_dir = None, img_format = 'pdf')
#     plt.close()
#
#     plot_2d(energy / rydberg, z_powers, rho_powers)
#     cp.utils.save_current_figure(name = 'energy', target_dir = None, img_format = 'pdf')


# def cylindrical_slice_norm_energy(z_points, state):
#     rho_points = z_points / 2
#
#     norm = np.zeros(len(rho_points))
#     energy = np.zeros(len(rho_points))
#
#     for ii, (z, rho) in enumerate(zip(z_points, rho_points)):
#         print(ii)
#         spec = hyd.CylindricalSliceSpecification('test',
#                                                  initial_state = state,
#                                                  z_points = z, rho_points = rho)
#
#         sim = hyd.ElectricFieldSimulation(spec)
#
#         norm[ii] = sim.mesh.norm
#         energy[ii] = sim.mesh.energy_expectation_value
#
#     fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
#     fig.set_tight_layout(True)
#     axis = plt.subplot(111)
#
#     axis.set_xlabel(r'$z$ points', fontsize = 15)
#
#     axis.set_xscale('log')
#     axis.set_yscale('log')
#
#     axis.set_xlim(z_points[0], z_points[-1])
#
#     axis.grid(True, color = 'black', linestyle = ':')  # change grid color to make it show up against the colormesh
#
#     ################
#
#     axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \left< \psi | \psi \right> \right)$ ', fontsize = 15)
#     title = axis.set_title('Norm Error', fontsize = 15)
#     title.set_y(1.05)  # move title up a bit
#
#     scaled_norm = np.abs(1 - norm)
#     axis.set_ylim(np.min(scaled_norm), np.max(scaled_norm))
#     line, = axis.plot(z_points, scaled_norm)
#
#     cp.utils.save_current_figure(name = 'norm', target_dir = None, img_format = 'pdf')
#
#     ###############
#
#     axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \frac{\left< \psi | H | \psi \right>}{E_n} \right)$ ', fontsize = 15)
#     title = axis.set_title('Energy Error', fontsize = 15)
#
#     scaled_energy = np.abs(1 - np.abs((energy / rydberg)))
#     print(energy / eV)
#     print(scaled_energy)
#     axis.set_ylim(np.min(scaled_energy), np.max(scaled_energy))
#     line.set_ydata(scaled_energy)
#
#     cp.utils.save_current_figure(name = 'energy', target_dir = None, img_format = 'pdf')
#
#     #################
#
#     plt.close()


def cylindrical_slice_norm_energy(z_points, states, bound = 30 * bohr_radius):
    z_points = np.rint(z_points)
    rho_points = np.rint(z_points / 2)

    norms = {state: np.zeros(len(rho_points)) for state in states}
    energies = {state: np.zeros(len(rho_points)) for state in states}

    for state in states:
        for ii, (z, rho) in enumerate(zip(z_points, rho_points)):
            print(state, ii, z, rho)
            spec = hyd.CylindricalSliceSpecification('test',
                                                     z_bound = bound, rho_bound = bound,
                                                     initial_state = state,
                                                     z_points = z, rho_points = rho)

            sim = hyd.ElectricFieldSimulation(spec)

            norms[state][ii] = sim.mesh.norm
            energies[state][ii] = sim.mesh.energy_expectation_value

    fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
    fig.set_tight_layout(True)
    axis = plt.subplot(111)

    axis.set_xlabel(r'$N_z$, $N_{\rho} = N_z / 2$', fontsize = 15)

    axis.set_xscale('log')
    axis.set_yscale('log')

    axis.set_xlim(z_points[0], z_points[-1])

    axis.grid(True, color = 'black', linestyle = ':')  # change grid color to make it show up against the colormesh

    ################

    axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \left< \psi | \psi \right> \right)$', fontsize = 15)
    title = axis.set_title('Norm Error', fontsize = 15)
    title.set_y(1.05)  # move title up a bit

    scaled_norm = {state: np.abs(1 - norm) for state, norm in norms.items()}
    # axis.set_ylim(np.min(scaled_norm), np.max(scaled_norm))

    lines = []

    for state in states:
        lines.append(axis.plot(z_points, scaled_norm[state], label = r'${}$'.format(state.tex_str))[0])

    axis.legend(loc = 'best', fontsize = 12)

    cp.utils.save_current_figure(name = 'norm', target_dir = None, img_format = 'pdf')

    ###############

    axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \frac{\left< \psi | H | \psi \right>}{E_n} \right)$ ', fontsize = 15)
    title = axis.set_title('Energy Error', fontsize = 15)

    scaled_energy = {state: np.abs(1 - np.abs((energy / (rydberg / (state.n ** 2))))) for state, energy in energies.items()}
    # axis.set_ylim(np.min(scaled_energy), np.max(scaled_energy))

    for line, state in zip(lines, states):
        line.set_ydata(scaled_energy[state])

    cp.utils.save_current_figure(name = 'energy', target_dir = None, img_format = 'pdf')

    #################

    plt.close()


if __name__ == '__main__':
    # cylindrical_slice_2d_z_rho_points(10, 20, 10, 20, hyd.BoundState(1, 0, 0))
    cylindrical_slice_norm_energy(np.logspace(start = 2, stop = 11, base = 2, num = 5), [hyd.BoundState(n, l, 0) for n in range(1, 3 + 1) for l in range(n)], bound = 30 * bohr_radius)
