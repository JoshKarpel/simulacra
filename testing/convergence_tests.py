import os
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

import compy as cp
from compy.units import *
import compy.quantum.hydrogenic as hyd
import compy.quantum.hydrogenic.testing as hydt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


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

    for ii, (z, rho) in enumerate(zip(z_points, rho_points)):
        spec = hyd.CylindricalSliceSpecification('test',
                                                 z_bound = bound, rho_bound = bound,
                                                 z_points = z, rho_points = rho)

        sim = hyd.ElectricFieldSimulation(spec)

        for state in states:
            sim.mesh.g_mesh = sim.mesh.g_for_state(state)

            norms[state][ii] = sim.mesh.norm
            energies[state][ii] = sim.mesh.energy_expectation_value

            print(state, ii, z, rho, norms[state][ii], energies[state][ii] / (rydberg / (state.n ** 2)))

        del spec
        del sim.mesh
        del sim

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
    axis.set_ylim(.5 * min([np.min(norm) for norm in scaled_norm.values()]), 1.5 * max([np.max(norm) for norm in scaled_norm.values()]))

    lines = []

    for state in states:
        lines.append(axis.plot(z_points, scaled_norm[state], label = r'${}$'.format(state.tex_str))[0])

    axis.legend(loc = 'best', fontsize = 12)

    cp.utils.save_current_figure(name = 'cyl_norm_{}br'.format(uround(bound, bohr_radius, 0)), target_dir = OUT_DIR, img_format = 'pdf')

    ###############

    axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \frac{\left< \psi | H | \psi \right>}{E_n} \right)$ ', fontsize = 15)
    title = axis.set_title('Energy Error', fontsize = 15)

    scaled_energy = {state: np.abs(1 - np.abs((energy / (rydberg / (state.n ** 2))))) for state, energy in energies.items()}
    # axis.set_ylim(np.min(scaled_energy), np.max(scaled_energy))
    axis.set_ylim(.5 * min([np.min(energy) for energy in scaled_energy.values()]), 1.5 * max([np.max(energy) for energy in scaled_energy.values()]))

    for line, state in zip(lines, states):
        line.set_ydata(scaled_energy[state])

    cp.utils.save_current_figure(name = 'cyl_energy_{}br'.format(uround(bound, bohr_radius, 0)), target_dir = OUT_DIR, img_format = 'pdf')

    #################

    plt.close()


def spherical_slice_norm_energy(r_points, states, theta_points = 128, bound = 30 * bohr_radius):
    r_points = np.rint(r_points)

    norms = {state: np.zeros(len(r_points)) for state in states}
    energies = {state: np.zeros(len(r_points)) for state in states}

    for ii, r in enumerate(r_points):
        spec = hyd.SphericalSliceSpecification('test',
                                               r_bound = bound,
                                               r_points = r, theta_points = theta_points)

        sim = hyd.ElectricFieldSimulation(spec)

        for state in states:
            sim.mesh.g_mesh = sim.mesh.g_for_state(state)

            norms[state][ii] = sim.mesh.norm
            energies[state][ii] = sim.mesh.energy_expectation_value

            print(state, ii, r, theta_points, norms[state][ii], energies[state][ii] / (rydberg / (state.n ** 2)))

        del spec
        del sim.mesh
        del sim

    fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
    fig.set_tight_layout(True)
    axis = plt.subplot(111)

    axis.set_xlabel(r'$N_r$, $N_{{\theta}} = {}$'.format(theta_points), fontsize = 15)

    axis.set_xscale('log')
    axis.set_yscale('log')

    axis.set_xlim(r_points[0], r_points[-1])

    axis.grid(True, color = 'black', linestyle = ':')  # change grid color to make it show up against the colormesh

    ################

    axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \left< \psi | \psi \right> \right)$', fontsize = 15)
    title = axis.set_title('Norm Error', fontsize = 15)
    title.set_y(1.05)  # move title up a bit

    scaled_norm = {state: np.abs(1 - norm) for state, norm in norms.items()}
    # axis.set_ylim(np.min(scaled_norm), np.max(scaled_norm))
    axis.set_ylim(.5 * min([np.min(norm) for norm in scaled_norm.values()]), 1.5 * max([np.max(norm) for norm in scaled_norm.values()]))

    lines = []

    for state in states:
        lines.append(axis.plot(r_points, scaled_norm[state], label = r'${}$'.format(state.tex_str))[0])

    axis.legend(loc = 'best', fontsize = 12)

    cp.utils.save_current_figure(name = 'sph_norm_{}_{}br'.format(theta_points, uround(bound, bohr_radius, 0)), target_dir = OUT_DIR, img_format = 'pdf')

    ###############

    axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \frac{\left< \psi | H | \psi \right>}{E_n} \right)$ ', fontsize = 15)
    title = axis.set_title('Energy Error', fontsize = 15)

    scaled_energy = {state: np.abs(1 - np.abs((energy / (rydberg / (state.n ** 2))))) for state, energy in energies.items()}
    # axis.set_ylim(np.min(scaled_energy), np.max(scaled_energy))
    axis.set_ylim(.5 * min([np.min(energy) for energy in scaled_energy.values()]), 1.5 * max([np.max(energy) for energy in scaled_energy.values()]))

    for line, state in zip(lines, states):
        line.set_ydata(scaled_energy[state])

    cp.utils.save_current_figure(name = 'sph_energy_{}_{}br'.format(theta_points, uround(bound, bohr_radius, 0)), target_dir = OUT_DIR, img_format = 'pdf')

    #################

    plt.close()


def spherical_harmonic_norm_energy(r_points, states, spherical_harmonics = 128, bound = 30 * bohr_radius):
    r_points = np.rint(r_points)

    norms = {state: np.zeros(len(r_points)) for state in states}
    energies = {state: np.zeros(len(r_points)) for state in states}

    for ii, r in enumerate(r_points):
        spec = hyd.SphericalHarmonicSpecification('test',
                                                  r_bound = bound,
                                                  r_points = r, spherical_harmonics_max_l = spherical_harmonics - 1)

        sim = hyd.ElectricFieldSimulation(spec)

        for state in states:
            sim.mesh.g_mesh = sim.mesh.g_for_state(state)

            norms[state][ii] = sim.mesh.norm
            energies[state][ii] = sim.mesh.energy_expectation_value

            print(state, ii, r, spherical_harmonics, norms[state][ii], energies[state][ii] / (rydberg / (state.n ** 2)))

        del spec
        del sim.mesh
        del sim

    fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
    fig.set_tight_layout(True)
    axis = plt.subplot(111)

    axis.set_xlabel(r'$N_r$, $N_l = {}$'.format(spherical_harmonics), fontsize = 15)

    axis.set_xscale('log')
    axis.set_yscale('log')

    axis.set_xlim(r_points[0], r_points[-1])

    axis.grid(True, color = 'black', linestyle = ':')  # change grid color to make it show up against the colormesh

    ################

    axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \left< \psi | \psi \right> \right)$', fontsize = 15)
    title = axis.set_title('Norm Error', fontsize = 15)
    title.set_y(1.05)  # move title up a bit

    scaled_norm = {state: np.abs(1 - norm) for state, norm in norms.items()}
    # axis.set_ylim(np.min(scaled_norm), np.max(scaled_norm))
    axis.set_ylim(.5 * min([np.min(norm) for norm in scaled_norm.values()]), 1.5 * max([np.max(norm) for norm in scaled_norm.values()]))

    lines = []

    for state in states:
        lines.append(axis.plot(r_points, scaled_norm[state], label = r'${}$'.format(state.tex_str))[0])

    axis.legend(loc = 'best', fontsize = 12)

    cp.utils.save_current_figure(name = 'sphharm__norm_{}_{}br'.format(spherical_harmonics, uround(bound, bohr_radius, 0)), target_dir = OUT_DIR, img_format = 'pdf')

    ###############

    axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \frac{\left< \psi | H | \psi \right>}{E_n} \right)$ ', fontsize = 15)
    title = axis.set_title('Energy Error', fontsize = 15)

    scaled_energy = {state: np.abs(1 - np.abs((energy / (rydberg / (state.n ** 2))))) for state, energy in energies.items()}
    # axis.set_ylim(np.min(scaled_energy), np.max(scaled_energy))
    axis.set_ylim(.5 * min([np.min(energy) for energy in scaled_energy.values()]), 1.5 * max([np.max(energy) for energy in scaled_energy.values()]))

    for line, state in zip(lines, states):
        line.set_ydata(scaled_energy[state])

    cp.utils.save_current_figure(name = 'sphharm_energy_{}_{}br'.format(spherical_harmonics, uround(bound, bohr_radius, 0)), target_dir = OUT_DIR, img_format = 'pdf')

    #################

    plt.close()


def spherical_harmonic_norm_energy_evolved(r_points, states, spherical_harmonics = 128, bound = 30 * bohr_radius, evolve_for = 10000 * asec, evolve_at = 1 * asec):
    r_points = np.rint(r_points)

    norms = {state: np.zeros(len(r_points)) for state in states}
    energies = {state: np.zeros(len(r_points)) for state in states}
    init_overlap = {state: np.zeros(len(r_points)) for state in states}

    for ii, r in enumerate(r_points):
        for state in states:
            spec = hyd.SphericalHarmonicSpecification('test',
                                                      r_bound = bound,
                                                      initial_state = state,
                                                      r_points = r, spherical_harmonics_max_l = spherical_harmonics - 1,
                                                      time_final = evolve_for, time_step = evolve_at)

            sim = hyd.ElectricFieldSimulation(spec)

            sim.run_simulation()

            norms[state][ii] = sim.mesh.norm
            energies[state][ii] = sim.mesh.energy_expectation_value
            init_overlap[state][ii] = sim.mesh.state_overlap(state)

            print(state, ii, r, spherical_harmonics, norms[state][ii], energies[state][ii] / (rydberg / (state.n ** 2)))

            del spec
            del sim.mesh
            del sim

    fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
    fig.set_tight_layout(True)
    axis = plt.subplot(111)

    axis.set_xlabel(r'$N_r$, $N_l = {}$'.format(spherical_harmonics), fontsize = 15)

    axis.set_xscale('log')
    axis.set_yscale('log')

    axis.set_xlim(r_points[0], r_points[-1])

    axis.grid(True, color = 'black', linestyle = ':')  # change grid color to make it show up against the colormesh

    ################

    axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \left< \psi | \psi \right> \right)$', fontsize = 15)
    title = axis.set_title('Norm Error', fontsize = 15)
    title.set_y(1.05)  # move title up a bit

    scaled_norm = {state: np.abs(1 - norm) for state, norm in norms.items()}
    # axis.set_ylim(np.min(scaled_norm), np.max(scaled_norm))
    axis.set_ylim(.5 * min([np.min(norm) for norm in scaled_norm.values()]), 1.5 * max([np.max(norm) for norm in scaled_norm.values()]))

    lines = []

    for state in states:
        lines.append(axis.plot(r_points, scaled_norm[state], label = r'${}$'.format(state.tex_str))[0])

    axis.legend(loc = 'best', fontsize = 12)

    cp.utils.save_current_figure(name = 'sphharm__norm_{}_{}brevolvedFor{}asec'.format(spherical_harmonics, uround(bound, bohr_radius, 0), uround(evolve_for, asec, 0)), target_dir = OUT_DIR, img_format = 'pdf')

    ###############

    axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \frac{\left< \psi | H | \psi \right>}{E_n} \right)$ ', fontsize = 15)
    title = axis.set_title('Energy Error', fontsize = 15)

    scaled_energy = {state: np.abs(1 - np.abs((energy / (rydberg / (state.n ** 2))))) for state, energy in energies.items()}
    # axis.set_ylim(np.min(scaled_energy), np.max(scaled_energy))
    axis.set_ylim(.5 * min([np.min(energy) for energy in scaled_energy.values()]), 1.5 * max([np.max(energy) for energy in scaled_energy.values()]))

    for line, state in zip(lines, states):
        line.set_ydata(scaled_energy[state])

    cp.utils.save_current_figure(name = 'sphharm_energy_{}_{}br_evolvedFor{}asec'.format(spherical_harmonics, uround(bound, bohr_radius, 0), uround(evolve_for, asec, 0)), target_dir = OUT_DIR, img_format = 'pdf')

    #################

    ###############

    axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \left< \psi | \psi(t=0) \right> \right)$ ', fontsize = 15)
    title = axis.set_title('Initial State Overlap Error', fontsize = 15)

    scaled_overlap = {state: np.abs(1 - overlap) for state, overlap in init_overlap.items()}
    # axis.set_ylim(np.min(scaled_energy), np.max(scaled_energy))
    axis.set_ylim(.5 * min([np.min(overlap) for overlap in scaled_overlap.values()]), 1.5 * max([np.max(overlap) for overlap in scaled_overlap.values()]))

    for line, state in zip(lines, states):
        line.set_ydata(scaled_overlap[state])

    cp.utils.save_current_figure(name = 'sphharm_initOverlap_{}_{}br_evolvedFor{}asec'.format(spherical_harmonics, uround(bound, bohr_radius, 0), uround(evolve_for, asec, 0)), target_dir = OUT_DIR, img_format = 'pdf')

    #################

    plt.close()


def spherical_harmonic_time_stability(r_point_count, states, spherical_harmonics = 128, bound = 30 * bohr_radius, evolve_for = 10000 * asec, evolve_at = 1 * asec):
    norms = {state: None for state in states}
    energies = {state: None for state in states}
    init_overlap = {state: None for state in states}

    for state in states:
        spec = hyd.SphericalHarmonicSpecification('test',
                                                  r_bound = bound,
                                                  initial_state = state,
                                                  r_points = r_point_count, spherical_harmonics_max_l = spherical_harmonics - 1,
                                                  time_final = evolve_for, time_step = evolve_at)

        sim = hyd.ElectricFieldSimulation(spec)

        times = sim.times.copy()

        sim.run_simulation()

        norms[state] = sim.norm_vs_time
        # energies[state] = sim.mesh.energy_expectation_value
        init_overlap[state] = sim.state_overlaps_vs_time[state]

        # print(state, ii, r, spherical_harmonics, norms[state][ii], energies[state][ii] / (rydberg / (state.n ** 2)))

        del spec
        del sim.mesh
        del sim

    fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
    fig.set_tight_layout(True)
    axis = plt.subplot(111)

    axis.set_xlabel(r'$t$ (as)'.format(spherical_harmonics), fontsize = 15)

    axis.set_yscale('log')

    axis.set_xlim(times[0] / asec, times[-1] / asec)

    axis.grid(True, color = 'black', linestyle = ':')  # change grid color to make it show up against the colormesh

    ################

    axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \left< \psi | \psi \right> \right)$', fontsize = 15)
    title = axis.set_title('Norm Error', fontsize = 15)
    title.set_y(1.05)  # move title up a bit

    scaled_norm = {state: np.abs(1 - norm) for state, norm in norms.items()}
    # axis.set_ylim(np.min(scaled_norm), np.max(scaled_norm))
    axis.set_ylim(.5 * min([np.min(norm) for norm in scaled_norm.values()]), 1.5 * max([np.max(norm) for norm in scaled_norm.values()]))

    lines = []

    for state in states:
        lines.append(axis.plot(times / asec, scaled_norm[state], label = r'${}$'.format(state.tex_str))[0])

    axis.legend(loc = 'best', fontsize = 12)

    cp.utils.save_current_figure(name = 'sphharm__norm_{}_{}brevolvedFor{}asec_overtime'.format(spherical_harmonics, uround(bound, bohr_radius, 0), uround(evolve_for, asec, 0)), target_dir = OUT_DIR, img_format = 'pdf')

    ###############

    # axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \frac{\left< \psi | H | \psi \right>}{E_n} \right)$ ', fontsize = 15)
    # title = axis.set_title('Energy Error', fontsize = 15)
    #
    # scaled_energy = {state: np.abs(1 - np.abs((energy / (rydberg / (state.n ** 2))))) for state, energy in energies.items()}
    # # axis.set_ylim(np.min(scaled_energy), np.max(scaled_energy))
    # axis.set_ylim(.5 * min([np.min(energy) for energy in scaled_energy.values()]), 1.5 * max([np.max(energy) for energy in scaled_energy.values()]))
    #
    # for line, state in zip(lines, states):
    #     line.set_ydata(scaled_energy[state])
    #
    # cp.utils.save_current_figure(name = 'sphharm_energy_{}_{}br_evolvedFor{}asec'.format(spherical_harmonics, uround(bound, bohr_radius, 0), uround(evolve_for, asec, 0)), target_dir = OUT_DIR, img_format = 'pdf')

    #################

    ###############

    axis.set_ylabel(r'$\mathrm{Log}_{10} \left( 1 - \left< \psi | \psi(t=0) \right> \right)$ ', fontsize = 15)
    title = axis.set_title('Initial State Overlap Error', fontsize = 15)

    scaled_overlap = {state: np.abs(1 - overlap) for state, overlap in init_overlap.items()}
    # axis.set_ylim(np.min(scaled_energy), np.max(scaled_energy))
    axis.set_ylim(.5 * min([np.min(overlap) for overlap in scaled_overlap.values()]), 1.5 * max([np.max(overlap) for overlap in scaled_overlap.values()]))

    for line, state in zip(lines, states):
        line.set_ydata(scaled_overlap[state])

    cp.utils.save_current_figure(name = 'sphharm_initOverlap_{}_{}br_evolvedFor{}asec_overtime'.format(spherical_harmonics, uround(bound, bohr_radius, 0), uround(evolve_for, asec, 0)), target_dir = OUT_DIR, img_format = 'pdf')

    #################

    plt.close()





if __name__ == '__main__':
    # cylindrical_slice_2d_z_rho_points(10, 20, 10, 20, hyd.BoundState(1, 0, 0))
    # z = [100, 101, 102, 103 200, 201, 202, 203, 400, 401, 402, 403, 600, 601, 602, 603, 800, 801, 802, 803, 1000, 1001, 1002, 1003]
    # nn = np.linspace(100, 1000, num = 200)
    linear_points = np.logspace(start = 7, stop = 11, base = 2, num = 100)
    radial_points = np.logspace(start = 7, stop = 10, base = 2, num = 100)
    angular_points = 2 ** 7
    n_max = 3
    states = [hyd.BoundState(n, l, 0) for n in range(1, n_max + 1) for l in range(n)]
    bound = 40 * bohr_radius

    linear_points = 2 ** np.array([6, 7, 8, 9, 10, 11])

    with cp.utils.Logger() as logger:
        # cylindrical_slice_norm_energy(linear_points, states, bound = bound)
        # spherical_slice_norm_energy(radial_points, states, bound = bound, theta_points = angular_points)
        # spherical_harmonic_norm_energy(radial_points, states, bound = bound, spherical_harmonics = angular_points)

        # spherical_harmonic_norm_energy_evolved(radial_points, states, spherical_harmonics = angular_points, bound = bound,
        #                                        evolve_for = 1000 * asec, evolve_at = 10 * asec)

        # spherical_harmonic_time_stability(1000, states, spherical_harmonics = angular_points, bound = bound,
        #                                   evolve_for = 1000 * asec, evolve_at = 1 * asec)

        for zz in linear_points:
            for state in states:
                spec = hyd.CylindricalSliceSpecification('{}_{}__{}'.format(state.n, state.l, zz),
                                                         z_points = zz, rho_points = zz / 2,
                                                         time_final = 10000 * asec, time_step = 1 * asec)
                sim = hydt.StaticConvergenceTestingSimulation(spec)

                sim.run_simulation()

                sim.plot_error_vs_time(save = True, target_dir = OUT_DIR)
