import logging
import os

import numpy as np
import scipy.interpolate as interp

import compy as cp
import compy.cy as cy
import ionization as ion
from ionization import integrodiff as ide
from compy.units import *

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


def comparison_plot(dt_list, t_by_dt, y_by_dt, title):
    fig = cp.utils.get_figure('full')
    ax = fig.add_subplot(111)

    for dt, t, y in zip(dt_list, t_by_dt, y_by_dt):
        ax.plot(t / asec, np.abs(y) ** 2,
                label = r'$\Delta t = {} \, \mathrm{{as}}$'.format(round(dt, 3)),
                linewidth = .2,
                )

    ax.legend(loc = 'best')
    ax.set_xlabel(r'Time $t$ ($\mathrm{as}$)')
    ax.set_ylabel(r'$   \left| a_{\alpha}(t) \right|^2   $')
    ax.grid(True, **ion.GRID_KWARGS)

    cp.utils.save_current_figure('{}__comparison'.format(title), target_dir = OUT_DIR)

    plt.close()


def error_plot(dt_list, t_by_dt, y_by_dt, title):
    dt_min_index = np.argmin(dt_list)
    longest_t = t_by_dt[dt_min_index]
    best_y = y_by_dt[dt_min_index]

    fig = cp.utils.get_figure('full')
    ax = fig.add_subplot(111)

    for dt, t, y in zip(dt_list, t_by_dt, y_by_dt):
        terp = interp.interp1d(t, y)

        plot_y = terp(longest_t)
        diff = np.abs(plot_y) ** 2 - np.abs(best_y ** 2)

        ax.plot(longest_t / asec, diff,
                label = r'$\Delta t = {} \, \mathrm{{as}}$'.format(round(dt, 3)),
                linewidth = .2,
                )

    ax.legend(loc = 'best')
    ax.set_xlabel(r'Time $t$ ($\mathrm{as}$)')
    ax.set_ylabel(r'$   \left| a_{\alpha}(t) \right|^2 - \left| a_{\alpha}^{\mathrm{best}}(t) \right|^2  $')
    ax.grid(True, **ion.GRID_KWARGS)

    cp.utils.save_current_figure('{}__error'.format(title), target_dir = OUT_DIR)

    plt.close()


def error_log_plot(dt_list, t_by_dt, y_by_dt, title):
    dt_min_index = np.argmin(dt_list)
    longest_t = t_by_dt[dt_min_index]
    best_y = y_by_dt[dt_min_index]

    fig = cp.utils.get_figure('full')
    ax = fig.add_subplot(111)

    for dt, t, y in zip(dt_list, t_by_dt, y_by_dt):
        terp = interp.interp1d(t, y)

        plot_y = terp(longest_t)
        diff = 1 - (np.abs(plot_y) ** 2 / np.abs(best_y ** 2))

        ax.plot(longest_t / asec, diff,
                label = r'$\Delta t = {} \, \mathrm{{as}}$'.format(round(dt, 3)),
                linewidth = 1,
                )

    ax.legend(loc = 'best')
    ax.set_xlabel(r'Time $t$ ($\mathrm{as}$)')
    ax.set_ylabel(r'$  1 -  \left| a_{\alpha}(t) \right|^2 / \left| a_{\alpha}^{\mathrm{best}}(t) \right|^2 $')

    ax.grid(True, which = 'major', **ion.GRID_KWARGS)
    ax.grid(True, which = 'minor', **ion.GRID_KWARGS)

    ax.set_yscale('log')
    # ax.set_ylim(bottom = 1e-10, top = 1)

    cp.utils.save_current_figure('{}__error_log'.format(title), target_dir = OUT_DIR)

    plt.close()


def convergence_plot(dt_list, t_by_dt, y_by_dt, title):
    dt_min_index = np.argmin(dt_list)
    longest_t = t_by_dt[dt_min_index]
    best_y = y_by_dt[dt_min_index]

    fig = cp.utils.get_figure('full')
    ax = fig.add_subplot(111)

    final = [np.abs(np.abs(y[-1]) - np.abs(best_y[-1])) for y in y_by_dt]

    ax.plot(dt_list[:-1], final[:-1])

    ax.set_xlabel(r'Time Step $\Delta t$ ($\mathrm{as}$)')
    ax.set_ylabel(r'$   \left| \left| a_{\alpha}(t_{\mathrm{final}}) \right| - \left| a_{\alpha}^{\mathrm{best}}(t_{\mathrm{final}}) \right| \right|  $')

    ax.grid(True, which = 'major', **ion.GRID_KWARGS)
    ax.grid(True, which = 'minor', **ion.GRID_KWARGS)

    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim(bottom = .01 * np.nanmin(final), top = 1)

    cp.utils.save_current_figure('{}__convergence'.format(title), target_dir = OUT_DIR)

    plt.close()


def convergence_plot_squared(dt_list, t_by_dt, y_by_dt, title):
    dt_min_index = np.argmin(dt_list)
    longest_t = t_by_dt[dt_min_index]
    best_y = y_by_dt[dt_min_index]

    fig = cp.utils.get_figure('full')
    ax = fig.add_subplot(111)

    final = [np.abs(np.abs(y[-1]) ** 2 - np.abs(best_y[-1]) ** 2) for y in y_by_dt]

    ax.plot(dt_list[:-1], final[:-1])

    ax.set_xlabel(r'Time Step $\Delta t$ ($\mathrm{as}$)')
    ax.set_ylabel(r'$   \left| \left| a_{\alpha}(t_{\mathrm{final}}) \right|^2 - \left| a_{\alpha}^{\mathrm{best}}(t_{\mathrm{final}}) \right|^2 \right|  $')
    ax.grid(True, which = 'major', **ion.GRID_KWARGS)
    ax.grid(True, which = 'minor', **ion.GRID_KWARGS)

    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim(bottom = .01 * np.nanmin(final), top = 1)

    cp.utils.save_current_figure('{}__convergence_squared'.format(title), target_dir = OUT_DIR)

    plt.close()


if __name__ == '__main__':
    # electric_field = ion.Rectangle(start_time = -500 * asec, end_time = 500 * asec, amplitude = 1 * atomic_electric_field)

    t_bound_per_pw = 5
    pw = 50

    electric_field = ion.SincPulse(pulse_width = pw * asec, fluence = 1 * Jcm2,
                                   window = ion.RectangularTimeWindow(on_time = -(t_bound_per_pw - 1) * pw * asec,
                                                                      off_time = (t_bound_per_pw - 1) * pw * asec))

    t_bound = pw * t_bound_per_pw

    q = electron_charge
    m = electron_mass_reduced
    L = bohr_radius

    tau_alpha = 4 * m * (L ** 2) / hbar
    prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

    # dt_list = np.array([50, 25, 10, 5, 2, 1, .5, .1])
    # dt_list = np.array([10, 5, 2, 1, .5, .1])
    dt_list = np.logspace(1, -1.5, 10)

    t_by_dt = []
    y_by_dt = []

    # method = 'trapezoid'
    method = 'simpson'

    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO,
                         file_logs = True, file_dir = OUT_DIR, file_name = method, file_mode = 'w', file_level = logging.INFO) as logger:
        for dt in dt_list:
            spec = ide.BoundStateIntegroDifferentialEquationSpecification('{}__{}__dt={}as'.format(method, electric_field.__class__.__name__, round(dt, 3)),
                                                                          time_initial = -t_bound * asec, time_final = t_bound * asec, time_step = dt * asec,
                                                                          prefactor = prefactor,
                                                                          f = electric_field.get_electric_field_amplitude,
                                                                          kernel = ide.gaussian_kernel, kernel_kwargs = dict(tau_alpha = tau_alpha),
                                                                          integration_method = method,
                                                                          )

            sim = spec.to_simulation()

            sim.run_simulation()

            sim.plot_solution(target_dir = OUT_DIR,
                              y_axis_label = r'$   \left| a_{\alpha}(t) \right|^2  $',
                              f_axis_label = r'${}(t)$'.format(str_efield),
                              f_scale = 'AEF')

            t_by_dt.append(sim.times)
            y_by_dt.append(sim.y)

            logger.info(sim.info())

        title = '{}__{}'.format(method, electric_field.__class__.__name__)

        comparison_plot(dt_list, t_by_dt, y_by_dt, title)
        error_plot(dt_list, t_by_dt, y_by_dt, title)
        error_log_plot(dt_list, t_by_dt, y_by_dt, title)
        convergence_plot(dt_list, t_by_dt, y_by_dt, title)
        convergence_plot_squared(dt_list, t_by_dt, y_by_dt, title)
