import os
import logging
import collections as co
from copy import deepcopy

from tqdm import tqdm

import numpy as np
import scipy.integrate as integ

import compy as cp
import ionization as ion
from compy.units import *

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)

if __name__ == '__main__':
    with log as logger:
        scaling = 50
        electric_field_peak = .1 * np.sqrt(scaling) * 5
        pulse_width = 10 / scaling

        t_initial = -50
        t_final = 50
        n_list = [500, 1000, 2000, 5000, 10000, 20000]
        # n_list = np.logspace(2, 5, num = 50).astype(int)

        t_vs_n = []
        a_vs_n = []
        a_vs_n_downsampled = []

        for n in n_list:
            times = np.linspace(t_initial, t_final, n)
            dt = np.abs(times[1] - times[0])
            print('\n', 'n', n, 'dt', dt)

            # electric_field = electric_field_peak * cp.math.sinc(times / pulse_width)
            # electric_field -= np.mean(electric_field)

            electric_field = electric_field_peak * (1 - np.cos(times + 1e-10)) / (times + 1e-10)

            cp.utils.xy_plot('electric_field_vs_time',
                             times, electric_field,
                             x_label = r'$t$', y_label = '$E(t)$',
                             target_dir = OUT_DIR, )

            delta = 1 / 15
            prefactor = -1e-3

            a = np.zeros(len(times), dtype = np.complex128) * np.NaN
            a[0] = 1

            for current_index, current_time in tqdm(enumerate(times[:-1]), total = len(times[:-1])):
                # print('index', current_index)
                time_difference_argument = delta * (current_time - times[:current_index + 1])
                # print(time_difference_argument)

                current_prefactor = prefactor * electric_field[current_index]

                # integrate through the current time step
                # integral_through_current_step = current_prefactor * dt * np.sum(electric_field[:current_index + 1] * a[:current_index + 1] * np.exp(1j * delta * time_difference_argument) * cp.math.sinc(time_difference_argument))
                integral_through_current_step = current_prefactor * integ.simps(y = electric_field[:current_index + 1] * a[:current_index + 1] * np.exp(1j * delta * time_difference_argument) * cp.math.sinc(time_difference_argument),
                                                                                x = times[:current_index + 1])

                k1 = integral_through_current_step
                a_midpoint_for_k2 = a[current_index] + (dt * k1 / 2)  # dt / 2 here because we moved forward to midpoint

                k2 = integral_through_current_step + (current_prefactor * dt * a_midpoint_for_k2 / 2)  # dt / 2 because it's half of an interval that we're integrating over
                a_midpoint_for_k3 = a[current_index] + (dt * k2 / 2)  # estimate midpoint based on estimate of slope at midpoint

                k3 = integral_through_current_step + (current_prefactor * dt * a_midpoint_for_k3)  # estimate slope based on midpoint again
                a_end_for_k4 = a[current_index] + (dt * k3)  # estimate next point based on estimate of slope at midpoint

                k4 = integral_through_current_step + (current_prefactor * dt * a_end_for_k4)  # estimate slope based on next point

                a[current_index + 1] = a[current_index] + dt * (k1 + (2 * k2) + (2 * k3) + k4) / 6  # estimate next point

                # print('increments', k1, k2, k3, k4)
                # print()

            t_vs_n.append(times)
            a_vs_n.append(np.abs(a) ** 2)
            a_vs_n_downsampled.append(cp.utils.downsample(times, t_vs_n[0], np.abs(a) ** 2))

            cp.utils.xy_plot('a_vs_time__n={}'.format(n),
                             times, np.abs(a) ** 2,
                             x_label = r'$t$', y_label = r'$\left| a_{\alpha}(t) \right|^2$',
                             # y_lower_limit = 0.99, y_upper_limit = 1.005,
                             target_dir = OUT_DIR)

        cp.utils.xy_plot('a_vs_t_vs_n',
                         t_vs_n[0], *a_vs_n_downsampled,
                         line_labels = (str(n) for n in n_list),
                         x_label = r'$t$', y_label = r'$\left| a_{\alpha}(t) \right|^2$',
                         target_dir = OUT_DIR)

        a_vs_n_downsampled_error = [np.abs(a - a_vs_n_downsampled[-1]) for a in a_vs_n_downsampled]  # find difference between most accurate a

        cp.utils.xy_plot('a_vs_t_vs_n__error',
                         t_vs_n[0], *a_vs_n_downsampled_error,
                         line_labels = (str(n) for n in n_list),
                         x_label = r'$t$', y_label = r'$\left| a_{{\alpha}}(t) \right|^2 - \left| a_{{\alpha}}(t) \right|^2 (n={})$'.format(n_list[-1]),
                         y_log_axis = True,
                         target_dir = OUT_DIR)

        a_error_vs_n = [error[-1] for error in a_vs_n_downsampled]

        cp.utils.xy_plot('a_vs_n__error',
                         n_list, a_error_vs_n,
                         x_label = r'$n$', y_label = 'Error at final point',
                         x_log_axis = True, y_log_axis = True,
                         target_dir = OUT_DIR)

        # plt.close()
        #
        # for n, t, a in zip(n_list, t_vs_n, a_vs_n):
        #     plt.plot(t, a, label = str(n))
        #
        # ax = plt.gca()
        # ax.set_xlabel(r'$t$')
        # ax.set_ylabel(r'$\left| a_{\alpha}(t) \right|^2$')
        #
        # plt.legend(loc = 'best')
        #
        # cp.utils.save_current_figure('a_vs_t_vs_n', target_dir = OUT_DIR, img_scale = 1)
