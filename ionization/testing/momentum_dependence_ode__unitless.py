import os
import logging
import collections as co
from copy import deepcopy

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
        n_list = [100, 250, 500, 1000, 2000, 5000, 10000]

        t_vs_n = []
        a_vs_n = []

        for n in n_list:
            times = np.linspace(t_initial, t_final, n)
            dt = np.abs(times[1] - times[0])
            print('n', n, 'dt', dt)

            electric_field = electric_field_peak * cp.math.sinc(times / pulse_width)
            electric_field -= np.mean(electric_field)

            cp.utils.xy_plot('electric_field_vs_time',
                             times, electric_field,
                             x_label = r'$t$', y_label = '$E(t)$',
                             target_dir = OUT_DIR, )

            delta = 1 / 15
            prefactor = -1e-3

            a = np.zeros(len(times), dtype = np.complex128) * np.NaN
            a[0] = 1

            for current_index, current_time in enumerate(times[:-1]):
                print('index', current_index)
                time_difference_argument = delta * (current_time - times[:current_index + 1])
                # print(time_difference_argument)

                current_prefactor = prefactor * electric_field[current_index]

                # integral_through_current_step = current_prefactor * dt * np.sum(electric_field[:current_index + 1] * a[:current_index + 1] * np.exp(1j * delta * time_difference_argument) * cp.math.sinc(time_difference_argument))
                integral_through_current_step = current_prefactor * integ.simps(y = electric_field[:current_index + 1] * a[:current_index + 1] * np.exp(1j * delta * time_difference_argument) * cp.math.sinc(time_difference_argument),
                                                                                x = times[:current_index + 1])

                k1 = integral_through_current_step
                a_midpoint_for_k2 = a[current_index] + (dt * k1 / 2)  # half here because we moved forward to midpoint

                k2 = integral_through_current_step + (current_prefactor * dt * a_midpoint_for_k2 / 2)  # dt / 2 because it's half of an interval that we're integrating over
                a_midpoint_for_k3 = a[current_index] + (dt * k2 / 2)

                k3 = integral_through_current_step + (current_prefactor * dt * a_midpoint_for_k3)
                a_end_for_k4 = a[current_index] + (dt * k3)

                k4 = integral_through_current_step + (current_prefactor * dt * a_end_for_k4)

                a[current_index + 1] = a[current_index] + dt * (k1 + (2 * k2) + (2 * k3) + k4) / 6

                # print('increments', k1, k2, k3, k4)
                print()

            t_vs_n.append(times)
            a_vs_n.append(np.abs(a) ** 2)

            cp.utils.xy_plot('a_vs_time__n={}'.format(n),
                             times, np.abs(a) ** 2,
                             x_label = r'$t$', y_label = r'$\left| a_{\alpha}(t) \right|^2$',
                             y_lower_limit = 0.99, y_upper_limit = 1.005,
                             target_dir = OUT_DIR)

        plt.close()

        for n, t, a in zip(n_list, t_vs_n, a_vs_n):
            plt.plot(t, a, label = str(n))

        ax = plt.gca()
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$\left| a_{\alpha}(t) \right|^2$')

        plt.legend(loc = 'best')

        cp.utils.save_current_figure('a_vs_t_vs_n', target_dir = OUT_DIR, img_scale = 1)
