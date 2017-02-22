import datetime as dt
import functools
import logging
import itertools as it
import collections
from copy import copy, deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ

import compy as cp
import compy.cy as cy
from compy.units import *
from . import core, potentials, states

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def return_one(x, **kwargs):
    return 1


def gaussian_kernel(x, *, tau_alpha):
    return (1 + (1j * x / tau_alpha)) ** (-1.5)


class BoundStateIntegroDifferentialEquationSpecification(cp.Specification):
    integration_method = cp.utils.RestrictedValues('integration_method', ('simpson', 'trapezoid'))

    def __init__(self, name,
                 time_initial = 0 * asec, time_final = 200 * asec, time_step = 1 * asec,
                 y_initial = 1,
                 prefactor = 1,
                 f = return_one, f_kwargs = None,
                 kernel = return_one, kernel_kwargs = None,
                 integration_method = 'simpson',
                 **kwargs):
        """
        Initialize an IntegroDifferentialEquationSpecification from the given parameters.

        The differential equation should be of the form
        dy/dt = prefactor * f(t) * integral[ y(t') * f(t') * kernel(t - t')  ; {t', t_initial, t} ]

        :param name:
        :param time_initial:
        :param time_final:
        :param time_step:
        :param y_initial: initial value of y
        :param kwargs:
        """
        super().__init__(name, simulation_type = BoundStateIntegroDifferentialEquationSimulation, **kwargs)

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        self.y_initial = y_initial

        self.prefactor = prefactor

        self.f = f
        self.f_kwargs = dict()
        if f_kwargs is not None:
            self.f_kwargs.update(f_kwargs)

        self.kernel = kernel
        self.kernel_kwargs = dict()
        if kernel_kwargs is not None:
            self.kernel_kwargs.update(kernel_kwargs)

        self.integration_method = integration_method

    def info(self):
        ide_parameters = [
            "IDE Parameters: dy/dt = prefactor * f(t) * integral[ y(t') * f(t') * kernel(t - t')  ; {t', t_initial, t} ]",
            '   Initial State: y = {}'.format(self.y_initial),
            '   prefactor: {}'.format(self.prefactor),
            '   f(t): {} with kwargs {}'.format(self.f.__name__, self.f_kwargs),
            '   kernel(t): {} with kwargs {}'.format(self.kernel.__name__, self.kernel_kwargs),
        ]
        time_evolution = [
            'Time Evolution:',
            '   Initial Time: {} as'.format(uround(self.time_initial, asec)),
            '   Final Time: {} as'.format(uround(self.time_final, asec)),
            '   Time Step: {} as'.format(uround(self.time_step, asec)),
            '   Integration Method: {}'.format(self.integration_method)
        ]

        return '\n'.join(ide_parameters + time_evolution)


class BoundStateIntegroDifferentialEquationSimulation(cp.Simulation):
    def __init__(self, spec):
        super().__init__(spec)

        total_time = self.spec.time_final - self.spec.time_initial
        self.times = np.linspace(self.spec.time_initial, self.spec.time_final, int(total_time / self.spec.time_step) + 1)
        self.time_index = 0
        self.time_steps = len(self.times)

        self.f_eval = self.spec.f(self.times, **self.spec.f_kwargs)

        self.y = np.zeros(self.time_steps, dtype = np.complex128) * np.NaN
        self.y[0] = self.spec.y_initial

    @property
    def time(self):
        return self.times[self.time_index]

    def run_simulation(self):
        logger.info('Performing time evolution on {} ({})'.format(self.name, self.file_name))
        self.status = 'running'

        dydt = 0
        integrand_previous = 0

        while self.time_index < self.time_steps - 1:
            dt = self.time - self.times[self.time_index - 1]
            # print('dt (as)', dt / asec)

            time_difference = self.time - self.times[:self.time_index + 1]  # slice up to current time index
            prefactor = self.spec.prefactor * self.f_eval[self.time_index]

            # print('f at t', self.spec.f(self.time))
            # print('time diffs', time_difference)
            # print('current prefactor', prefactor)

            # integrate through the current time step
            integrand = self.f_eval[:self.time_index + 1] * self.y[:self.time_index + 1] * self.spec.kernel(time_difference, **self.spec.kernel_kwargs)
            if self.spec.integration_method == 'simpson':
                dydt = prefactor * integ.simps(y = integrand,
                                               x = self.times[:self.time_index + 1])
            elif self.spec.integration_method == 'trapezoid' and self.time_index != 0:
                dydt = prefactor * integ.trapz(y = integrand,
                                               x = self.times[:self.time_index + 1])

            # print('integrand', integrand)
            # print('dy/dt', dydt)

            k1 = dydt
            y_midpoint_for_k2 = self.y[self.time_index] + (dt * k1 / 2)  # dt / 2 here because we moved forward to midpoint

            k2 = dydt + (prefactor * dt * y_midpoint_for_k2 / 2)  # dt / 2 because it's half of an interval that we're integrating over
            y_midpoint_for_k3 = self.y[self.time_index] + (dt * k2 / 2)  # estimate midpoint based on estimate of slope at midpoint

            k3 = dydt + (prefactor * dt * y_midpoint_for_k3)  # estimate slope based on midpoint again
            y_end_for_k4 = self.y[self.time_index] + (dt * k3)  # estimate next point based on estimate of slope at midpoint

            k4 = dydt + (prefactor * dt * y_end_for_k4)  # estimate slope based on next point

            # print('dy', dt * (k1 + (2 * k2) + (2 * k3) + k4) / 6)

            self.y[self.time_index + 1] = self.y[self.time_index] + dt * (k1 + (2 * k2) + (2 * k3) + k4) / 6  # estimate next point

            self.time_index += 1

            logger.debug('{} {} ({}) evolved to time index {} / {} ({}%)'.format(self.__class__.__name__, self.name, self.file_name, self.time_index, self.time_steps - 1,
                                                                                 np.around(100 * (self.time_index + 1) / self.time_steps, 2)))

        self.status = 'finished'
        logger.info('Finished performing time evolution on {} {} ({})'.format(self.__class__.__name__, self.name, self.file_name))

    def plot_solution(self, log = False, y_axis_label = None, x_scale = 'asec', f_axis_label = None, f_scale = 1, abs_squared = True, **kwargs):
        fig = cp.utils.get_figure('full')

        x_scale_unit, x_scale_name = unit_value_and_name_from_unit(x_scale)
        f_scale_unit, f_scale_name = unit_value_and_name_from_unit(f_scale)

        grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [5, 1], hspace = 0.07)  # TODO: switch to fixed axis construction
        ax_y = plt.subplot(grid_spec[0])
        ax_f = plt.subplot(grid_spec[1], sharex = ax_y)

        ax_f.plot(self.times / x_scale_unit, self.f_eval / f_scale_unit, color = core.ELECTRIC_FIELD_COLOR, linewidth = 2)
        if abs_squared:
            y = np.abs(self.y) ** 2
        else:
            y = self.y
        ax_y.plot(self.times / x_scale_unit, y, color = 'black', linewidth = 2)

        if log:
            ax_y.set_yscale('log')
            min_overlap = np.min(self.state_overlaps_vs_time[self.spec.initial_state])
            ax_y.set_ylim(bottom = max(1e-9, min_overlap * .1), top = 1.0)
            ax_y.grid(True, which = 'both', **core.GRID_KWARGS)
        else:
            ax_y.set_ylim(0.0, 1.0)
            ax_y.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_y.grid(True, **core.GRID_KWARGS)

        ax_y.set_xlim(self.spec.time_initial / x_scale_unit, self.spec.time_final / x_scale_unit)

        ax_f.set_xlabel('Time $t$ (${}$)'.format(x_scale_name), fontsize = 13)
        if y_axis_label is None:
            y_axis_label = r'$y(t)$'
        ax_y.set_ylabel(y_axis_label, fontsize = 13)
        if f_axis_label is None:
            f_axis_label = r'$f(t)$'.format(f_scale_name)
        ax_f.set_ylabel(r'{} (${}$)'.format(f_axis_label, f_scale_name), fontsize = 13, color = core.ELECTRIC_FIELD_COLOR)

        ax_y.tick_params(labelright = True)
        ax_f.tick_params(labelright = True)
        ax_y.xaxis.tick_top()

        plt.rcParams['xtick.major.pad'] = 5
        plt.rcParams['ytick.major.pad'] = 5

        # Find at most n+1 ticks on the y-axis at 'nice' locations
        max_yticks = 4
        yloc = plt.MaxNLocator(max_yticks, prune = 'upper')
        ax_f.yaxis.set_major_locator(yloc)

        max_xticks = 6
        xloc = plt.MaxNLocator(max_xticks, prune = 'both')
        ax_f.xaxis.set_major_locator(xloc)

        ax_f.tick_params(axis = 'x', which = 'major', labelsize = 10)
        ax_f.tick_params(axis = 'y', which = 'major', labelsize = 10)
        ax_y.tick_params(axis = 'both', which = 'major', labelsize = 10)

        ax_f.grid(True, **core.GRID_KWARGS)

        postfix = ''
        if log:
            postfix += '__log'
        prefix = self.file_name

        name = prefix + '__solution_vs_time{}'.format(postfix)

        cp.utils.save_current_figure(name = name, **kwargs)

        plt.close()
