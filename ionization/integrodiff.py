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
                 evolution_method = 'FE',
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
        self.evolution_method = evolution_method

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
            '   Integration Method: {}'.format(self.integration_method),
            '   Evolution Method: {}'.format(self.evolution_method),
        ]

        return '\n'.join(ide_parameters + time_evolution)


class BoundStateIntegroDifferentialEquationSimulation(cp.Simulation):
    def __init__(self, spec):
        super().__init__(spec)

        if self.spec.evolution_method == 'ARK4':
            self.times = np.array([self.spec.time_initial])
            self.time_index = 0
            self.time_steps = np.NaN

            self.time_step = self.spec.time_step

            self.y = np.array([self.spec.y_initial])
        else:
            total_time = self.spec.time_final - self.spec.time_initial
            self.times = np.linspace(self.spec.time_initial, self.spec.time_final, int(total_time / self.spec.time_step) + 1)
            self.time_index = 0
            self.time_steps = len(self.times)

            self.f_eval = self.spec.f(self.times, **self.spec.f_kwargs)

            self.y = np.zeros(self.time_steps, dtype = np.complex128) * np.NaN
            self.y[0] = self.spec.y_initial

        if self.spec.integration_method == 'simpson':
            self.integrate = integ.simps
        elif self.spec.integration_method == 'trapezoid':
            self.integrate = integ.trapz

    @property
    def time(self):
        return self.times[self.time_index]

    def evolve_FE(self):
        dt = self.times[self.time_index + 1] - self.time

        k = self.spec.prefactor * self.f_eval[self.time_index] * self.integrate(y = self.f_eval[:self.time_index + 1] * self.y[:self.time_index + 1] * self.spec.kernel(self.time - self.times[:self.time_index + 1], **self.spec.kernel_kwargs),
                                                                                x = self.times[:self.time_index + 1])
        self.y[self.time_index + 1] = self.y[self.time_index] + (dt * k)  # estimate next point

    def evolve_BE(self):
        dt = self.times[self.time_index + 1] - self.time

        k = self.spec.prefactor * self.f_eval[self.time_index + 1] * self.integrate(y = self.f_eval[:self.time_index + 1] * self.y[:self.time_index + 1] * self.spec.kernel(self.times[self.time_index + 1] - self.times[:self.time_index + 1], **self.spec.kernel_kwargs),
                                                                                    x = self.times[:self.time_index + 1])

        self.y[self.time_index + 1] = (self.y[self.time_index] + (dt * k)) / (1 - self.spec.prefactor * ((dt * self.f_eval[self.time_index + 1]) ** 2))  # estimate next point

    def evolve_TRAP(self):
        dt = self.times[self.time_index + 1] - self.time

        k_1 = self.spec.prefactor * self.f_eval[self.time_index + 1] * self.integrate(y = self.f_eval[:self.time_index + 1] * self.y[:self.time_index + 1] * self.spec.kernel(self.times[self.time_index + 1] - self.times[:self.time_index + 1], **self.spec.kernel_kwargs),
                                                                                      x = self.times[:self.time_index + 1])

        k_2 = self.spec.prefactor * self.f_eval[self.time_index] * self.integrate(y = self.f_eval[:self.time_index + 1] * self.y[:self.time_index + 1] * self.spec.kernel(self.times[self.time_index] - self.times[:self.time_index + 1], **self.spec.kernel_kwargs),
                                                                                  x = self.times[:self.time_index + 1])

        self.y[self.time_index + 1] = (self.y[self.time_index] + (dt * (k_1 + k_2) / 2)) / (1 - .5 * self.spec.prefactor * ((dt * self.f_eval[self.time_index + 1]) ** 2))  # estimate next point

    def evolve_RK4(self):
        dt = self.times[self.time_index + 1] - self.time
        # print('dt (as)', dt / asec)

        times_curr = self.times[:self.time_index + 1]
        times_half = np.append(self.times[:self.time_index + 1], self.time + dt / 2)
        times_next = self.times[:self.time_index + 2]

        time_difference_curr = self.time - times_curr  # slice up to current time index
        time_difference_half = (self.time + dt / 2) - times_half
        time_difference_next = self.times[self.time_index + 1] - times_next

        kernel_curr = self.spec.kernel(time_difference_curr, **self.spec.kernel_kwargs)
        kernel_half = self.spec.kernel(time_difference_half, **self.spec.kernel_kwargs)
        kernel_next = self.spec.kernel(time_difference_next, **self.spec.kernel_kwargs)

        f_curr = self.f_eval[self.time_index]
        f_half = self.spec.f(self.time + (dt / 2))
        f_next = self.f_eval[self.time_index + 1]

        f_times_y_curr = self.f_eval[:self.time_index + 1] * self.y[:self.time_index + 1]

        # integrate through the current time step
        integrand_for_k1 = f_times_y_curr * kernel_curr
        integral_for_k1 = self.integrate(y = integrand_for_k1, x = times_curr)
        k1 = self.spec.prefactor * f_curr * integral_for_k1
        y_midpoint_for_k2 = self.y[self.time_index] + (dt * k1 / 2)  # dt / 2 here because we moved forward to midpoint

        integrand_for_k2 = np.append(f_times_y_curr, f_half * y_midpoint_for_k2) * kernel_half
        integral_for_k2 = self.integrate(y = integrand_for_k2, x = times_half)
        k2 = self.spec.prefactor * f_half * integral_for_k2  # dt / 2 because it's half of an interval that we're integrating over
        y_midpoint_for_k3 = self.y[self.time_index] + (dt * k2 / 2)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k3 = np.append(f_times_y_curr, f_half * y_midpoint_for_k3) * kernel_half
        integral_for_k3 = self.integrate(y = integrand_for_k3, x = times_half)
        k3 = self.spec.prefactor * f_half * integral_for_k3  # dt / 2 because it's half of an interval that we're integrating over
        y_end_for_k4 = self.y[self.time_index] + (dt * k3)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k4 = np.append(f_times_y_curr, f_next * y_end_for_k4) * kernel_next
        integral_for_k4 = self.integrate(y = integrand_for_k4, x = times_next)
        k4 = self.spec.prefactor * f_next * integral_for_k4

        # print(k1, k2, k3, k4)
        # print(integral_through_current_step, (prefactor * dt * y_midpoint_for_k2 / 2))
        # print('dy', dt * (k1 + (2 * k2) + (2 * k3) + k4) / 6)

        # print()
        # print('1', k1, np.abs(k1 / k1))
        # print('2', k2, np.abs(k2 / k1))
        # print('3', k3, np.abs(k3 / k1))
        # print('4', k4, np.abs(k4 / k1))
        # print('avg', ((k1 + (2 * k2) + (2 * k3) + k4) / 6), np.abs(((k1 + (2 * k2) + (2 * k3) + k4) / 6) / k1))
        # print()
        #
        self.y[self.time_index + 1] = self.y[self.time_index] + (dt * (k1 + (2 * k2) + (2 * k3) + k4) / 6)  # estimate next point

    def evolve_ARK4(self):
        times_curr = self.times
        times_quarter = np.append(self.times[:self.time_index + 1], self.time + self.time_step / 4)
        times_half = np.append(self.times[:self.time_index + 1], self.time + self.time_step / 2)
        times_three_quarter = np.append(self.times[:self.time_index + 1], self.time + 3 * self.time_step / 4)
        times_next = np.append(self.times[:self.time_index + 1], self.time + self.time_step)

        time_difference_curr = self.time - times_curr  # slice up to current time index
        time_difference_quarter = (self.time + self.time_step / 4) - times_quarter
        time_difference_half = (self.time + self.time_step / 2) - times_half
        time_difference_three_quarter = (self.time + 3 * self.time_step / 4) - times_three_quarter
        time_difference_next = self.time + self.time_step - times_next

        kernel_curr = self.spec.kernel(time_difference_curr, **self.spec.kernel_kwargs)
        kernel_quarter = self.spec.kernel(time_difference_quarter, **self.spec.kernel_kwargs)
        kernel_half = self.spec.kernel(time_difference_half, **self.spec.kernel_kwargs)
        kernel_three_quarter = self.spec.kernel(time_difference_three_quarter, **self.spec.kernel_kwargs)
        kernel_next = self.spec.kernel(time_difference_next, **self.spec.kernel_kwargs)

        f_curr = self.spec.f(self.time)
        f_quarter = self.spec.f(self.time + (self.time_step / 4))
        f_half = self.spec.f(self.time + (self.time_step / 2))
        f_three_quarter = self.spec.f(self.time + (3 * self.time_step / 4))
        f_next = self.spec.f(self.time + self.time_step)

        f_times_y_curr = self.spec.f(self.times, **self.spec.f_kwargs) * self.y

        # CALCULATE FULL STEP ESTIMATE
        integrand_for_k1 = f_times_y_curr * kernel_curr
        integral_for_k1 = self.integrate(y = integrand_for_k1, x = times_curr)
        k1 = self.spec.prefactor * f_curr * integral_for_k1
        y_midpoint_for_k2 = self.y[self.time_index] + (self.time_step * k1 / 2)  # self.time_step / 2 here because we moved forward to midpoint

        integrand_for_k2 = np.append(f_times_y_curr, f_half * y_midpoint_for_k2) * kernel_half
        integral_for_k2 = self.integrate(y = integrand_for_k2, x = times_half)
        k2 = self.spec.prefactor * f_half * integral_for_k2  # self.time_step / 2 because it's half of an interval that we're integrating over
        y_midpoint_for_k3 = self.y[self.time_index] + (self.time_step * k2 / 2)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k3 = np.append(f_times_y_curr, f_half * y_midpoint_for_k3) * kernel_half
        integral_for_k3 = self.integrate(y = integrand_for_k3, x = times_half)
        k3 = self.spec.prefactor * f_half * integral_for_k3  # self.time_step / 2 because it's half of an interval that we're integrating over
        y_end_for_k4 = self.y[self.time_index] + (self.time_step * k3)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k4 = np.append(f_times_y_curr, f_next * y_end_for_k4) * kernel_next
        integral_for_k4 = self.integrate(y = integrand_for_k4, x = times_next)
        k4 = self.spec.prefactor * f_next * integral_for_k4

        full_step_estimate = self.y[self.time_index] + (self.time_step * (k1 + (2 * k2) + (2 * k3) + k4) / 6)

        # CALCULATE DOUBLE HALF STEP ESTIMATE

        # k1 is identical from above
        # integrand_for_k1 = f_times_y_curr * kernel_curr
        # integral_for_k1 = self.integrate(y = integrand_for_k1, x = times_curr)
        # k1 = self.spec.prefactor * f_curr * integral_for_k1
        y_midpoint_for_k2 = self.y[self.time_index] + (self.time_step * k1 / 4)  # self.time_step / 4 here because we moved forward to midpoint of midpoint

        integrand_for_k2 = np.append(f_times_y_curr, f_quarter * y_midpoint_for_k2) * kernel_quarter
        integral_for_k2 = self.integrate(y = integrand_for_k2, x = times_quarter)
        k2 = self.spec.prefactor * f_half * integral_for_k2  # self.time_step / 2 because it's half of an interval that we're integrating over
        y_midpoint_for_k3 = self.y[self.time_index] + (self.time_step * k2 / 4)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k3 = np.append(f_times_y_curr, f_quarter * y_midpoint_for_k3) * kernel_quarter
        integral_for_k3 = self.integrate(y = integrand_for_k3, x = times_quarter)
        k3 = self.spec.prefactor * f_half * integral_for_k3  # self.time_step / 2 because it's half of an interval that we're integrating over
        y_end_for_k4 = self.y[self.time_index] + (self.time_step * k2 / 2)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k4 = np.append(f_times_y_curr, f_half * y_end_for_k4) * kernel_half
        integral_for_k4 = self.integrate(y = integrand_for_k4, x = times_half)
        k4 = self.spec.prefactor * f_half * integral_for_k4

        y_half = self.y[self.time_index] + (self.time_step * (k1 + (2 * k2) + (2 * k3) + k4) / 6)
        f_times_y_half = np.append(f_times_y_curr * kernel_curr, f_half * y_half)

        integrand_for_k1 = f_times_y_half * kernel_half
        integral_for_k1 = self.integrate(y = integrand_for_k1, x = times_half)
        k1 = self.spec.prefactor * f_half * integral_for_k1
        y_midpoint_for_k2 = y_half + (self.time_step * k1 / 4)  # self.time_step / 4 here because we moved forward to midpoint of midpoint

        # THE PROBLEM IS THAT KERNEL_THREE_QUARTERS DOESNT INCLUDE THE HALFWAY POINT
        # OH MY GOD THIS IS AWFUL
        # MAYBE I SHOULD JUST DO THE BETTER METHOD THEY DESCRIBE

        integrand_for_k2 = np.append(f_times_y_half, f_three_quarter * y_midpoint_for_k2) * kernel_three_quarter
        integral_for_k2 = self.integrate(y = integrand_for_k2, x = times_three_quarter)
        k2 = self.spec.prefactor * f_three_quarter * integral_for_k2  # self.time_step / 2 because it's half of an interval that we're integrating over
        y_midpoint_for_k3 = self.y[self.time_index] + (self.time_step * k2 / 4)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k3 = np.append(f_times_y_half, f_three_quarter * y_midpoint_for_k3) * kernel_three_quarter
        integral_for_k3 = self.integrate(y = integrand_for_k3, x = times_three_quarter)
        k3 = self.spec.prefactor * f_three_quarter * integral_for_k3  # self.time_step / 2 because it's half of an interval that we're integrating over
        y_end_for_k4 = self.y[self.time_index] + (self.time_step * k2 / 2)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k4 = np.append(f_times_y_half, f_next * y_end_for_k4) * kernel_next
        integral_for_k4 = self.integrate(y = integrand_for_k4, x = times_next)
        k4 = self.spec.prefactor * f_next * integral_for_k4

        double_half_step_estimate = self.y[self.time_index] + (self.time_step * (k1 + (2 * k2) + (2 * k3) + k4) / 6)

        ##########################

        delta_1 = double_half_step_estimate - full_step_estimate

        delta_0 = 1e-3 * self.time_step * double_half_step_estimate

        s = 0.95

        if delta_0 >= delta_1:  # step was ok
            self.time_step = s * self.time_step * np.abs(delta_0 / delta_1) ** (1 / 5)
            logger.debug('Accepted RK4 step. Increased time step to {}'.format(self.time_step))
            np.append(self.y, double_half_step_estimate + (delta_1 / 15))
        else:  # reject step
            self.time_step = s * self.time_step * np.abs(delta_0 / delta_1) ** (1 / 4)
            logger.debug('Rejected RK4 step. Decreased time step to {}'.format(self.time_step))
            self.evolve_ARK4()

    def run_simulation(self):
        logger.info('Performing time evolution on {} ({})'.format(self.name, self.file_name))
        self.status = 'running'

        while self.time < self.spec.time_final:
            getattr(self, 'evolve_' + self.spec.evolution_method)()

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
