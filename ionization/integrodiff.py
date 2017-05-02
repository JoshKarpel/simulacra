import datetime as dt
import functools
import logging
import itertools as it
import collections
from copy import copy, deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

import compy as cp
from compy.units import *
from . import core, potentials, states

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def return_one(x, **kwargs):
    return 1


def gaussian_kernel(x, *, tau_alpha):
    return (1 + (1j * x / tau_alpha)) ** (-1.5)


class IntegroDifferentialEquationSpecification(cp.Specification):
    integration_method = cp.utils.RestrictedValues('integration_method', ('simpson', 'trapezoid'))

    def __init__(self, name,
                 time_initial = 0 * asec, time_final = 200 * asec, time_step = 1 * asec,
                 a_initial = 1,
                 prefactor = 1,
                 electric_potential = potentials.NoElectricField(),
                 electric_potential_dc_correction = True,
                 kernel = return_one, kernel_kwargs = None,
                 integration_method = 'simpson',
                 evolution_method = 'FE',
                 simulation_type = None,
                 checkpoints = False, checkpoint_every = 20, checkpoint_dir = None,
                 **kwargs):
        """
        Initialize an IntegroDifferentialEquationSpecification from the given parameters.

        The differential equation should be of the form
        dy/dt = prefactor * f(t) * integral[ y(t') * f(t') * kernel(t - t')  ; {t', t_initial, t} ]

        :param name:
        :param time_initial:
        :param time_final:
        :param time_step:
        :param a_initial: initial value of y
        :param kwargs:
        """
        if simulation_type is None:
            simulation_type = IntegroDifferentialEquationSimulation
        super().__init__(name, simulation_type = simulation_type, **kwargs)

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        self.a_initial = a_initial

        self.prefactor = prefactor

        self.electric_potential = electric_potential
        self.electric_potential_dc_correction = electric_potential_dc_correction

        self.kernel = kernel
        self.kernel_kwargs = dict()
        if kernel_kwargs is not None:
            self.kernel_kwargs.update(kernel_kwargs)

        self.integration_method = integration_method
        self.evolution_method = evolution_method

        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir

    def info(self):
        checkpoint = ['Checkpointing: ']
        if self.checkpoints:
            if self.checkpoint_dir is not None:
                working_in = self.checkpoint_dir
            else:
                working_in = 'cwd'
            checkpoint[0] += 'every {} time steps, working in {}'.format(self.checkpoint_every, working_in)
        else:
            checkpoint[0] += 'disabled'

        ide_parameters = [
            "IDE Parameters: da/dt = prefactor * f(t) * integral[ a(t') * f(t') * kernel(t - t')  ; {t', t_initial, t} ]",
            '   Initial State: a = {}'.format(self.a_initial),
            '   Prefactor: {}'.format(self.prefactor),
            '   Electric Potential: {}'.format(self.electric_potential),
            '   Kernel: {} with kwargs {}'.format(self.kernel.__name__, self.kernel_kwargs),
        ]

        time_evolution = [
            'Time Evolution:',
            '   Initial Time: {} as'.format(uround(self.time_initial, asec)),
            '   Final Time: {} as'.format(uround(self.time_final, asec)),
            '   Time Step: {} as'.format(uround(self.time_step, asec)),
            '   Integration Method: {}'.format(self.integration_method),
            '   Evolution Method: {}'.format(self.evolution_method),
        ]

        return '\n'.join(checkpoint + ide_parameters + time_evolution)


class IntegroDifferentialEquationSimulation(cp.Simulation):
    def __init__(self, spec):
        super().__init__(spec)

        total_time = self.spec.time_final - self.spec.time_initial
        self.times = np.linspace(self.spec.time_initial, self.spec.time_final, int(total_time / self.spec.time_step) + 1)
        self.time_index = 0

        self.a = np.zeros(self.time_steps, dtype = np.complex128) * np.NaN
        self.a[0] = self.spec.a_initial

        if self.spec.electric_potential_dc_correction:
            electric_field_vs_time = self.spec.electric_potential.get_electric_field_amplitude(self.times)
            average_electric_field = integrate.simps(electric_field_vs_time, x = self.times) / total_time

            old_pot = self.spec.electric_potential

            self.spec.electric_potential += potentials.Rectangle(start_time = self.times[0], end_time = self.times[-1], amplitude = -average_electric_field)

            logger.warning('Replaced electric potential {} --> {} for {} {}'.format(old_pot, self.spec.electric_potential, self.__class__.__name__, self.name))

        self.electric_field_vs_time = self.spec.electric_potential.get_electric_field_amplitude(self.times)

        if self.spec.integration_method == 'simpson':
            self.integrate = integrate.simps
        elif self.spec.integration_method == 'trapezoid':
            self.integrate = integrate.trapz

    @property
    def time_steps(self):
        return len(self.times)

    @property
    def time(self):
        return self.times[self.time_index]

    def evolve_FE(self):
        dt = self.times[self.time_index + 1] - self.time

        k = self.spec.prefactor * self.electric_field_vs_time[self.time_index] * self.integrate(y = self.electric_field_vs_time[:self.time_index + 1] * self.a[:self.time_index + 1] * self.spec.kernel(self.time - self.times[:self.time_index + 1], **self.spec.kernel_kwargs),
                                                                                                x = self.times[:self.time_index + 1])
        self.a[self.time_index + 1] = self.a[self.time_index] + (dt * k)  # estimate next point

    def evolve_BE(self):
        dt = self.times[self.time_index + 1] - self.time

        k = self.spec.prefactor * self.electric_field_vs_time[self.time_index + 1] * self.integrate(y = self.electric_field_vs_time[:self.time_index + 1] * self.a[:self.time_index + 1] * self.spec.kernel(self.times[self.time_index + 1] - self.times[:self.time_index + 1], **self.spec.kernel_kwargs),
                                                                                                    x = self.times[:self.time_index + 1])

        self.a[self.time_index + 1] = (self.a[self.time_index] + (dt * k)) / (1 - self.spec.prefactor * ((dt * self.electric_field_vs_time[self.time_index + 1]) ** 2))  # estimate next point

    def evolve_TRAP(self):
        dt = self.times[self.time_index + 1] - self.time

        k_1 = self.spec.prefactor * self.electric_field_vs_time[self.time_index + 1] * self.integrate(y = self.electric_field_vs_time[:self.time_index + 1] * self.a[:self.time_index + 1] * self.spec.kernel(self.times[self.time_index + 1] - self.times[:self.time_index + 1], **self.spec.kernel_kwargs),
                                                                                                      x = self.times[:self.time_index + 1])

        k_2 = self.spec.prefactor * self.electric_field_vs_time[self.time_index] * self.integrate(y = self.electric_field_vs_time[:self.time_index + 1] * self.a[:self.time_index + 1] * self.spec.kernel(self.times[self.time_index] - self.times[:self.time_index + 1], **self.spec.kernel_kwargs),
                                                                                                  x = self.times[:self.time_index + 1])

        self.a[self.time_index + 1] = (self.a[self.time_index] + (dt * (k_1 + k_2) / 2)) / (1 - .5 * self.spec.prefactor * ((dt * self.electric_field_vs_time[self.time_index + 1]) ** 2))  # estimate next point

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

        f_curr = self.electric_field_vs_time[self.time_index]
        f_half = self.spec.electric_potential.get_electric_field_amplitude(self.time + (dt / 2))
        f_next = self.electric_field_vs_time[self.time_index + 1]

        f_times_y_curr = self.electric_field_vs_time[:self.time_index + 1] * self.a[:self.time_index + 1]

        # integrate through the current time step
        integrand_for_k1 = f_times_y_curr * kernel_curr
        integral_for_k1 = self.integrate(y = integrand_for_k1, x = times_curr)
        k1 = self.spec.prefactor * f_curr * integral_for_k1
        y_midpoint_for_k2 = self.a[self.time_index] + (dt * k1 / 2)  # dt / 2 here because we moved forward to midpoint

        integrand_for_k2 = np.append(f_times_y_curr, f_half * y_midpoint_for_k2) * kernel_half
        integral_for_k2 = self.integrate(y = integrand_for_k2, x = times_half)
        k2 = self.spec.prefactor * f_half * integral_for_k2  # dt / 2 because it's half of an interval that we're integrating over
        y_midpoint_for_k3 = self.a[self.time_index] + (dt * k2 / 2)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k3 = np.append(f_times_y_curr, f_half * y_midpoint_for_k3) * kernel_half
        integral_for_k3 = self.integrate(y = integrand_for_k3, x = times_half)
        k3 = self.spec.prefactor * f_half * integral_for_k3  # dt / 2 because it's half of an interval that we're integrating over
        y_end_for_k4 = self.a[self.time_index] + (dt * k3)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k4 = np.append(f_times_y_curr, f_next * y_end_for_k4) * kernel_next
        integral_for_k4 = self.integrate(y = integrand_for_k4, x = times_next)
        k4 = self.spec.prefactor * f_next * integral_for_k4

        self.a[self.time_index + 1] = self.a[self.time_index] + (dt * (k1 + (2 * k2) + (2 * k3) + k4) / 6)  # estimate next point

    def run_simulation(self):
        logger.info(f'Performing time evolution on {self.name} ({self.file_name}), starting from time index {self.time_index}')
        self.status = 'running'

        while self.time < self.spec.time_final:
            getattr(self, 'evolve_' + self.spec.evolution_method)()

            self.time_index += 1

            logger.debug('{} {} ({}) evolved to time index {} / {} ({}%)'.format(self.__class__.__name__, self.name, self.file_name, self.time_index, self.time_steps - 1,
                                                                                 np.around(100 * (self.time_index + 1) / self.time_steps, 2)))

            if self.spec.checkpoints:
                if (self.time_index + 1) % self.spec.checkpoint_every == 0:
                    self.save(target_dir = self.spec.checkpoint_dir)
                    self.status = cp.STATUS_RUN
                    logger.info('Checkpointed {} {} ({}) at time step {}'.format(self.__class__.__name__, self.name, self.file_name, self.time_index + 1))

        self.status = 'finished'
        logger.info(f'Finished performing time evolution on {self.name} ({self.file_name})')

    def plot_a_vs_time(self, log = False, time_scale = 'asec', field_scale = 'AEF',
                       show_title = False,
                       plot_name = 'file_name',
                       **kwargs):
        fig = cp.plots.get_figure('full')

        x_scale_unit, x_scale_name = unit_value_and_name_from_unit(time_scale)
        f_scale_unit, f_scale_name = unit_value_and_name_from_unit(field_scale)

        grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [5, 1], hspace = 0.07)  # TODO: switch to fixed axis construction
        ax_a = plt.subplot(grid_spec[0])
        ax_f = plt.subplot(grid_spec[1], sharex = ax_a)

        ax_f.plot(self.times / x_scale_unit, self.spec.electric_potential.get_electric_field_amplitude(self.times) / f_scale_unit, color = cp.plots.RED, linewidth = 2)

        overlap = np.abs(self.a) ** 2
        ax_a.plot(self.times / x_scale_unit, overlap, color = 'black', linewidth = 2)

        if log:
            ax_a.set_yscale('log')
            min_overlap = np.min(overlap)
            ax_a.set_ylim(bottom = max(1e-9, min_overlap * .1), top = 1.0)
            ax_a.grid(True, which = 'both', **cp.plots.GRID_KWARGS)
        else:
            ax_a.set_ylim(0.0, 1.0)
            ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_a.grid(True, **cp.plots.GRID_KWARGS)

        ax_a.set_xlim(self.spec.time_initial / x_scale_unit, self.spec.time_final / x_scale_unit)

        ax_f.set_xlabel(r'Time $t$ (${}$)'.format(x_scale_name), fontsize = 13)
        ax_a.set_ylabel(r'$\left| a_{\alpha}(t) \right|^2$', fontsize = 13)
        ax_f.set_ylabel(r'${}$ (${}$)'.format(str_efield, f_scale_name), fontsize = 13, color = cp.plots.RED)

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
        ax_a.tick_params(axis = 'both', which = 'major', labelsize = 10)

        ax_a.tick_params(labelleft = True,
                         labelright = True,
                         labeltop = True,
                         labelbottom = False,
                         bottom = True,
                         top = True,
                         left = True,
                         right = True)
        ax_f.tick_params(labelleft = True,
                         labelright = True,
                         labeltop = False,
                         labelbottom = True,
                         bottom = True,
                         top = True,
                         left = True,
                         right = True)

        ax_f.grid(True, **cp.plots.GRID_KWARGS)

        if show_title:
            title = ax_a.set_title(self.name)
            title.set_y(1.15)

        postfix = ''
        if log:
            postfix += '__log'
        prefix = getattr(self, plot_name)

        name = prefix + '__solution_vs_time{}'.format(postfix)

        cp.plots.save_current_figure(name = name, **kwargs)

        plt.close()


class AdaptiveIntegroDifferentialEquationSpecification(IntegroDifferentialEquationSpecification):
    error_on = cp.utils.RestrictedValues('erron_on', ('y', 'dydt'))

    def __init__(self, name,
                 minimum_time_step = None, maximum_time_step = 1 * asec,
                 epsilon = 1e-3, error_on = 'y', safety_factor = .98,
                 evolution_method = 'ARK4',
                 simulation_type = None,
                 **kwargs):
        """
        Initiliaze a AdaptiveIntegroDifferentialEquationSpecification from the given parameters.

        if error_on == 'y':
            allowed_error = epsilon * y_current
        if error_on == 'dydt':
            allowed_error = epsilon * time_step_current * dy/dt_current

        :param name: the name of the Specification
        :param minimum_time_step: the smallest allowable time step
        :param maximum_time_step: the largest allowable time step
        :param epsilon: the fractional error tolerance
        :param error_on: 'y' or 'dydt', see above
        :param safety_factor: a number slightly less than 1 to ensure safe time step adjustment
        :param evolution_method:
        :param kwargs: kwargs are passed to IntegroDifferentialEquationSpecification's __init__ method
        """
        if simulation_type is None:
            simulation_type = AdaptiveIntegroDifferentialEquationSimulation
        super().__init__(name, evolution_method = evolution_method, simulation_type = simulation_type, **kwargs)

        self.minimum_time_step = minimum_time_step
        self.maximum_time_step = maximum_time_step

        self.epsilon = epsilon
        self.error_on = error_on

        self.safety_factor = safety_factor

    def info(self):
        checkpoint = ['Checkpointing: ']
        if self.checkpoints:
            if self.checkpoint_dir is not None:
                working_in = self.checkpoint_dir
            else:
                working_in = 'cwd'
            checkpoint[0] += 'every {} time steps, working in {}'.format(self.checkpoint_every, working_in)
        else:
            checkpoint[0] += 'disabled'

        ide_parameters = [
            "IDE Parameters: dy/dt = prefactor * f(t) * integral[ y(t') * f(t') * kernel(t - t')  ; {t', t_initial, t} ]",
            '   Initial State: y = {}'.format(self.a_initial),
            '   Prefactor: {}'.format(self.prefactor),
            '   Electric Potential: {}'.format(self.electric_potential),
            '   Kernel: {} with kwargs {}'.format(self.kernel.__name__, self.kernel_kwargs),
        ]

        time_evolution = [
            'Time Evolution:',
            '   Initial Time: {} as'.format(uround(self.time_initial, asec)),
            '   Final Time: {} as'.format(uround(self.time_final, asec)),
            '   Initial Time Step: {} as'.format(uround(self.time_step, asec)),
            '   Minimum Time Step: {} as'.format(uround(self.minimum_time_step, asec)),
            '   Maximum Time Step: {} as'.format(uround(self.maximum_time_step, asec)),
            '   Epsilon: {}'.format(self.epsilon),
            '   Error On: {}'.format(self.error_on),
            '   Safety Factor: {}'.format(self.safety_factor),
            '   Integration Method: {}'.format(self.integration_method),
            '   Evolution Method: {}'.format(self.evolution_method),
        ]

        return '\n'.join(checkpoint + ide_parameters + time_evolution)


class AdaptiveIntegroDifferentialEquationSimulation(IntegroDifferentialEquationSimulation):
    def __init__(self, spec):
        super().__init__(spec)

        self.times = np.array([self.spec.time_initial])

        self.time_step = self.spec.time_step

        self.a = np.array([self.spec.a_initial])
        self.time_steps_by_times = np.array([np.NaN])

        self.computed_time_steps = 0

        # if self.spec.electric_potential_dc_correction:
        #     total_time = self.spec.time_final - self.spec.time_initial
        #     densest_time_step_count = total_time / self.spec.minimum_time_step
        #
        #     approx_times = np.linspace(self.spec.time_initial, self.spec.time_final, densest_time_step_count)
        #
        #     electric_field_vs_time = self.spec.electric_potential.get_electric_field_amplitude(approx_times)
        #     average_electric_field = integrate.simps(electric_field_vs_time, x = approx_times) / total_time
        #
        #     old_pot = self.spec.electric_potential
        #
        #     self.spec.electric_potential += potentials.Rectangle(start_time = self.spec.time_initial, end_time = self.spec.time_initial, amplitude = -average_electric_field)
        #
        #     logger.warning('Replaced electric potential {} --> {} for {} {}'.format(old_pot, self.spec.electric_potential, self.__class__.__name__, self.name))

    def evolve_ARK4(self):
        """
        Evolve y forward in time by the time step, controlling for the local truncation error.

        :return: None
        """
        times_curr = self.times
        times_half = np.append(self.times[:self.time_index + 1], self.time + self.time_step / 2)
        times_next = np.append(self.times[:self.time_index + 1], self.time + self.time_step)

        time_difference_curr = self.time - times_curr  # slice up to current time index
        time_difference_half = (self.time + self.time_step / 2) - times_half
        time_difference_next = self.time + self.time_step - times_next

        kernel_curr = self.spec.kernel(time_difference_curr, **self.spec.kernel_kwargs)
        kernel_half = self.spec.kernel(time_difference_half, **self.spec.kernel_kwargs)
        kernel_next = self.spec.kernel(time_difference_next, **self.spec.kernel_kwargs)

        f_curr = self.spec.electric_potential.get_electric_field_amplitude(self.time)
        f_quarter = self.spec.electric_potential.get_electric_field_amplitude(self.time + (self.time_step / 4))
        f_half = self.spec.electric_potential.get_electric_field_amplitude(self.time + (self.time_step / 2))
        f_three_quarter = self.spec.electric_potential.get_electric_field_amplitude(self.time + (3 * self.time_step / 4))
        f_next = self.spec.electric_potential.get_electric_field_amplitude(self.time + self.time_step)

        f_times_y_curr = self.spec.electric_potential.get_electric_field_amplitude(self.times) * self.a

        # CALCULATE FULL STEP ESTIMATE
        integrand_for_k1 = f_times_y_curr * kernel_curr
        integral_for_k1 = self.integrate(y = integrand_for_k1, x = times_curr)
        k1 = self.spec.prefactor * f_curr * integral_for_k1
        y_midpoint_for_k2 = self.a[self.time_index] + (self.time_step * k1 / 2)  # self.time_step / 2 here because we moved forward to midpoint

        integrand_for_k2 = np.append(f_times_y_curr, f_half * y_midpoint_for_k2) * kernel_half
        integral_for_k2 = self.integrate(y = integrand_for_k2, x = times_half)
        k2 = self.spec.prefactor * f_half * integral_for_k2  # self.time_step / 2 because it's half of an interval that we're integrating over
        y_midpoint_for_k3 = self.a[self.time_index] + (self.time_step * k2 / 2)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k3 = np.append(f_times_y_curr, f_half * y_midpoint_for_k3) * kernel_half
        integral_for_k3 = self.integrate(y = integrand_for_k3, x = times_half)
        k3 = self.spec.prefactor * f_half * integral_for_k3  # self.time_step / 2 because it's half of an interval that we're integrating over
        y_end_for_k4 = self.a[self.time_index] + (self.time_step * k3)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k4 = np.append(f_times_y_curr, f_next * y_end_for_k4) * kernel_next
        integral_for_k4 = self.integrate(y = integrand_for_k4, x = times_next)
        k4 = self.spec.prefactor * f_next * integral_for_k4

        full_step_estimate = self.a[self.time_index] + (self.time_step * (k1 + (2 * k2) + (2 * k3) + k4) / 6)

        # CALCULATE DOUBLE HALF STEP ESTIMATE

        times_quarter = np.append(self.times[:self.time_index + 1], self.time + self.time_step / 4)
        times_three_quarter = np.append(np.append(self.times[:self.time_index + 1], self.time + self.time_step / 2), self.time + 3 * self.time_step / 4)
        times_next = np.append(np.append(self.times[:self.time_index + 1], self.time + self.time_step / 2), self.time + self.time_step)

        time_difference_quarter = (self.time + self.time_step / 4) - times_quarter
        time_difference_three_quarter = (self.time + 3 * self.time_step / 4) - times_three_quarter
        time_difference_next = self.time + self.time_step - times_next

        kernel_quarter = self.spec.kernel(time_difference_quarter, **self.spec.kernel_kwargs)
        kernel_three_quarter = self.spec.kernel(time_difference_three_quarter, **self.spec.kernel_kwargs)
        kernel_next = self.spec.kernel(time_difference_next, **self.spec.kernel_kwargs)

        # k1 is identical from above
        # integrand_for_k1 = f_times_y_curr * kernel_curr
        # integral_for_k1 = self.integrate(y = integrand_for_k1, x = times_curr)
        # k1 = self.spec.prefactor * f_curr * integral_for_k1
        y_midpoint_for_k2 = self.a[self.time_index] + (self.time_step * k1 / 4)  # self.time_step / 4 here because we moved forward to midpoint of midpoint

        integrand_for_k2 = np.append(f_times_y_curr, f_quarter * y_midpoint_for_k2) * kernel_quarter
        integral_for_k2 = self.integrate(y = integrand_for_k2, x = times_quarter)
        k2 = self.spec.prefactor * f_quarter * integral_for_k2  # self.time_step / 2 because it's half of an interval that we're integrating over
        y_midpoint_for_k3 = self.a[self.time_index] + (self.time_step * k2 / 4)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k3 = np.append(f_times_y_curr, f_quarter * y_midpoint_for_k3) * kernel_quarter
        integral_for_k3 = self.integrate(y = integrand_for_k3, x = times_quarter)
        k3 = self.spec.prefactor * f_quarter * integral_for_k3  # self.time_step / 2 because it's half of an interval that we're integrating over
        y_end_for_k4 = self.a[self.time_index] + (self.time_step * k2 / 2)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k4 = np.append(f_times_y_curr, f_half * y_end_for_k4) * kernel_half
        integral_for_k4 = self.integrate(y = integrand_for_k4, x = times_half)
        k4 = self.spec.prefactor * f_half * integral_for_k4

        y_half = self.a[self.time_index] + ((self.time_step / 2) * (k1 + (2 * k2) + (2 * k3) + k4) / 6)
        f_times_y_half = np.append(f_times_y_curr, f_half * y_half)

        integrand_for_k1 = f_times_y_half * kernel_half
        integral_for_k1 = self.integrate(y = integrand_for_k1, x = times_half)
        k1 = self.spec.prefactor * f_half * integral_for_k1
        y_midpoint_for_k2 = y_half + (self.time_step * k1 / 4)  # self.time_step / 4 here because we moved forward to midpoint of midpoint

        integrand_for_k2 = np.append(f_times_y_half, f_three_quarter * y_midpoint_for_k2) * kernel_three_quarter
        integral_for_k2 = self.integrate(y = integrand_for_k2, x = times_three_quarter)
        k2 = self.spec.prefactor * f_three_quarter * integral_for_k2  # self.time_step / 2 because it's half of an interval that we're integrating over
        y_midpoint_for_k3 = self.a[self.time_index] + (self.time_step * k2 / 4)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k3 = np.append(f_times_y_half, f_three_quarter * y_midpoint_for_k3) * kernel_three_quarter
        integral_for_k3 = self.integrate(y = integrand_for_k3, x = times_three_quarter)
        k3 = self.spec.prefactor * f_three_quarter * integral_for_k3  # self.time_step / 2 because it's half of an interval that we're integrating over
        y_end_for_k4 = self.a[self.time_index] + (self.time_step * k2 / 2)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k4 = np.append(f_times_y_half, f_next * y_end_for_k4) * kernel_next
        integral_for_k4 = self.integrate(y = integrand_for_k4, x = times_next)
        k4 = self.spec.prefactor * f_next * integral_for_k4

        deriv_estimate = (k1 + (2 * k2) + (2 * k3) + k4) / 6
        double_half_step_estimate = y_half + ((self.time_step / 2) * deriv_estimate)

        ##########################

        self.computed_time_steps += 1

        delta_1 = double_half_step_estimate - full_step_estimate  # estimate truncation error from difference in estimates of y

        if self.spec.error_on == 'y':
            delta_0 = self.spec.epsilon * self.a[self.time_index]
        elif self.spec.error_on == 'dydt':
            delta_0 = self.spec.epsilon * self.time_step * deriv_estimate

        ratio = np.abs(delta_0 / delta_1)

        if ratio >= 1 or np.isinf(ratio) or np.isnan(ratio) or self.time_step == self.spec.minimum_time_step:  # step was ok
            self.times = np.append(self.times, self.time + self.time_step)
            self.time_steps_by_times = np.append(self.time_steps_by_times, self.time_step)
            self.a = np.append(self.a, double_half_step_estimate + (delta_1 / 15))

            old_step = self.time_step  # for log message
            if delta_1 != 0:  # don't adjust time step if truncation error is zero
                self.time_step = self.spec.safety_factor * self.time_step * (ratio ** (1 / 5))

            # ensure new time step is inside min and max allowable time steps
            if self.spec.maximum_time_step is not None:
                self.time_step = min(self.spec.maximum_time_step, self.time_step)
            if self.spec.minimum_time_step is not None:
                self.time_step = max(self.spec.minimum_time_step, self.time_step)

            logger.debug('Accepted RK4 step to {} as. Increased time step to {} as from {} as'.format(uround(self.times[-1], 'asec', 6), uround(self.time_step, 'asec', 6), uround(old_step, 'asec', 5)))
        else:  # reject step
            old_step = self.time_step  # for log message
            self.time_step = self.spec.safety_factor * self.time_step * (ratio ** (1 / 4))  # set new time step

            if self.spec.maximum_time_step is not None:
                self.time_step = min(self.spec.maximum_time_step, self.time_step)
            if self.spec.minimum_time_step is not None:
                self.time_step = max(self.spec.minimum_time_step, self.time_step)

            logger.debug('Rejected RK4 step. Decreased time step to {} as from {} as'.format(uround(self.time_step, 'asec', 6), uround(old_step, 'asec', 6)))
            self.evolve_ARK4()  # retry with new time step

    def run_simulation(self):
        logger.info(f'Performing time evolution on {self.name} ({self.file_name}), starting from time index {self.time_index}')
        self.status = cp.STATUS_RUN

        while self.time < self.spec.time_final:
            getattr(self, 'evolve_' + self.spec.evolution_method)()

            self.time_index += 1

            logger.debug('{} {} ({}) evolved to time index {}'.format(self.__class__.__name__, self.name, self.file_name, self.time_index))

            if self.spec.checkpoints:
                if (self.time_index + 1) % self.spec.checkpoint_every == 0:
                    self.save(target_dir = self.spec.checkpoint_dir)
                    self.status = cp.STATUS_RUN
                    logger.info('Checkpointed {} {} ({}) at time step {}'.format(self.__class__.__name__, self.name, self.file_name, self.time_index + 1))

        self.status = cp.STATUS_FIN
        logger.info(f'Finished performing time evolution on {self.name} ({self.file_name})')


class VelocityGaugeIntegroDifferentialEquationSpecification(cp.Specification):
    integration_method = cp.utils.RestrictedValues('integration_method', ('simpson', 'trapezoid'))

    def __init__(self, name,
                 time_initial = 0 * asec, time_final = 200 * asec, time_step = 1 * asec,
                 test_mass = electron_mass, test_charge = electron_charge,
                 a_initial = 1,
                 prefactor = 1,
                 electric_potential = potentials.NoElectricField(),
                 electric_potential_dc_correction = True,
                 kernel = return_one, kernel_kwargs = None,
                 integration_method = 'simpson',
                 evolution_method = 'RK4',
                 simulation_type = None,
                 checkpoints = False, checkpoint_every = 20, checkpoint_dir = None,
                 **kwargs):
        """
        Initialize an IntegroDifferentialEquationSpecification from the given parameters.

        The differential equation should be of the form
        dy/dt = prefactor * f(t) * integral[ y(t') * f(t') * kernel(t - t')  ; {t', t_initial, t} ]

        :param name:
        :param time_initial:
        :param time_final:
        :param time_step:
        :param a_initial: initial value of y
        :param kwargs:
        """
        if simulation_type is None:
            simulation_type = VelocityGaugeIntegroDifferentialEquationSimulation
        super().__init__(name, simulation_type = simulation_type, **kwargs)

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        self.test_mass = test_mass
        self.test_charge = test_charge

        self.a_initial = a_initial

        self.prefactor = prefactor

        self.electric_potential = electric_potential
        self.electric_potential_dc_correction = electric_potential_dc_correction

        self.kernel = kernel
        self.kernel_kwargs = dict()
        if kernel_kwargs is not None:
            self.kernel_kwargs.update(kernel_kwargs)

        self.integration_method = integration_method
        self.evolution_method = evolution_method

        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir

    def info(self):
        checkpoint = ['Checkpointing: ']
        if self.checkpoints:
            if self.checkpoint_dir is not None:
                working_in = self.checkpoint_dir
            else:
                working_in = 'cwd'
            checkpoint[0] += 'every {} time steps, working in {}'.format(self.checkpoint_every, working_in)
        else:
            checkpoint[0] += 'disabled'

        ide_parameters = [
            "IDE Parameters: da/dt = prefactor * f(t) * integral[ y(t') * f(t') * kernel(t, t')  ; {t', t_initial, t} ]",
            '   Initial State: a = {}'.format(self.a_initial),
            '   Prefactor: {}'.format(self.prefactor),
            '   Electric Potential: {}'.format(self.electric_potential),
            '   Kernel: {} with kwargs {}'.format(self.kernel.__name__, self.kernel_kwargs),
        ]

        time_evolution = [
            'Time Evolution:',
            '   Initial Time: {} as'.format(uround(self.time_initial, asec)),
            '   Final Time: {} as'.format(uround(self.time_final, asec)),
            '   Time Step: {} as'.format(uround(self.time_step, asec)),
            '   Integration Method: {}'.format(self.integration_method),
            '   Evolution Method: {}'.format(self.evolution_method),
        ]

        return '\n'.join(checkpoint + ide_parameters + time_evolution)


def velocity_guassian_kernel(time_diff, quiver_diff, tau_alpha = 1, width = 1):
    time_diff_inner = 1 / (1 + (1j * time_diff / tau_alpha))
    alpha_diff_inner = (quiver_diff / width) ** 2

    exp = np.exp(-alpha_diff_inner * time_diff_inner / 8)
    inv = time_diff_inner ** 1.5
    diff = 1 - (.25 * alpha_diff_inner * time_diff_inner)

    return exp * diff * inv


class VelocityGaugeIntegroDifferentialEquationSimulation(cp.Simulation):
    def __init__(self, spec):
        super().__init__(spec)

        total_time = self.spec.time_final - self.spec.time_initial
        self.times = np.linspace(self.spec.time_initial, self.spec.time_final, int(total_time / self.spec.time_step) + 1)
        self.time_index = 0

        self.a = np.zeros(self.time_steps, dtype = np.complex128) * np.NaN
        self.a[0] = self.spec.a_initial

        if self.spec.electric_potential_dc_correction:
            electric_field_vs_time = self.spec.electric_potential.get_electric_field_amplitude(self.times)
            average_electric_field = integrate.simps(electric_field_vs_time, x = self.times) / total_time

            old_pot = self.spec.electric_potential

            self.spec.electric_potential += potentials.Rectangle(start_time = self.times[0], end_time = self.times[-1], amplitude = -average_electric_field)

            logger.warning('Replaced electric potential {} --> {} for {} {}'.format(old_pot, self.spec.electric_potential, self.__class__.__name__, self.name))

        self.time_step = np.abs(self.times[1] - self.times[0])

        self.electric_field_vs_time = self.spec.electric_potential.get_electric_field_amplitude(self.times)
        self.vector_potential_vs_time = -np.cumsum(self.electric_field_vs_time) * self.time_step
        self.quiver_motion_vs_time = -np.cumsum(self.vector_potential_vs_time) * self.time_step * (self.spec.test_charge / self.spec.test_mass)

        if self.spec.integration_method == 'simpson':
            self.integrate = integrate.simps
        elif self.spec.integration_method == 'trapezoid':
            self.integrate = integrate.trapz

    @property
    def time_steps(self):
        return len(self.times)

    @property
    def time(self):
        return self.times[self.time_index]

    def evolve_FE(self):
        time_diff = self.times[self.time_index] - self.times[:self.time_index + 1]
        quiver_diff = self.quiver_motion_vs_time[self.time_index] - self.quiver_motion_vs_time[:self.time_index + 1]
        integral = self.integrate(y = self.vector_potential_vs_time[:self.time_index + 1] * self.a[:self.time_index + 1] * self.spec.kernel(time_diff, quiver_diff, **self.spec.kernel_kwargs),
                                  x = self.times[:self.time_index + 1])

        k = self.spec.prefactor * self.vector_potential_vs_time[self.time_index] * integral

        self.a[self.time_index + 1] = self.a[self.time_index] + (self.time_step * k)  # estimate next point

    def evolve_BE(self):
        time_diff = self.times[self.time_index + 1] - self.times[:self.time_index + 1]
        quiver_diff = self.quiver_motion_vs_time[self.time_index + 1] - self.quiver_motion_vs_time[:self.time_index + 1]
        integral = self.integrate(y = self.vector_potential_vs_time[:self.time_index + 1] * self.a[:self.time_index + 1] * self.spec.kernel(time_diff, quiver_diff, **self.spec.kernel_kwargs),
                                  x = self.times[:self.time_index + 1])

        k = self.spec.prefactor * self.vector_potential_vs_time[self.time_index + 1] * integral

        self.a[self.time_index + 1] = (self.a[self.time_index] + (self.time_step * k)) / (1 - self.spec.prefactor * ((self.time_step * self.vector_potential_vs_time[self.time_index + 1]) ** 2))  # estimate next point

    def evolve_TRAP(self):
        time_diff_1 = self.times[self.time_index] - self.times[:self.time_index + 1]
        quiver_diff_1 = self.quiver_motion_vs_time[self.time_index] - self.quiver_motion_vs_time[:self.time_index + 1]
        integral_1 = self.integrate(y = self.vector_potential_vs_time[:self.time_index + 1] * self.a[:self.time_index + 1] * self.spec.kernel(time_diff_1, quiver_diff_1, **self.spec.kernel_kwargs),
                                    x = self.times[:self.time_index + 1])
        k_1 = self.spec.prefactor * self.vector_potential_vs_time[self.time_index] * integral_1

        time_diff_2 = self.times[self.time_index + 1] - self.times[:self.time_index + 1]
        quiver_diff_2 = self.quiver_motion_vs_time[self.time_index + 1] - self.quiver_motion_vs_time[:self.time_index + 1]
        integral_2 = self.integrate(y = self.vector_potential_vs_time[:self.time_index + 1] * self.a[:self.time_index + 1] * self.spec.kernel(time_diff_2, quiver_diff_2, **self.spec.kernel_kwargs),
                                    x = self.times[:self.time_index + 1])
        k_2 = self.spec.prefactor * self.vector_potential_vs_time[self.time_index + 1] * integral_2

        self.a[self.time_index + 1] = (self.a[self.time_index] + (self.time_step * (k_1 + k_2) / 2)) / (1 - .5 * self.spec.prefactor * ((self.time_step * self.vector_potential_vs_time[self.time_index + 1]) ** 2))

    def evolve_RK4(self):
        times_curr = self.times[:self.time_index + 1]
        time_at_half = self.time + self.time_step / 2
        times_half = np.append(self.times[:self.time_index + 1], time_at_half)
        times_next = self.times[:self.time_index + 2]

        time_difference_curr = self.time - times_curr  # slice up to current time index
        time_difference_half = time_at_half - times_half
        time_difference_next = self.times[self.time_index + 1] - times_next

        vector_potential_curr = self.vector_potential_vs_time[self.time_index]
        vector_potential_half = (self.vector_potential_vs_time[self.time_index] + self.vector_potential_vs_time[self.time_index + 1]) / 2
        vector_potential_next = self.vector_potential_vs_time[self.time_index + 1]

        quiver_at_half = (self.quiver_motion_vs_time[self.time_index] + self.quiver_motion_vs_time[self.time_index + 1]) / 2
        quiver_half = np.append(self.quiver_motion_vs_time[:self.time_index + 1], quiver_at_half)

        quiver_diff_curr = self.quiver_motion_vs_time[self.time_index] - self.quiver_motion_vs_time[:self.time_index + 1]
        quiver_diff_half = quiver_at_half - quiver_half  # not right
        quiver_diff_next = self.quiver_motion_vs_time[self.time_index + 1] - self.quiver_motion_vs_time[:self.time_index + 2]

        kernel_curr = self.spec.kernel(time_difference_curr, quiver_diff_curr, **self.spec.kernel_kwargs)
        kernel_half = self.spec.kernel(time_difference_half, quiver_diff_half, **self.spec.kernel_kwargs)
        kernel_next = self.spec.kernel(time_difference_next, quiver_diff_next, **self.spec.kernel_kwargs)

        A_times_a_curr = self.vector_potential_vs_time[:self.time_index + 1] * self.a[:self.time_index + 1]

        integrand_for_k1 = A_times_a_curr * kernel_curr
        integral_for_k1 = self.integrate(y = integrand_for_k1,
                                         x = times_curr)
        k1 = self.spec.prefactor * vector_potential_curr * integral_for_k1
        a_midpoint_for_k2 = self.a[self.time_index] + (self.time_step * k1 / 2)

        integrand_for_k2 = np.append(A_times_a_curr, vector_potential_half * a_midpoint_for_k2) * kernel_half
        integral_for_k2 = self.integrate(y = integrand_for_k2,
                                         x = times_half)
        k2 = self.spec.prefactor * vector_potential_half * integral_for_k2
        a_midpoint_for_k3 = self.a[self.time_index] + (self.time_step * k2 / 2)

        integrand_for_k3 = np.append(A_times_a_curr, vector_potential_half * a_midpoint_for_k3) * kernel_half
        integral_for_k3 = self.integrate(y = integrand_for_k3,
                                         x = times_half)
        k3 = self.spec.prefactor * vector_potential_half * integral_for_k3
        a_end_for_k4 = self.a[self.time_index] + (self.time_step * k3)

        integrand_for_k4 = np.append(A_times_a_curr, vector_potential_next * a_end_for_k4) * kernel_next
        integral_for_k4 = self.integrate(y = integrand_for_k4,
                                         x = times_next)
        k4 = self.spec.prefactor * vector_potential_next * integral_for_k4

        self.a[self.time_index + 1] = self.a[self.time_index] + (self.time_step * (k1 + (2 * k2) + (2 * k3) + k4) / 6)  # estimate next point

    def run_simulation(self):
        logger.info(f'Performing time evolution on {self.name} ({self.file_name}), starting from time index {self.time_index}')
        self.status = 'running'

        while self.time < self.spec.time_final:
            getattr(self, 'evolve_' + self.spec.evolution_method)()

            self.time_index += 1

            logger.debug('{} {} ({}) evolved to time index {} / {} ({}%)'.format(self.__class__.__name__, self.name, self.file_name, self.time_index, self.time_steps - 1,
                                                                                 np.around(100 * (self.time_index + 1) / self.time_steps, 2)))

            if self.spec.checkpoints:
                if (self.time_index + 1) % self.spec.checkpoint_every == 0:
                    self.save(target_dir = self.spec.checkpoint_dir)
                    self.status = cp.STATUS_RUN
                    logger.info('Checkpointed {} {} ({}) at time step {}'.format(self.__class__.__name__, self.name, self.file_name, self.time_index + 1))

        self.status = 'finished'
        logger.info(f'Finished performing time evolution on {self.name} ({self.file_name})')

    def plot_fields_vs_time(self, time_scale = 'asec', field_scale = 'AEF', vector_scale = 'atomic_momentum', quiver_scale = 'bohr_radius',
                            **kwargs):
        with cp.plots.FigureManager(f'{self.name}__fields_vs_time', **kwargs) as figman:
            fig = figman.fig
            ax = fig.add_subplot(111)

            t_scale_unit, t_scale_name = unit_value_and_name_from_unit(time_scale)
            f_scale_unit, f_scale_name = unit_value_and_name_from_unit(field_scale)
            a_scale_unit, a_scale_name = unit_value_and_name_from_unit(vector_scale)
            q_scale_unit, q_scale_name = unit_value_and_name_from_unit(quiver_scale)

            ax.plot(self.times / t_scale_unit, self.electric_field_vs_time / f_scale_unit, label = fr'${str_efield}(t)$ (${f_scale_name}$)')
            ax.plot(self.times / t_scale_unit, np.abs(self.spec.test_charge) * self.vector_potential_vs_time / a_scale_unit, label = fr'$e \, {str_afield}(t)$ (${a_scale_name}$)')
            ax.plot(self.times / t_scale_unit, self.quiver_motion_vs_time / q_scale_unit, label = fr'$\alpha(t)$ (${q_scale_name}$)')

            ax.set_xlim(self.times[0] / t_scale_unit, self.times[-1] / t_scale_unit)

            ax.set_xlabel(fr'Time $t$ (${t_scale_name}$)')

            ax.grid(True, **cp.plots.GRID_KWARGS)
            ax.legend(loc = 'best')

    def plot_a_vs_time(self, log = False, time_scale = 'asec', field_scale = 'AEF',
                       show_title = False,
                       plot_name = 'file_name',
                       **kwargs):
        fig = cp.plots.get_figure('full')

        x_scale_unit, x_scale_name = unit_value_and_name_from_unit(time_scale)
        f_scale_unit, f_scale_name = unit_value_and_name_from_unit(field_scale)

        grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [5, 1], hspace = 0.07)  # TODO: switch to fixed axis construction
        ax_a = plt.subplot(grid_spec[0])
        ax_f = plt.subplot(grid_spec[1], sharex = ax_a)

        ax_f.plot(self.times / x_scale_unit, self.spec.electric_potential.get_electric_field_amplitude(self.times) / f_scale_unit, color = cp.plots.RED, linewidth = 2)

        overlap = np.abs(self.a) ** 2
        ax_a.plot(self.times / x_scale_unit, overlap, color = 'black', linewidth = 2)

        if log:
            ax_a.set_yscale('log')
            min_overlap = np.min(overlap)
            ax_a.set_ylim(bottom = max(1e-9, min_overlap * .1), top = 1.0)
            ax_a.grid(True, which = 'both', **cp.plots.GRID_KWARGS)
        else:
            ax_a.set_ylim(0.0, 1.0)
            ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_a.grid(True, **cp.plots.GRID_KWARGS)

        ax_a.set_xlim(self.spec.time_initial / x_scale_unit, self.spec.time_final / x_scale_unit)

        ax_f.set_xlabel(r'Time $t$ (${}$)'.format(x_scale_name), fontsize = 13)
        ax_a.set_ylabel(r'$\left| a_{\alpha}(t) \right|^2$', fontsize = 13)
        ax_f.set_ylabel(r'${}$ (${}$)'.format(str_efield, f_scale_name), fontsize = 13, color = cp.plots.RED)

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
        ax_a.tick_params(axis = 'both', which = 'major', labelsize = 10)

        ax_a.tick_params(labelleft = True,
                         labelright = True,
                         labeltop = True,
                         labelbottom = False,
                         bottom = True,
                         top = True,
                         left = True,
                         right = True)
        ax_f.tick_params(labelleft = True,
                         labelright = True,
                         labeltop = False,
                         labelbottom = True,
                         bottom = True,
                         top = True,
                         left = True,
                         right = True)

        ax_f.grid(True, **cp.plots.GRID_KWARGS)

        if show_title:
            title = ax_a.set_title(self.name)
            title.set_y(1.15)

        postfix = ''
        if log:
            postfix += '__log'
        prefix = getattr(self, plot_name)

        name = prefix + '__solution_vs_time{}'.format(postfix)

        cp.plots.save_current_figure(name = name, **kwargs)

        plt.close()
