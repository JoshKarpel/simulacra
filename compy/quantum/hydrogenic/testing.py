import datetime as dt
import logging
import functools
import os

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from compy import core, math, utils
import compy.quantum as qm
from compy.quantum.hydrogenic import simulations
import compy.units as un

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class StaticConvergenceTestingSimulation(simulations.ElectricFieldSimulation):
    @property
    def norm_error_vs_time(self):
        return np.abs(1 - self.norm_vs_time)

    @property
    def initial_state_overlap_error_vs_time(self):
        return np.abs(1 - self.state_overlaps_vs_time[self.spec.initial_state])

    @property
    def initial_state_energy_expectation_error_vs_time(self):
        return np.abs(1 - np.abs((self.energy_expectation_value_vs_time_internal / (un.rydberg / (self.spec.initial_state.n ** 2)))))

    def attach_norm_error_vs_time_to_axis(self, axis, label = None):
        if label is None:
            label = r'{}'.format(self.spec.initial_state.tex_str)
        line, = axis.plot(self.times / un.asec, self.norm_error_vs_time, label = label)

        return line

    def attach_initial_state_overlap_error_to_axis(self, axis, label = None):
        if label is None:
            label = r'{}'.format(self.spec.initial_state.tex_str)
        line, = axis.plot(self.times / un.asec, self.initial_state_overlap_error_vs_time, label = label)

        return line

    def attach_initial_state_energy_expectation_error_to_axis(self, axis, label = None):
        if label is None:
            label = r'{}'.format(self.spec.initial_state.tex_str)
        line, = axis.plot(self.times / un.asec, self.initial_state_energy_expectation_error_vs_time, label = label)

        return line

    def plot_error_vs_time(self, show = False, save = False, **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        title = axis.set_title(r'Error vs. Time for ${}$'.format(self.spec.initial_state.tex_str), fontsize = 15)
        title.set_y(1.05)
        axis.set_xlabel(r'Time (as)', fontsize = 15)
        axis.set_ylabel(r'Error', fontsize = 15)

        axis.set_yscale('log')
        if self.times[-1] >= 1000 * un.asec:
            axis.set_xscale('log')

        axis.set_xlim(self.times[0] / un.asec, self.times[-1] / un.asec)
        axis.grid(True, color = 'black', linestyle = ':')  # change grid color to make it show up against the colormesh
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

        self.attach_norm_error_vs_time_to_axis(axis, label = r'$1 - \left< \psi | \psi(t=0) \right>$')
        self.attach_initial_state_overlap_error_to_axis(axis, label = r'$1 - \left< \psi | {} \right>$'.format(self.spec.initial_state.tex_str))
        self.attach_initial_state_energy_expectation_error_to_axis(axis, label = r'$1 - \frac{{\left< \psi | H | \psi \right>}}{{E_{}}}$'.format(self.spec.initial_state.n))

        axis.legend(loc = 'best', fontsize = 12)

        if save:
            utils.save_current_figure(name = self.spec.file_name + '__error_vs_time', **kwargs)
        if show:
            plt.show()

        plt.close()
