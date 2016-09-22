import datetime as dt
import logging
import functools
import os

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from compy import core, math, utils
import compy.quantum as qm
from compy.quantum.hydrogenic import simulations, animators, potentials
import compy.units as un
import compy.cy as cy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ConvergenceTestingSimulation(simulations.ElectricFieldSimulation):
    @property
    def norm_error_vs_time(self):
        return np.abs(1 - self.norm_vs_time)

    @property
    def initial_state_overlap_error_vs_time(self):
        return np.abs(1 - self.state_overlaps_vs_time[self.spec.initial_state])

    @property
    def initial_state_energy_expectation_error_vs_time(self):
        return np.abs(1 - np.abs((self.energy_expectation_value_vs_time_internal / (rydberg / (self.spec.initial_state.n ** 2)))))

    def attach_norm_error_vs_time_to_axis(self, axis):
        line, = axis.plot(self.times / un.asec, self.norm_error_vs_time, label = r'{}'.format(self.spec.initial_state.tex_str))

        return line

    def attach_initial_state_overlap_error_to_axis(self, axis):
        line, = axis.plot(self.times / un.asec, self.initial_state_overlap_error_vs_time, label = r'{}'.format(self.spec.initial_state.tex_str))

        return line

    def attach_initial_state_energy_expectation_error_to_axis(self, axis):
        line, = axis.plot(self.times / un.asec, self.initial_state_energy_expectation_error_vs_time, label = r'{}'.format(self.spec.initial_state.tex_str))

        return line

    def plot_error_vs_time(self, show = False, save = False, **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        axis.set_xlabel(r'Time (as)', fontsize = 15)

        axis.set_yscale('log')

        axis.set_xlim(self.times[0] / un.asec, self.times[-1] / un.asec)

        axis.grid(True, color = 'black', linestyle = ':')  # change grid color to make it show up against the colormesh

        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

        if save:
            utils.save_current_figure(name = self.spec.file_name + '__error_vs_time', **kwargs)
        if show:
            plt.show()

        plt.close()
