import datetime as dt
import logging
import functools
import os

import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt

from compy import core, math, utils
import compy.optics.core as opt
import compy.units as un

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Material:
    def __init__(self, name = 'material', length = 1 * un.cm):
        self.name = name
        self.length = length

    def index(self, wavelength):
        raise NotImplementedError

    def plot_index_vs_wavelength(self, wavelength_min, wavelength_max, show = False, save = False, **kwargs):
        wavelengths = np.linspace(wavelength_min, wavelength_max, 10 ** 4)
        indices = np.array([self.index(wavelength) for wavelength in wavelengths])

        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        plt.plot(wavelengths / un.nm, indices, label = self.name)

        # title = axis.set_title(r' for ${}$'.format(self.spec.initial_state.tex_str), fontsize = 15)
        # title.set_y(1.05)

        axis.set_xlabel(r'Wavelength $\lambda$ (nm)', fontsize = 15)
        axis.set_ylabel(r'Index of Refraction $n$', fontsize = 15)

        axis.set_xlim(wavelengths[0] / un.nm, wavelengths[-1] / un.nm)

        y_range = np.max(indices) - np.min(indices)
        axis.set_ylim(np.min(indices) - 0.1 * y_range, np.max(indices) + 0.1 * y_range)

        axis.grid(True, color = 'black', linestyle = ':')
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

        if save:
            utils.save_current_figure(name = self.name + '__index_vs_wavelength', **kwargs)
        if show:
            plt.show()

        plt.close()

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class ConstantIndex(Material):
    def __init__(self, name = 'vacuum', length = 1 * un.cm, index = 1):
        super(ConstantIndex, self).__init__(name = name, length = length)

        self.index = index

    def index(self, wavelength):
        return self.index


bk7_b = (1.03961212, 0.231792344, 1.01046945)
bk7_c = (6.00069867e-3 * (un.um ** 2), 2.00179144e-2 * (un.um ** 2), 1.03560653e2 * (un.um ** 2))


class Glass(Material):
    def __init__(self, name = 'BK7', length = 1 * un.cm, b = bk7_b, c = bk7_c):
        super(Glass, self).__init__(name = name, length = length)

        self.b = np.array(b)
        self.c = np.array(c)

    def index(self, wavelength):
        return np.sqrt(1 + np.sum((self.b / (wavelength ** 2 - self.c))) * (wavelength ** 2))


class Mode:

    __slots__ = ('center_frequency', 'power', 'linewidth', '_phase')

    def __init__(self, center_frequency = 100 * un.THz, power = 1 * un.W, linewidth = 1 * un.GHz, phase = 0):
        self.center_frequency = center_frequency
        self.power = power
        self.linewidth = linewidth
        self._phase = phase

    @property
    def phase(self):
        return self._phase % un.twopi

    @phase.setter
    def phase(self, phase):
        self._phase = phase % un.twopi

    @property
    def wavelength(self):
        return opt.photon_wavelength_from_frequency(self.center_frequency)

    def propagate(self, material):
        self.phase += material.length * (un.twopi * material.index(self.wavelength) / self.wavelength)

    def evaluate_at_t(self, t):
        return np.sin(self.center_frequency * t)  # NEED TO MULTIPLE BY RIGHT THING HERE TO GET FIELD AMPLITUDE

    def __str__(self):
        return 'Mode(center_frequency = {} THz, linewidth = {} GHz, power = {} W, phase = {})'.format(un.uround(self.center_frequency, un.THz, 3),
                                                                                            un.uround(self.linewidth, un.GHz, 3),
                                                                                            un.uround(self.power, un.W, 3),
                                                                                            self.phase)

    def __repr__(self):
        return 'Mode(center_frequency = {}, linewidth = {}, power = {}, phase = {})'.format(self.center_frequency, self.linewidth, self.power, self.phase)


class Beam:
    def __init__(self, *modes):
        self.modes = list(modes)

    def __iter__(self):
        yield from self.modes

    def __str__(self):
        return 'Beam: {}'.format(', '.join([str(mode) for mode in self.modes]))

    def __repr__(self):
        return 'Beam({})'.format(', '.join([repr(mode) for mode in self.modes]))

    def propagate(self, material):
        for mode in self.modes:
            mode.propagate(material)

    def evaluate_at_time(self, t):
        sum = np.zeros(np.shape(t))

        for beam in self.modes:
            sum += beam.evaluate_at_t(t)

        return sum

    def plot_field_vs_time(self, times, show = False, save = False, **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        plt.plot(t / un.fsec, self.evaluate_at_time(t))

        title = axis.set_title(r'Beam Electric Field vs. Time', fontsize = 15)
        title.set_y(1.05)
        axis.set_xlabel(r'$t (fs)$', fontsize = 15)
        axis.set_ylabel(r'$E(t)$', fontsize = 15)

        axis.set_xlim(t[0], t[-1])
        axis.grid(True, color = 'black', linestyle = '--')
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

        if save:
            utils.save_current_figure(name = 'electric_field_vs_time', **kwargs)
        if show:
            plt.show()

        plt.close()


def modulate_beam(beam, frequency_shift = 90 * un.THz, upshift_efficiency = 1e-4, downshift_efficiency = 1e-6):
    new_modes = []

    for mode in beam:
        upshift = Mode(center_frequency = mode.center_frequency + frequency_shift,
                       power = mode.power * upshift_efficiency,
                       linewidth = mode.linewidth,
                       phase = mode.phase)

        downshift = Mode(center_frequency = mode.center_frequency - frequency_shift,
                         power = mode.power * downshift_efficiency,
                         linewidth = mode.linewidth,
                         phase = mode.phase)

        notshift = Mode(center_frequency = mode.center_frequency,
                        power = mode.power - upshift.power - downshift.power,
                        linewidth = mode.linewidth,
                        phase = mode.phase)

        new_modes.append(upshift)
        new_modes.append(downshift)
        new_modes.append(notshift)

    return Beam(*new_modes)

