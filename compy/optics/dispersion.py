import datetime as dt
import logging
import functools
import os

import numpy as np
import numpy.fft as fft
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
        return '{} cm of {}'.format(un.uround(self.length, un.cm, 2), self.name)

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

fs_b = (0.696166300, 0.407942600, 0.897479400)
fs_c = (4.67914826e-3 * (un.um ** 2), 1.35120631e-2 * (un.um ** 2), 97.9340025 * (un.um ** 2))


class Glass(Material):
    def __init__(self, name = 'BK7', length = 1 * un.cm, b = bk7_b, c = bk7_c):
        super(Glass, self).__init__(name = name, length = length)

        self.b = np.array(b)
        self.c = np.array(c)

    @utils.memoize()
    def index(self, wavelength):
        return np.sqrt(1 + np.sum((self.b / (wavelength ** 2 - self.c))) * (wavelength ** 2))


class Mode:
    __slots__ = ('frequency', 'intensity', 'linewidth', '_phase')

    def __init__(self, frequency = 100 * un.THz, intensity = 1 * un.W / (un.cm ** 2), phase = 0):
        self.frequency = frequency
        self.intensity = intensity
        self._phase = phase

    @property
    def phase(self):
        return self._phase % un.twopi

    @phase.setter
    def phase(self, phase):
        self._phase = phase % un.twopi

    @property
    def amplitude(self):
        return np.sqrt(self.intensity * 2 / (un.c * un.epsilon_0))

    @property
    def angular_frequency(self):
        return un.twopi * self.frequency

    @property
    def wavelength(self):
        return opt.photon_wavelength_from_frequency(self.frequency)

    def propagate(self, material):
        # dphase = un.twopi * material.length * material.index(self.wavelength) / self.wavelength
        # self.phase += dphase

        self.phase += un.twopi * material.length * material.index(self.wavelength) / self.wavelength

        # logger.debug('Propagated mode {} through {}, phase changed by {}'.format(self, material, dphase))

    def evaluate_at_time(self, t):
        return self.amplitude * np.exp((1j * un.twopi * self.frequency * t) + self.phase)

    def __str__(self):
        return 'Mode(frequency = {} THz, wavelength = {} nm, intensity = {} W/m^2, phase = {})'.format(un.uround(self.frequency, un.THz, 3),
                                                                                   un.uround(self.wavelength, un.nm, 3),
                                                                                   un.uround(self.intensity, un.W, 3),
                                                                                   self.phase)

    def __repr__(self):
        return 'Mode(frequency = {}, wavelength = {}, intensity = {}, phase = {})'.format(self.frequency, self.wavelength, self.intensity, self.phase)


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
        result = np.zeros(np.shape(t), dtype = np.complex128)

        # frequencies = np.zeros(len(self.modes), dtype = np.complex128)
        # phases = np.zeros(len(self.modes), dtype = np.complex128)
        # amplitudes = np.zeros(len(self.modes), dtype = np.complex128)

        for beam in self.modes:
            result += beam.evaluate_at_time(t)

        # for ii, beam in enumerate(self.modes):
        #     frequencies[ii] = beam.center_frequency
        #     phases[ii] = beam.phase
        #     amplitudes[ii] = beam.amplitude

        # return np.sum(amplitudes * np.exp((1j * un.twopi * frequencies * t) + phases))

        return result

    def fft_field(self, t):
        return fft.fft(np.real(self.evaluate_at_time(t)), norm = 'ortho'), fft.fftfreq(len(t), t[1] - t[0])

    def plot_field_vs_time(self, time_initial = -200 * un.fsec, time_final = 200 * un.fsec, time_points = 10 ** 6, show = False, save = False, name_postfix = '', **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        t = np.linspace(time_initial, time_final, time_points)

        electric_field = self.evaluate_at_time(t)

        plt.plot(t / un.fsec, np.real(electric_field),
                 label = 'Electric Field', color = 'blue', linestyle = '-')
        plt.plot(t / un.fsec, np.abs(electric_field), t / un.fsec, -np.abs(electric_field),
                 label = 'Envelope', color = 'red', linestyle = '--')

        title = axis.set_title(r'Beam Electric Field vs. Time', fontsize = 15)
        title.set_y(1.05)
        axis.set_xlabel(r'$t$ (fs)', fontsize = 15)
        axis.set_ylabel(r'$E(t)$ (V/m)', fontsize = 15)

        axis.set_xlim(t[0] / un.fsec, t[-1] / un.fsec)
        axis.grid(True, color = 'grey', linestyle = ':')
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

        if save:
            utils.save_current_figure(name = 'electric_field_vs_time__' + name_postfix, **kwargs)
        if show:
            plt.show()

        plt.close()

    def plot_fft(self, time_initial = -200 * un.fsec, time_final = 200 * un.fsec, time_points = 10 ** 6, show = False, save = False, name_postfix = '', **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        t = np.linspace(time_initial, time_final, time_points)
        fft_field, fft_freq = self.fft_field(t)
        shifted_fft = fft.fftshift(fft_field)
        shifted_freq = fft.fftshift(fft_freq)

        plt.plot(shifted_freq / un.THz, np.abs(shifted_fft))

        title = axis.set_title(r'FFT', fontsize = 15)
        title.set_y(1.05)
        axis.set_xlabel(r'$f$ (THz)', fontsize = 15)
        axis.set_ylabel(r'$\left|\hat{E}(f)\right|$', fontsize = 15)

        # axis.set_xlim(shifted_freq[0] / un.THz, shifted_freq[-1] / un.THz)
        axis.grid(True, color = 'black', linestyle = '--')
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

        if save:
            utils.save_current_figure(name = 'fft__' + name_postfix, **kwargs)
        if show:
            plt.show()

        plt.close()


def modulate_beam(beam, frequency_shift = 90 * un.THz, upshift_efficiency = 1e-6, downshift_efficiency = 1e-6):
    new_modes = []

    for mode in beam:
        upshift = Mode(frequency = mode.frequency + frequency_shift,
                       intensity = mode.intensity * upshift_efficiency,
                       phase = mode.phase)
        new_modes.append(upshift)

        downshift = Mode(frequency = mode.frequency - frequency_shift,
                         intensity = mode.intensity * downshift_efficiency,
                         phase = mode.phase)
        if downshift.frequency > 0:  # only add the downshifted mode if it's frequency is still greater than zero
            new_modes.append(downshift)

        notshift = Mode(frequency = mode.frequency,
                        intensity = mode.intensity - upshift.intensity - downshift.intensity,
                        phase = mode.phase)

        new_modes.append(notshift)

        logger.debug('Generated sidebands for mode {}: {}, {}, {}'.format(mode, upshift, downshift, notshift))

    return Beam(*new_modes)


def bandblock_beam(beam, wavelength_min, wavelength_max, filter_by = 1e-6):
    new_modes = []

    for mode in beam:
        if wavelength_min < mode.wavelength < wavelength_max:
            new_mode = Mode(frequency = mode.frequency,
                            intensity = mode.intensity * filter_by,
                            phase = mode.phase)

            logger.debug('Filtered mode {} to {}'.format(mode, new_mode))
        else:
            new_mode = mode

            logger.debug('Filter ignored mode {}'.format(mode))

        new_modes.append(new_mode)

    return Beam(*new_modes)