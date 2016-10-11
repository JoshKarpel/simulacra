import logging

import numpy as np
import matplotlib.pyplot as plt

from compy import utils, math
import compy.units as un

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def photon_wavelength_from_frequency(frequency):
    return un.c / frequency


def photon_frequency_from_wavelength(wavelength):
    return un.c / wavelength


def photon_wavenumber_from_wavelength(wavelength):
    raise NotImplementedError


def photon_wavelength_from_wavenumber(wavenumber):
    raise NotImplementedError


def photon_angular_frequency_from_frequency(frequency):
    return un.twopi * frequency


def photon_frequency_from_angular_frequency(angular_frequency):
    return angular_frequency / un.twopi


def photon_wavelength_from_angular_frequency(angular_frequency):
    return photon_wavelength_from_frequency(photon_frequency_from_angular_frequency(angular_frequency))


def photon_angular_frequency_from_wavelength(wavelength):
    return photon_angular_frequency_from_frequency(photon_frequency_from_wavelength(wavelength))


def calculate_gvd(frequencies, material):
    angular_frequencies = photon_angular_frequency_from_frequency(frequencies)
    wavelengths = photon_wavelength_from_frequency(frequencies)

    n = material.index(wavelengths)

    dw = angular_frequencies[1] - angular_frequencies[0]
    dn_dw = math.centered_first_derivative(n, dw)
    ddn_ddw = math.centered_first_derivative(dn_dw, dw)

    gvd = ((2 / un.c) * dn_dw) + ((angular_frequencies / un.c) * ddn_ddw)

    return gvd


class Material:
    def __init__(self, name = 'Material', length = 1 * un.cm):
        self.name = name
        self.length = length

    def index(self, wavelength):
        raise NotImplementedError

    def plot_index_vs_wavelength(self, wavelength_min, wavelength_max, show = False, save = False, **kwargs):
        wavelengths = np.linspace(wavelength_min, wavelength_max, 1e6)
        indices = np.array([self.index(wavelength) for wavelength in wavelengths])

        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        plt.plot(wavelengths / un.nm, indices, label = self.name)

        title = axis.set_title(r'$n$ vs. $\lambda$ for {}'.format(self.name), fontsize = 15)
        title.set_y(1.05)

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

    def plot_gvd_vs_wavelength(self, wavelength_min, wavelength_max, show = False, save = False, ** kwargs):
        lower_frequency = photon_frequency_from_wavelength(wavelength_max)
        upper_frequency = photon_frequency_from_wavelength(wavelength_min)
        frequencies = np.linspace(lower_frequency, upper_frequency, 1e6)  # have to start with evenly spaced frequencies for the simple centered difference in calculate_gvd() to work
        wavelengths = photon_wavelength_from_frequency(frequencies)

        gvd = calculate_gvd(frequencies, self)

        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        y_scale = (un.fsec ** 2) / un.mm

        plt.plot(wavelengths / un.nm, un.uround(gvd, y_scale, 10), label = self.name)

        title = axis.set_title(r'GVD vs. $\lambda$ for {}'.format(self.name), fontsize = 15)
        title.set_y(1.05)

        axis.set_xlabel(r'Wavelength $\lambda$ (nm)', fontsize = 15)
        axis.set_ylabel(r'GVD ($\mathrm{fs}^2/\mathrm{mm})$', fontsize = 15)

        axis.set_xlim(np.min(wavelengths) / un.nm, np.max(wavelengths) / un.nm)

        y_range = np.max(gvd) - np.min(gvd)
        axis.set_ylim((np.min(gvd) - 0.1 * y_range) / y_scale, (np.max(gvd) + 0.1 * y_range) / y_scale)

        axis.grid(True, color = 'black', linestyle = ':')
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

        if save:
            utils.save_current_figure(name = self.name + '__gvd_vs_wavelength', **kwargs)
        if show:
            plt.show()

        plt.close()

    def __str__(self):
        return '{} cm of {}'.format(un.uround(self.length, un.cm, 2), self.name)

    def __repr__(self):
        return '{}(length = {} cm)'.format(self.__class__.__name__, un.uround(self.length, un.cm, 3))


class ConstantIndex(Material):
    def __init__(self, name = 'Vacuum', length = 1 * un.cm, index = 1):
        super(ConstantIndex, self).__init__(name = name, length = length)

        self._index = index

    def index(self, wavelength):
        return self._index


class SellmeierGlass(Material):
    def __init__(self, name = 'Glass', length = 1 * un.cm, b = (0, 0, 0), c = (0, 0, 0)):
        super(SellmeierGlass, self).__init__(name = name, length = length)

        self.b = np.array(b)
        self.c = np.array(c)

    def index(self, wavelength):
        return np.sqrt(1 + wavelength ** 2 * ((self.b[0] / (wavelength ** 2 - self.c[0])) + (self.b[1] / (wavelength ** 2 - self.c[1])) + (self.b[2] / (wavelength ** 2 - self.c[2]))))


bk7_b = (1.03961212, 0.231792344, 1.01046945)
bk7_c = (6.00069867e-3 * (un.um ** 2), 2.00179144e-2 * (un.um ** 2), 1.03560653e2 * (un.um ** 2))


class BK7(SellmeierGlass):
    def __init__(self, name = 'BK7', length = 1 * un.cm):
        super(BK7, self).__init__(name = name, length = length, b = bk7_b, c = bk7_c)


fs_b = (0.696166300, 0.407942600, 0.897479400)
fs_c = (4.67914826e-3 * (un.um ** 2), 1.35120631e-2 * (un.um ** 2), 97.9340025 * (un.um ** 2))


class FS(SellmeierGlass):
    def __init__(self, name = 'FS', length = 1 * un.cm):
        super(FS, self).__init__(name = name, length = length, b = fs_b, c = fs_c)
