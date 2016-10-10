import datetime as dt
import logging
import functools
import os
import collections

import numpy as np
import numpy.fft as nfft
import scipy as sp
import scipy.integrate as integ
import scipy.optimize as optim
import matplotlib
import matplotlib.pyplot as plt

from compy import core, math, utils
import compy.optics.core as opt
import compy.units as un

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        self._phase = phase  # % un.twopi

    @property
    def amplitude(self):
        return np.sqrt(self.intensity * 2 / (un.c * un.epsilon_0))

    @property
    def angular_frequency(self):
        return un.twopi * self.frequency

    @property
    def wavelength(self):
        return opt.photon_wavelength_from_frequency(self.frequency)

    @property
    def wavenumber(self):
        return un.twopi / self.wavelength

    def propagate(self, material):
        # dphase = un.twopi * material.length * material.index(self.wavelength) / self.wavelength
        # self.phase += dphase

        self.phase += un.twopi * material.length * material.index(self.wavelength) / self.wavelength

        # logger.debug('Propagated mode {} through {}, phase changed by {}'.format(self, material, dphase))

    def evaluate_at_time(self, t):
        return self.amplitude * np.exp(1j * ((un.twopi * self.frequency * t) + self.phase))

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
        return nfft.fft(np.real(self.evaluate_at_time(t)), norm = 'ortho'), nfft.fftfreq(len(t), t[1] - t[0])

    def plot_field_vs_time(self, time_initial = -500 * un.fsec, time_final = 500 * un.fsec, time_points = 10 ** 5, show = False, save = False, name_postfix = '', **kwargs):
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
        shifted_fft = nfft.fftshift(fft_field)
        shifted_freq = nfft.fftshift(fft_freq)

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
        if wavelength_min < mode.wavelengths < wavelength_max:
            new_mode = Mode(frequency = mode.frequency,
                            intensity = mode.intensity * filter_by,
                            phase = mode.phase)

            logger.debug('Filtered mode {} to {}'.format(mode, new_mode))
        else:
            new_mode = mode

            logger.debug('Filter ignored mode {}'.format(mode))

        new_modes.append(new_mode)

    return Beam(*new_modes)


IFFTResult = collections.namedtuple('IFFTResult', ('time', 'field'))
PulseWidthFitResult = collections.namedtuple('PulseWidthFitResult', ('center', 'sigma', 'prefactor', 'covariance_matrix'))


class ContinuousAmplitudeSpectrumSpecification(core.Specification):
    def __init__(self, name, frequencies, initial_amplitude, materials, **kwargs):
        super(ContinuousAmplitudeSpectrumSpecification, self).__init__(name, **kwargs)

        self.frequencies = frequencies
        self.initial_amplitude = initial_amplitude.astype(np.complex128)

        self.materials = materials

    @classmethod
    def from_power_spectrum_csv(cls, name, frequencies, materials,
                                path_to_csv, total_power = 100 * un.mW, x_units = 'nm', y_units = 'dBm',
                                fit_guess_center = 800 * un.nm, fit_guess_fwhm = 40 * un.nm):
        wavelengths, power = np.loadtxt(path_to_csv, delimiter = ',', unpack = True, skiprows = 1)

        wavelengths *= un.unit_names_to_values[x_units]

        if y_units == 'dBm':
            power = 1 * un.mW * (10 ** (power / 10))
        else:
            power *= un.unit_names_to_values[y_units]

        spectrum_power = integ.simps(power, wavelengths)
        power_ratio = total_power / spectrum_power
        power *= power_ratio

        guesses = (fit_guess_center, fit_guess_fwhm, np.max(power))
        popt, pcov = optim.curve_fit(math.gaussian, wavelengths, power, p0 = guesses)

        fitted_power = math.gaussian(opt.photon_wavelength_from_frequency(frequencies), *popt)
        fitted_amplitude = np.sqrt(fitted_power)  # TODO: correct units

        return cls(name, frequencies, fitted_amplitude, materials)


class ContinuousAmplitudeSpectrumSimulation(core.Simulation):
    def __init__(self, spec):
        super(ContinuousAmplitudeSpectrumSimulation, self).__init__(spec)

        self.frequencies = spec.frequencies
        self.df = self.frequencies[1] - self.frequencies[0]
        self.amplitude = spec.initial_amplitude.copy()

        self.materials = spec.materials

        self.pulse_fits_vs_materials = []

    @property
    def angular_frequencies(self):
        return opt.photon_angular_frequency_from_frequency(self.frequencies)

    @property
    def wavelengths(self):
        return opt.photon_wavelength_from_frequency(self.frequencies)

    @property
    def power(self):
        return np.real(np.abs(self.amplitude) ** 2)

    def fft(self):
        t = nfft.fftshift(nfft.fftfreq(len(self.frequencies), self.df))

        reference_frequency = self.frequencies[len(self.frequencies) // 2]
        shifted_spectrum = nfft.fftshift(self.amplitude)

        field = nfft.fftshift(nfft.ifft(shifted_spectrum, norm = 'ortho')) * np.exp(1j * reference_frequency * t)  # restore reference frequency

        return IFFTResult(time = t, field = field)

    def fit_pulse(self):
        fft_result = self.fft()

        t, field = fft_result

        t_center = t[np.argmax(np.abs(field))]
        field_max = np.max(np.abs(field))

        guesses = [t_center, 100 * un.fsec, field_max]
        popt, pcov = optim.curve_fit(math.gaussian, t, np.real(np.abs(field)), p0 = guesses,
                                     bounds = ([-10 * un.fsec + t_center, 5 * un.fsec, 0], [t_center + 10 * un.fsec, 10 * un.psec, np.inf]))

        return PulseWidthFitResult(center = popt[0], sigma = popt[1], prefactor = popt[2], covariance_matrix = pcov), fft_result

    def plot_amplitude_vs_frequency(self, **kwargs):
        raise NotImplementedError

    def plot_power_vs_frequency(self, **kwargs):
        utils.xy_plot(np.real(self.frequencies), [self.power], name = '{}__power_vs_frequency'.format(self.name), x_label = r'Frequency $f$', **kwargs)

    def plot_amplitude_vs_wavelength(self, **kwargs):
        raise NotImplementedError

    def plot_power_vs_wavelength(self, **kwargs):
        utils.xy_plot(np.real(self.wavelengths), [self.power],
                      name = '{}__power_vs_wavelength'.format(self.name), x_label = r'Wavelength $\lambda$', **kwargs)

    def plot_electric_field_vs_time(self, show = False, save = False, x_scale = 'fs', y_scale = None, **kwargs):
        fit_result, fft_result = self.fit_pulse()

        t_center, t_width, prefactor, _ = fit_result
        t, field = fft_result

        fitted_envelope = math.gaussian(t, t_center, t_width, prefactor)

        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)
        fig.set_tight_layout(True)
        axis = plt.subplot(111)

        if x_scale is not None:
            scaled_t = t / un.unit_names_to_values[x_scale]
        else:
            scaled_t = t

        if y_scale is not None:
            e_scale = un.unit_names_to_values[y_scale]
        else:
            e_scale = 1

        axis.plot(scaled_t, np.real(field) / e_scale, label = r'$E(t)$')
        axis.plot(scaled_t, np.abs(field) / e_scale, label = r'$\left| E(t) \right|$')
        axis.plot(scaled_t, fitted_envelope / e_scale,
                  label = r'$\tau = {}$ {}'.format(un.uround(math.gaussian_fwhm_from_sigma(fit_result.sigma), un.unit_names_to_values[x_scale], 2), un.unit_names_to_tex_strings[x_scale]),
                  linestyle = '--', color = 'orange')

        title = axis.set_title(r'Electric Field vs. Time', fontsize = 15)
        title.set_y(1.05)

        x_label = r'Time $t$'
        x_label += r' ({})'.format(un.unit_names_to_tex_strings[x_scale])
        axis.set_xlabel(r'{}'.format(x_label), fontsize = 15)

        y_label = r'Electric Field $E(t)$'
        if y_scale is not None:
            y_label += r' ({})'.format(un.unit_names_to_tex_strings[y_scale])
        axis.set_ylabel(r'{}'.format(y_label), fontsize = 15)

        x_range = 4 * t_width
        lower_limit_x = t_center - x_range
        upper_limit_x = t_center + x_range
        axis.set_xlim(lower_limit_x / un.unit_names_to_values[x_scale], upper_limit_x / un.unit_names_to_values[x_scale])

        axis.grid(True, color = 'gray', linestyle = ':', alpha = 0.9)
        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)

        axis.legend(loc = 'best', fontsize = 12)

        if save:
            utils.save_current_figure(name = '{}__electric_field_vs_time'.format(self.name), **kwargs)
        if show:
            plt.show()

        plt.close()

        return fit_result

    def propagate(self, material):
        self.amplitude *= np.exp(1j * un.twopi * material.length * material.index(self.wavelengths) * self.frequencies / un.c)

    def run_simulation(self, plot_intermediate_electric_fields = False, target_dir = None):
        logger.info('Performing propagation on {} ({})'.format(self.name, self.file_name))

        self.status = 'running'
        logger.debug("{} {} status set to 'running'".format(self.__class__.__name__, self.name))

        if target_dir is None:
            target_dir = os.getcwd()

        if plot_intermediate_electric_fields:
            self.pulse_fits_vs_materials.append(self.plot_electric_field_vs_time(save = True, target_dir = target_dir, name_postfix = '_0of{}'.format(len(self.materials))))

        for ii, material in enumerate(self.materials):
            self.propagate(material)

            if plot_intermediate_electric_fields:
                self.pulse_fits_vs_materials.append(self.plot_electric_field_vs_time(save = True, target_dir = target_dir, name_postfix = '_{}of{}'.format(ii + 1, len(self.materials))))
            else:
                self.pulse_fits_vs_materials.append(self.fit_pulse()[0])

            logger.debug('Propagated {} through {} ({}/{} Materials)'.format(self.name, material, ii + 1, len(self.materials)))

        self.end_time = dt.datetime.now()
        self.elapsed_time = self.end_time - self.start_time

        self.status = 'finished'
        logger.debug("Simulation status set to 'finished'")
        logger.info('Finished performing propagation on {} ({})'.format(self.name, self.file_name))

    def get_pulse_width_vs_materials(self):
        out = ['Initial: FWHM = {} fs,  Peak Amplitude = {}'.format(un.uround(math.gaussian_fwhm_from_sigma(self.pulse_fits_vs_materials[0].sigma), un.fsec, 2), self.pulse_fits_vs_materials[0].prefactor)]

        for material, fit in zip(self.materials, self.pulse_fits_vs_materials[1:]):
            out.append('After {}: FWHM = {} fs, Peak Amplitude = {}'.format(material, un.uround(math.gaussian_fwhm_from_sigma(fit.sigma), un.fsec, 2), fit.prefactor))

        return '\n'.join(out)
