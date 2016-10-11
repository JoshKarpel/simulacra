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


class BeamModifier:
    def __init__(self, name = 'BeamModifier'):
        self.name = name

    def propagate(self, frequencies, amplitudes):
        raise NotImplementedError


class BandBlockBeam(BeamModifier):
    def __init__(self, wavelength_min = 700 * un.nm, wavelength_max = 900 * un.nm, reduction_factor = 1e-6):
        super(BandBlockBeam, self).__init__(name = 'BandBlock')

        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max

        self.reduction_factor = reduction_factor

    def __str__(self):
        return '{} ({}-{} nm)'.format(self.name, un.uround(self.wavelength_min, un.nm, 3), un.uround(self.wavelength_max, un.nm, 3))

    def __repr__(self):
        return 'BandBlockBeam(wavelength_min = {}, wavelength_max = {}, r = {}'.format(self.wavelength_min, self.wavelength_max, self.reduction_factor)

    def propagate(self, frequencies, amplitudes):
        block_amplitudes = self.reduction_factor * amplitudes
        wavelengths = opt.photon_wavelength_from_frequency(frequencies)

        return np.where(np.greater_equal(wavelengths, self.wavelength_min) * np.less_equal(wavelengths, self.wavelength_max), block_amplitudes, amplitudes)


class ModulateBeam(BeamModifier):
    def __init__(self, frequency_shift = 90 * un.THz, upshift_efficiency = 1e-6, downshift_efficiency = 1e-6):
        super(ModulateBeam, self).__init__(name = 'Modulator')

        self.frequency_shift = frequency_shift
        self.downshift_efficiency = downshift_efficiency
        self.upshift_efficiency = upshift_efficiency

    def __str__(self):
        return '{} ({} THz shift)'.format(self.name, un.uround(self.frequency_shift, un.THz, 3))

    def __repr__(self):
        return 'ModulateBeam(frequency_shift = {}, upshift_efficiency = {}, downshift_efficiency = {}'.format(frequency_shift, upshift_efficiency, downshift_efficiency)

    def propagate(self, frequencies, amplitudes):
        new_amplitudes = np.zeros(np.shape(amplitudes), dtype = np.complex128)
        shift = int(self.frequency_shift / (frequencies[1] - frequencies[0]))

        for ii, amp in enumerate(amplitudes):
            new_amplitudes[ii] += amp

            try:
                new_amplitudes[ii + shift] += amp * self.upshift_efficiency
            except IndexError as e:
                pass

            try:
                new_amplitudes[ii - shift] += amp * self.downshift_efficiency
            except IndexError as e:
                pass

        return new_amplitudes


IFFTResult = collections.namedtuple('IFFTResult', ('time', 'field'))
PulseWidthFitResult = collections.namedtuple('PulseWidthFitResult', ('center', 'sigma', 'prefactor', 'covariance_matrix'))


class ContinuousAmplitudeSpectrumSpecification(core.Specification):
    def __init__(self, name, frequencies, initial_amplitudes, optics, **kwargs):
        super(ContinuousAmplitudeSpectrumSpecification, self).__init__(name, **kwargs)

        self.frequencies = frequencies
        self.initial_amplitudes = initial_amplitudes.astype(np.complex128)

        self.optics = optics

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
        self.amplitudes = spec.initial_amplitudes.copy()

        self.pulse_fits_vs_materials = []

    @property
    def angular_frequencies(self):
        return opt.photon_angular_frequency_from_frequency(self.frequencies)

    @property
    def wavelengths(self):
        return opt.photon_wavelength_from_frequency(self.frequencies)

    @property
    def power(self):
        return np.real(np.abs(self.amplitudes) ** 2)

    def fft(self):
        t = nfft.fftshift(nfft.fftfreq(len(self.frequencies), self.df))

        reference_frequency = self.frequencies[len(self.frequencies) // 2]
        shifted_spectrum = nfft.fftshift(self.amplitudes)

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

        axis.plot(scaled_t, np.real(field) / e_scale, label = r'$E(t)$', color = 'blue')
        axis.plot(scaled_t, np.abs(field) / e_scale, label = r'$\left| E(t) \right|$', color = 'green')
        axis.plot(scaled_t, -np.abs(field) / e_scale, color = 'green')
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
        self.amplitudes *= np.exp(1j * un.twopi * material.length * material.index(self.wavelengths) * self.frequencies / un.c)

    def run_simulation(self, store_intermediate_fits = False, plot_intermediate_electric_fields = False, target_dir = None):
        logger.info('Performing propagation on {} ({})'.format(self.name, self.file_name))

        self.status = 'running'
        logger.debug("{} {} status set to 'running'".format(self.__class__.__name__, self.name))

        if target_dir is None:
            target_dir = os.getcwd()

        if plot_intermediate_electric_fields:
            self.pulse_fits_vs_materials.append(self.plot_electric_field_vs_time(save = True, target_dir = target_dir, name_postfix = '_0of{}'.format(len(self.spec.optics))))

        for ii, optic in enumerate(self.spec.optics):
            if isinstance(optic, opt.Material):
                self.propagate(optic)
            if isinstance(optic, BeamModifier):
                self.amplitudes = optic.propagate(self.frequencies, self.amplitudes)

            if plot_intermediate_electric_fields:
                self.pulse_fits_vs_materials.append(self.plot_electric_field_vs_time(save = True, target_dir = target_dir, name_postfix = '_{}of{}'.format(ii + 1, len(self.spec.optics))))
            elif store_intermediate_fits:
                self.pulse_fits_vs_materials.append(self.fit_pulse()[0])

            logger.debug('Propagated {} through {} ({}/{} optics)'.format(self.name, optic, ii + 1, len(self.spec.optics)))

        self.end_time = dt.datetime.now()
        self.elapsed_time = self.end_time - self.start_time

        self.status = 'finished'
        logger.debug("Simulation status set to 'finished'")
        logger.info('Finished performing propagation on {} ({})'.format(self.name, self.file_name))

    def get_pulse_width_vs_materials(self):
        try:
            out = ['Initial: FWHM = {} fs,  Peak Amplitude = {}'.format(un.uround(math.gaussian_fwhm_from_sigma(self.pulse_fits_vs_materials[0].sigma), un.fsec, 2), self.pulse_fits_vs_materials[0].prefactor)]

            for material, fit in zip(self.spec.optics, self.pulse_fits_vs_materials[1:]):
                out.append('After {}: FWHM = {} fs, Peak Amplitude = {}'.format(material, un.uround(math.gaussian_fwhm_from_sigma(fit.sigma), un.fsec, 2), fit.prefactor))

            return '\n'.join(out)
        except IndexError:
            return 'No Intermediate Pulse Data Stored'