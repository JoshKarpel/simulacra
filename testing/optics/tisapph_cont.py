import os

import scipy.integrate as integ
import scipy.optimize as optim
import numpy.fft as nfft
import matplotlib.pyplot as plt

import compy as cp
from compy.units import *
import compy.optics.dispersion as disp


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


if __name__ == '__main__':
    with cp.utils.Logger(stdout_level = logging.DEBUG) as logger:
        # bk = disp.Glass(name = 'BK7', length = 5 * cm)
        # wavelength_min = .2 * um
        # wavelength_max = 1.6 * um
        #
        # bk.plot_index_vs_wavelength(wavelength_min, wavelength_max, save = True, target_dir = OUT_DIR, img_format = 'pdf')

        wavelengths, dBm = np.loadtxt('tisapph_spectrum.txt', delimiter = ',', unpack = True, skiprows = 1)
        wavelengths = wavelengths * nm
        power_per_bin = 1 * mW * (10 ** (dBm / 10))
        bin_size = wavelengths[1] - wavelengths[0]
        print(wavelengths / nm)
        print(bin_size / nm)
        print(dBm)

        total_power = integ.simps(power_per_bin, wavelengths)
        print(total_power)
        print(total_power / mW)

        real_power = 150 * mW
        adjustment_ratio = real_power / total_power
        print(adjustment_ratio)

        power_per_bin *= adjustment_ratio
        print(integ.simps(power_per_bin, wavelengths))

        scaled_dBm = 10 * np.log10(power_per_bin / mW)

        cp.utils.xy_plot(wavelengths, [dBm], x_scale = 'nm',
                         title = r'Ti:Sapph Output Spectrum', x_label = r'Wavelength', y_label = r'Power per {} nm (dBm)'.format(uround(bin_size, nm, 1)),
                         save = True, target_dir = OUT_DIR, name = 'tisapph_spectrum__dbm')
        cp.utils.xy_plot(wavelengths, [power_per_bin], x_scale = 'nm', y_scale = 'mW',
                         title = r'Ti:Sapph Output Spectrum', x_label = r'Wavelength', y_label = r'Power per {} nm'.format(uround(bin_size, nm, 1)),
                         save = True, target_dir = OUT_DIR, name = 'tisapph_spectrum__power')
        cp.utils.xy_plot(wavelengths, [power_per_bin], x_scale = 'nm', y_scale = 'mW',
                         title = r'Ti:Sapph Output Spectrum', x_label = r'Wavelength', y_label = r'Power per {} nm'.format(uround(bin_size, nm, 1)),
                         log_y = True,
                         save = True, target_dir = OUT_DIR, name = 'tisapph_spectrum__power_log')

        def gaussian(x, center, sigma, prefactor):
            return prefactor * np.exp(-((x - center) / sigma) ** 2)

        guesses = np.array([800 * nm, 40 * nm, np.max(power_per_bin)])
        popt, pcov = optim.curve_fit(gaussian, wavelengths, power_per_bin, p0 = guesses)

        print(popt, pcov)

        fitted = gaussian(wavelengths, *popt)

        cp.utils.xy_plot(wavelengths, [power_per_bin, fitted], legends = ['Measured', 'Fitted'], x_scale = 'nm', y_scale = 'mW',
                         title = r'Fitted Ti:Sapph Output Spectrum', x_label = r'Wavelength', y_label = r'Power per {} nm'.format(uround(bin_size, nm, 1)),
                         save = True, target_dir = OUT_DIR, name = 'tisapph_spectrum__power_fit')

        cp.utils.xy_plot(wavelengths, [power_per_bin, fitted], legends = ['Measured', 'Fitted'], x_scale = 'nm', y_scale = 'mW',
                         title = r'Fitted Ti:Sapph Output Spectrum', x_label = r'Wavelength', y_label = r'Power per {} nm'.format(uround(bin_size, nm, 1)),
                         log_y = True,
                         save = True, target_dir = OUT_DIR, name = 'tisapph_spectrum__power_log_fit')

        # print(integ.simps(fitted, wavelengths))

        def power_spectrum(frequency, center, sigma, prefactor):
            return gaussian(c / frequency, center, sigma, prefactor)

        frequencies = np.linspace(100 * THz, 1000 * THz, 2 ** 20)
        df = frequencies[1] - frequencies[0]
        y = power_spectrum(frequencies, *popt)
        # print(len(frequencies))
        # print(len(y))

        cp.utils.xy_plot(frequencies, [y], x_scale = 'THz',
                         title = r'Fitted Ti:Sapph Output Spectrum', x_label = r'Frequency', y_label = r'Power Spectral Density',
                         save = True, target_dir = OUT_DIR, name = 'tisapph_frequencyspectrum__power_log_fit')

        # new_y = np.zeros(np.shape(y))
        # shift = int(90 * THz / df)
        # print(shift)
        #
        # for ii, entry in enumerate(y):
        #     new_y[ii] += entry
        #
        #     try:
        #         new_y[ii + shift] += entry
        #     except IndexError as e:
        #         # print('failed at {}'.format(ii))
        #         pass
        #
        #     try:
        #         new_y[ii - shift] += entry
        #     except IndexError as e:
        #         # print('failed at {}'.format(ii))
        #         pass
        #
        # cp.utils.xy_plot(frequencies, [new_y], legends = ['with sidebands'], x_scale = 'THz',
        #                  title = r'Fitted Ti:Sapph Output Spectrum', x_label = r'Frequency', y_label = r'Power Spectral Density',
        #                  save = True, target_dir = OUT_DIR, name = 'tisapph_frequencyspectrum_sidebands')

        # print(len(y) // 2)
        # f = frequencies - frequencies[len(frequencies) // 2]
        # print(f)
        # print(nfft.fftshift(f))
        ref = frequencies[len(frequencies) // 2]
        shifted_frequencies = nfft.fftshift(frequencies - ref).astype(np.complex128)
        # print(shifted_frequencies)

        y = np.sqrt(y)  # important! go from power spectrum to amplitude spectrum

        shifted_spectrum = nfft.fftshift(y).astype(np.complex128)

        t = nfft.fftshift(nfft.fftfreq(len(shifted_frequencies), df))

        time_domain = nfft.fftshift(nfft.ifft(shifted_spectrum, norm = 'ortho')) * np.exp(1j * ref * t)

        cp.utils.xy_plot(t, [np.real(time_domain), np.abs(time_domain)], legends = ['Field', 'Envelope'],
                         x_scale = 'fs', x_range = 200 * fsec,
                         title = r'Ti:Sapph Output', x_label = r'$t$', y_label = r'$E(t)$',
                         save = True, target_dir = OUT_DIR, name = 'tisapph_time_domain')

        center = t[np.argmax(time_domain)]
        amp_max = np.real(np.abs(np.max(time_domain)))
        print(center / fsec)
        print('amp', amp_max)
        popt, pcov = optim.curve_fit(gaussian, t, np.real(np.abs(time_domain)).astype(np.float64),
                                     p0 = [0, 40 * fsec, amp_max],
                                     bounds = ([-10 * fsec, 10 * fsec, -np.inf], [10 * fsec, 100 * fsec, np.inf]))
        print(popt, pcov)
        print('t center before', popt[0] / fsec)
        print('pw before', popt[1] / fsec)
        print('amp before', popt[2])

        ### PROPAGATE ###
        print('PROPAGATE')

        mat = disp.Glass('fs', length = 5 * cm, b = disp.fs_b, c = disp.fs_c)
        # mat = disp.Glass('bk7', length = 100 * cm, b = disp.bk7_b, c = disp.bk7_c)

        index_vs_frequency = np.zeros(len(frequencies), dtype = np.complex128)
        for ii, f in enumerate(frequencies):
            # print(f, uround(c / f, nm, 2), mat.index(c / f))
            index_vs_frequency[ii] = mat.index(c / f)

        index_vs_frequency = nfft.fftshift(index_vs_frequency)

        cp.utils.xy_plot(shifted_frequencies, [index_vs_frequency], x_scale = 'THz',
                         save = True, target_dir = OUT_DIR, name = 'index_vs_frequency')

        # print(index_vs_frequency.dtype)
        # print(shifted_frequencies.dtype)
        # print(shifted_spectrum.dtype)

        phase = (twopi * mat.length * index_vs_frequency * shifted_frequencies / c)
        print(phase)

        shifted_spectrum *= np.exp(1j * phase)

        time_domain = nfft.fftshift(nfft.ifft(shifted_spectrum, norm = 'ortho')) * np.exp(1j * ref * t)

        center = t[np.argmax(time_domain)]
        print(center / fsec)
        amp_max = np.real(np.abs(np.max(time_domain)))
        print('amp', amp_max)
        print(t.dtype)
        print(time_domain.astype(np.float64).dtype)
        popt, pcov = optim.curve_fit(gaussian, t, np.real(np.abs(time_domain)).astype(np.float64), p0 = [center, 30 * fsec, amp_max],
                                     bounds = ([-100 * fsec + center, 10 * fsec, -np.inf], [center + 100 * fsec, 100 * fsec, np.inf])
                                     )
        print(popt, pcov)
        print('t center after', popt[0] / fsec)
        print('pw after', popt[1] / fsec)
        print('amp after', popt[2])

        cp.utils.xy_plot(t - center, [np.real(time_domain), np.abs(time_domain)], legends = ['Field', 'Envelope'],
                         x_scale = 'fs',
                         x_center = 0, x_range = 200 * fsec,
                         title = r'Ti:Sapph Output', x_label = r'$t$', y_label = r'$E(t)$',
                         save = True, target_dir = OUT_DIR, name = 'tisapph_time_domain__PROP')


        #
        # f_min = c / (810 * nm)
        # f_max = c / (770 * nm)
        # print(f_min / THz, f_max / THz)
        # modes = [disp.Mode(frequency = f) for f in np.linspace(f_min, f_max, 100)]
        # beam = disp.Beam(*modes)
        #
        # # print(beam)
        # # print(repr(beam))
        #
        # # beam.plot_field_vs_time(save = True, target_dir = OUT_DIR, name_postfix = 'pre')
        # # beam.plot_fft(save = True, target_dir = OUT_DIR, name_postfix = 'pre')
        #
        # logger.info('prop')
        #
        # with cp.utils.Timer() as timer:
        #     # beam.propagate(bk)
        #     beam = disp.modulate_beam(beam, frequency_shift = 30 * THz, downshift_efficiency = .33, upshift_efficiency = .33)
        #     # beam.propagate(bk)
        #     # beam = disp.bandblock_beam(beam, 700 * nm, 900 * nm, filter_by = 1e-6)
        #
        # print(timer)
        #
        # # print(beam)
        # # print(repr(beam))
        #
        # logger.info('done')
        #
        # with cp.utils.Timer() as timer:
        #     beam.plot_field_vs_time(save = True, target_dir = OUT_DIR, name_postfix = 'post')
        #     # beam.plot_fft(save = True, target_dir = OUT_DIR, name_postfix = 'post')
        #
        # print(timer)

