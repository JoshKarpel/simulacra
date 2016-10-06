import os

import scipy.integrate as integ
import scipy.optimize as optim
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

        # plt.plot(wavelengths / nm, power_per_wavelength)
        # plt.show()

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

        def gaussian(wavelength, center, sigma, prefactor):
            return (prefactor / (np.sqrt(twopi) * sigma)) * np.exp(-0.5 * (((wavelength - center) / sigma) ** 2))

        popt, pcov = optim.curve_fit(gaussian, wavelengths, power_per_bin, p0 = [800 * nm, 40 * nm, np.max(power_per_bin)])

        print(popt, pcov)

        fitted = gaussian(wavelengths, *popt)

        cp.utils.xy_plot(wavelengths, [power_per_bin, fitted], legends = ['Measured', 'Fitted'], x_scale = 'nm', y_scale = 'mW',
                         title = r'Fitted Ti:Sapph Output Spectrum', x_label = r'Wavelength', y_label = r'Power per {} nm'.format(uround(bin_size, nm, 1)),
                         save = True, target_dir = OUT_DIR, name = 'tisapph_spectrum__power_fit')

        cp.utils.xy_plot(wavelengths, [power_per_bin, fitted], legends = ['Measured', 'Fitted'], x_scale = 'nm', y_scale = 'mW',
                         title = r'Fitted Ti:Sapph Output Spectrum', x_label = r'Wavelength', y_label = r'Power per {} nm'.format(uround(bin_size, nm, 1)),
                         log_y = True,
                         save = True, target_dir = OUT_DIR, name = 'tisapph_spectrum__power_log_fit')

        print(integ.simps(fitted, wavelengths))


        f_min = c / (810 * nm)
        f_max = c / (770 * nm)
        print(f_min / THz, f_max / THz)
        modes = [disp.Mode(frequency = f) for f in np.linspace(f_min, f_max, 100)]
        beam = disp.Beam(*modes)

        # print(beam)
        # print(repr(beam))

        # beam.plot_field_vs_time(save = True, target_dir = OUT_DIR, name_postfix = 'pre')
        # beam.plot_fft(save = True, target_dir = OUT_DIR, name_postfix = 'pre')

        logger.info('prop')

        with cp.utils.Timer() as timer:
            # beam.propagate(bk)
            beam = disp.modulate_beam(beam, frequency_shift = 30 * THz, downshift_efficiency = .33, upshift_efficiency = .33)
            # beam.propagate(bk)
            # beam = disp.bandblock_beam(beam, 700 * nm, 900 * nm, filter_by = 1e-6)

        print(timer)

        # print(beam)
        # print(repr(beam))

        logger.info('done')

        with cp.utils.Timer() as timer:
            beam.plot_field_vs_time(save = True, target_dir = OUT_DIR, name_postfix = 'post')
            # beam.plot_fft(save = True, target_dir = OUT_DIR, name_postfix = 'post')

        print(timer)

