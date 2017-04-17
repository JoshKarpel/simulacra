import logging
import os

import compy as cp
import dispersion as disp
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


if __name__ == '__main__':
    with cp.utils.Logger(stdout_level = logging.DEBUG, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG) as logger:
        # x = np.linspace(0, 1, 100)
        # y = 0.5 * x ** 2
        #
        # cp.utils.xy_plot(x, [y], save = True, target_dir = OUT_DIR, name = 'xy')
        #
        # dydx = cp.math.centered_first_derivative(y, x[1] - x[0])
        #
        # cp.utils.xy_plot(x, [dydx], save = True, target_dir = OUT_DIR, name = 'dydx')
        #
        # # ddyddx = cp.math.centered_second_derivative(y, x[1] - x[0])
        # ddyddx = cp.math.centered_first_derivative(dydx, x[1] - x[0])
        #
        # cp.utils.xy_plot(x, [ddyddx], save = True, target_dir = OUT_DIR, name = 'ddyddx')

        mat = disp.BK7(length = 1 * cm)

        disp.BK7().plot_gvd_vs_wavelength(300 * nm, 2000 * nm, save = True, target_dir = OUT_DIR)
        disp.FS().plot_gvd_vs_wavelength(300 * nm, 2000 * nm, save = True, target_dir = OUT_DIR)

        # angular_frequencies = twopi * np.linspace(100 * THz, 1000 * THz, 1e6)
        # wavelengths = disp.photon_wavelength_from_angular_frequency(angular_frequencies)
        #
        # # wavelengths = np.linspace(500 * nm, 1100 * nm, 1e6)
        # # angular_frequencies = twopi * disp.photon_frequency_from_wavelength(wavelengths)
        #
        # n = mat.index(wavelengths)
        #
        # dw = angular_frequencies[1] - angular_frequencies[0]
        #
        # dn_dw = cp.math.centered_first_derivative(n, dw)
        # ddn_ddw = cp.math.centered_first_derivative(dn_dw, dw)
        #
        # cp.utils.xy_plot(angular_frequencies / twopi, [n], save = True, target_dir = OUT_DIR, name = 'n_vs_w', x_scale = 'THz')
        # cp.utils.xy_plot(angular_frequencies / twopi, [dn_dw], save = True, target_dir = OUT_DIR, name = 'dndw_vs_w', x_scale = 'THz')
        # cp.utils.xy_plot(angular_frequencies / twopi, [ddn_ddw], save = True, target_dir = OUT_DIR, name = 'ddnddw_vs_w', x_scale = 'THz')
        #
        # gvd = ((2 / c) * dn_dw) + ((angular_frequencies / c) * ddn_ddw)
        #
        # cp.utils.xy_plot(angular_frequencies / twopi, [uround(gvd, fsec ** 2 / mm, 10)], save = True, target_dir = OUT_DIR, name = 'gvd_vs_w', x_scale = 'THz')
        # cp.utils.xy_plot(wavelengths, [uround(gvd, fsec ** 2 / mm, 10)], save = True, target_dir = OUT_DIR, name = 'gvd_vs_wavelength', x_scale = 'nm')


