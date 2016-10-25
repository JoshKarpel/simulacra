import logging
import os

import numpy as np

import compy as cp
import dispersion as disp
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger(stdout_level = logging.INFO, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG) as logger:
        f_min = 50 * THz
        f_max = 5000 * THz
        wavelength_min = c / f_max
        wavelength_max = c / f_min

        # wavelength_min = 450 * nm
        # wavelength_max = 1250 * nm
        # f_min = c / wavelength_max
        # f_max = c / wavelength_min

        print('freq min', f_min / THz)
        print('freq max', f_max / THz)
        print('wave min', wavelength_min / nm)
        print('wave max', wavelength_max / nm)

        power = 15
        frequencies = np.linspace(f_min, f_max, 2 ** power)

        methods = ['gaussian', 'spline']

        # optics = [disp.FS(name = 'a', length = 1 * cm),
        #           disp.FS(name = 'b', length = 2 * cm),
        #           disp.FS(name = 'c', length = 3 * cm),
        #           disp.ModulateBeam(frequency_shift = 90 * THz, upshift_efficiency = 1e-4, downshift_efficiency = 1e-4),
        #           disp.BandBlockBeam(reduction_factor = 1e-4)]

        optics = [
            disp.BK7(name = 'Pre-cavity Lenses', length = 4 * 3 * mm),
            disp.FS(name = 'Entering Modulator', length = 1 * inch),
            disp.ModulateBeam(90 * THz, upshift_efficiency = 1e-4, downshift_efficiency = 1e-4),
            disp.FS(name = 'Exiting Modulator', length = 1 * inch),
            disp.BK7(name = 'Post-Cavity Lenses', length = 4 * 3 * mm),
            disp.FS(name = 'Dichroic Mirror', length = np.sqrt(2) * 3.2 * mm),
            disp.FS(name = 'Beamsplitter', length = np.sqrt(2) * 5 * mm),
            disp.BandBlockBeam(reduction_factor = 10 ** -4)
        ]

        for method in methods:
            spec = disp.ContinuousAmplitudeSpectrumSpecification.from_power_spectrum_csv('{}'.format(method),
                                                                                         frequencies, optics,
                                                                                         "tisapph_spectrum2.txt", total_power = 150 * mW,
                                                                                         fit = method,
                                                                                         plot_fit = False, target_dir = os.path.join(OUT_DIR))

            sim = disp.ContinuousAmplitudeSpectrumSimulation(spec)

            sim.run_simulation()

            sim.plot_gdd_vs_wavelength(overlay_power_spectrum = True, target_dir = OUT_DIR,
                                       x_lower_lim = 550 * nm, x_upper_lim = 1150 * nm)
