import logging
import os

import numpy as np

import compy as cp
import dispersion as disp
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger(stdout_level = logging.DEBUG, file_logs = True, file_dir = OUT_DIR, file_level = logging.DEBUG) as logger:
        OUT_DIR = os.path.join(OUT_DIR, 'no_dispersion__100')

        f_min = 50 * THz
        f_max = 5000 * THz
        power = 21
        wavelength_min = c / f_max
        wavelength_max = c / f_min
        frequencies = np.linspace(f_min, f_max, 2 ** power)
        optics = [disp.FS(length = 1 * cm)]
        # optics = [disp.FS(length = 1 * cm), disp.ModulateBeam(90 * THz), disp.BandBlockBeam()]
        optics = [
            # disp.BK7(name = '4 lenses 1', length = 4 * 3 * mm),
            # disp.FS(name = 'cavity 1', length = 1 * inch),
            disp.ModulateBeam(90 * THz, upshift_efficiency = 1e-4, downshift_efficiency = 1e-6),
            # disp.FS(name = 'cavity 2', length = 1 * inch),
            # disp.BK7(name = '4 lenses 2', length = 4 * 3 * mm),
            # disp.FS(name = 'dichroic mirror', length = np.sqrt(2) * 3.2 * mm),
            # disp.FS(name = 'beamsplitter', length = np.sqrt(2) * 5 * mm),
            disp.BandBlockBeam(reduction_factor = 1e-2)
        ]

        disp.BK7().plot_index_vs_wavelength(wavelength_min, wavelength_max, target_dir = OUT_DIR)
        disp.FS().plot_index_vs_wavelength(wavelength_min, wavelength_max, target_dir = OUT_DIR)

        spec = disp.ContinuousAmplitudeSpectrumSpecification.from_power_spectrum_csv('TiSapph_{}THz_{}pts'.format(uround(f_max - f_min, THz, 0), power), frequencies, optics,
                                                                                     "tisapph_spectrum2.txt", total_power = 150 * mW,
                                                                                     plot_fit = True, target_dir = OUT_DIR)
        sim = disp.ContinuousAmplitudeSpectrumSimulation(spec)

        sim.plot_power_vs_frequency(target_dir = OUT_DIR, x_scale = 'THz', name_postfix = '_before')
        sim.plot_power_vs_wavelength(target_dir = OUT_DIR, x_scale = 'nm', name_postfix = '_before')

        sim.plot_autocorrelation(target_dir = OUT_DIR, name_postfix = '_before')

        sim.run_simulation(plot_intermediate_electric_fields = True, target_dir = OUT_DIR)

        sim.plot_electric_field_vs_time(target_dir = OUT_DIR, find_center = False)

        sim.plot_power_vs_frequency(target_dir = OUT_DIR, x_scale = 'THz', name_postfix = '_after')
        sim.plot_power_vs_wavelength(target_dir = OUT_DIR, x_scale = 'nm', name_postfix = '_after')

        logger.info('\n' + sim.get_pulse_width_vs_materials())

        sim.plot_autocorrelation(target_dir = OUT_DIR, name_postfix = '_after')
        sim.plot_gdd_vs_wavelength(target_dir = OUT_DIR)
