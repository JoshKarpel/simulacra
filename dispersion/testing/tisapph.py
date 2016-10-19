import logging
import os

import numpy as np

import compy as cp
import dispersion as disp
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


def run_sim(spec):
    with cp.utils.Logger(stdout_level = logging.DEBUG) as logger:
        FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
        OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)
        OUT_DIR = os.path.join(OUT_DIR, 'dispersion', spec.name)

        tau_range = 150 * fsec

        sim = disp.ContinuousAmplitudeSpectrumSimulation(spec)

        OUT_DIR_BEFORE = os.path.join(OUT_DIR, 'before')
        # sim.plot_power_vs_frequency(target_dir = OUT_DIR_BEFORE, x_scale = 'THz')
        # sim.plot_power_vs_frequency(target_dir = OUT_DIR_BEFORE, x_scale = 'THz', log_y = True, name_postfix = '_log')
        # sim.plot_power_vs_wavelength(target_dir = OUT_DIR_BEFORE, x_scale = 'nm')
        # sim.plot_power_vs_wavelength(target_dir = OUT_DIR_BEFORE, x_scale = 'nm', log_y = True, name_postfix = '_log')
        sim.plot_electric_field_vs_time(find_center = True, target_dir = OUT_DIR_BEFORE)
        sim.michelson_autocorrelation(tau_range = tau_range, target_dir = OUT_DIR_BEFORE)
        sim.intensity_autocorrelation(tau_range = tau_range, target_dir = OUT_DIR_BEFORE)
        # sim.intensity_autocorrelation_v2(tau_range = tau_range, target_dir = OUT_DIR_BEFORE)
        sim.interferometric_autocorrelation(tau_range = tau_range, target_dir = OUT_DIR_BEFORE)

        sim.run_simulation()

        OUT_DIR_AFTER = os.path.join(OUT_DIR, 'after')
        # sim.plot_power_vs_frequency(target_dir = OUT_DIR_AFTER, x_scale = 'THz')
        # sim.plot_power_vs_frequency(target_dir = OUT_DIR_AFTER, x_scale = 'THz', log_y = True, name_postfix = '_log')
        # sim.plot_power_vs_wavelength(target_dir = OUT_DIR_AFTER, x_scale = 'nm')
        # sim.plot_power_vs_wavelength(target_dir = OUT_DIR_AFTER, x_scale = 'nm', log_y = True, name_postfix = '_log')
        sim.plot_electric_field_vs_time(find_center = True, target_dir = OUT_DIR_AFTER)
        sim.michelson_autocorrelation(tau_range = tau_range, target_dir = OUT_DIR_AFTER)
        sim.intensity_autocorrelation(tau_range = tau_range, target_dir = OUT_DIR_AFTER)
        # sim.intensity_autocorrelation_v2(tau_range = tau_range, target_dir = OUT_DIR_AFTER)
        sim.interferometric_autocorrelation(tau_range = tau_range, target_dir = OUT_DIR_AFTER)


if __name__ == '__main__':
    with cp.utils.Logger(stdout_level = logging.DEBUG, file_logs = True, file_dir = OUT_DIR, file_level = logging.DEBUG) as logger:

        f_min = 50 * THz
        f_max = 5000 * THz
        power = 16
        wavelength_min = c / f_max
        wavelength_max = c / f_min
        frequencies = np.linspace(f_min, f_max, 2 ** power)
        optics = [disp.FS(length = 1 * cm)]
        # optics = [disp.FS(length = 1 * cm), disp.ModulateBeam(90 * THz), disp.BandBlockBeam()]

        # disp.BK7().plot_index_vs_wavelength(wavelength_min, wavelength_max, target_dir = OUT_DIR)
        # disp.FS().plot_index_vs_wavelength(wavelength_min, wavelength_max, target_dir = OUT_DIR)

        specs = []
        for fit_method in ('gaussian', 'spline'):
            for bandblock in (-2, -3, -4, -5, -6):
                optics = [
                    disp.BK7(name = '4 lenses 1', length = 4 * 3 * mm),
                    disp.FS(name = 'cavity 1', length = 1 * inch),
                    disp.ModulateBeam(90 * THz, upshift_efficiency = 1e-4, downshift_efficiency = 1e-6),
                    disp.FS(name = 'cavity 2', length = 1 * inch),
                    disp.BK7(name = '4 lenses 2', length = 4 * 3 * mm),
                    disp.FS(name = 'dichroic mirror', length = np.sqrt(2) * 3.2 * mm),
                    disp.FS(name = 'beamsplitter', length = np.sqrt(2) * 5 * mm),
                    disp.BandBlockBeam(reduction_factor = 10 ** bandblock)
                ]

                name = 'TiSapph_FitMethod={}_UpshiftEfficiencyExp=-4_BlockingPowerExp={}'.format(fit_method, bandblock)

                spec = disp.ContinuousAmplitudeSpectrumSpecification.from_power_spectrum_csv(name,
                                                                                             frequencies, optics,
                                                                                             "tisapph_spectrum2.txt", total_power = 150 * mW,
                                                                                             fit = fit_method,
                                                                                             plot_fit = True, target_dir = os.path.join(OUT_DIR, 'dispersion', name))

                specs.append(spec)

        cp.utils.multi_map(run_sim, specs, processes = 2)

        # sim = disp.ContinuousAmplitudeSpectrumSimulation(spec)
        #
        # # sim.plot_power_vs_frequency(target_dir = OUT_DIR, x_scale = 'THz', name_postfix = '_before')
        # # sim.plot_power_vs_frequency(target_dir = OUT_DIR, x_scale = 'THz', log_y = True, name_postfix = '_before_log')
        # # sim.plot_power_vs_wavelength(target_dir = OUT_DIR, x_scale = 'nm', name_postfix = '_before')
        # # sim.plot_power_vs_wavelength(target_dir = OUT_DIR, x_scale = 'nm', log_y = True, name_postfix = '_before_log')
        #
        # tau_range = 150 * fsec
        # sim.plot_electric_field_vs_time(target_dir = OUT_DIR, find_center = True)
        # sim.michelson_autocorrelation(tau_range = tau_range, target_dir = OUT_DIR)
        # sim.intensity_autocorrelation(tau_range = tau_range, target_dir = OUT_DIR)
        # sim.intensity_autocorrelation_v2(tau_range = tau_range, target_dir = OUT_DIR)
        # sim.interferometric_autocorrelation(tau_range = tau_range, target_dir = OUT_DIR)
        #
        # sim.plot_autocorrelation(target_dir = OUT_DIR, name_postfix = '_before')
        #
        # sim.run_simulation(plot_intermediate_electric_fields = True, target_dir = OUT_DIR)
        #
        # sim.plot_electric_field_vs_time(target_dir = OUT_DIR, find_center = False)
        #
        # sim.plot_power_vs_frequency(target_dir = OUT_DIR, x_scale = 'THz', name_postfix = '_after')
        # sim.plot_power_vs_wavelength(target_dir = OUT_DIR, x_scale = 'nm', name_postfix = '_after')
        #
        # logger.info('\n' + sim.get_pulse_width_vs_materials() + '\n')
        #
        # sim.plot_autocorrelation(target_dir = OUT_DIR, name_postfix = '_after')
        # sim.plot_gdd_vs_wavelength(target_dir = OUT_DIR)
