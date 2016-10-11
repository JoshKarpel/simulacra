import os

import compy as cp
from compy.units import *
import compy.optics.core as opt
import compy.optics.dispersion as disp


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


if __name__ == '__main__':
    with cp.utils.Logger(stdout_level = logging.DEBUG, file_logs = True, file_dir = OUT_DIR, file_level = logging.DEBUG) as logger:
        frequencies = np.linspace(100 * THz, 1000 * THz, 2 ** 20)
        optics = [opt.FS(length = 1 * cm), disp.ModulateBeam(90 * THz), disp.BandBlockBeam()]

        opt.BK7().plot_index_vs_wavelength(200 * nm, 1600 * nm, save = True, target_dir = OUT_DIR)
        opt.FS().plot_index_vs_wavelength(200 * nm, 1600 * nm, save = True, target_dir = OUT_DIR)

        spec = disp.ContinuousAmplitudeSpectrumSpecification.from_power_spectrum_csv('TiSapph', frequencies, optics, 'tisapph_spectrum.txt', total_power = 150 * mW)
        sim = disp.ContinuousAmplitudeSpectrumSimulation(spec)

        sim.plot_power_vs_frequency(save = True, target_dir = OUT_DIR, x_scale = 'THz')
        sim.plot_power_vs_wavelength(save = True, target_dir = OUT_DIR, x_scale = 'nm')

        sim.run_simulation(plot_intermediate_electric_fields = True, target_dir = OUT_DIR)

        # sim.plot_electric_field_vs_time(save = True, target_dir = OUT_DIR)

        sim.plot_power_vs_frequency(save = True, target_dir = OUT_DIR, x_scale = 'THz', name_postfix = '_after')
        sim.plot_power_vs_wavelength(save = True, target_dir = OUT_DIR, x_scale = 'nm', name_postfix = '_after')

        logger.info('\n' + sim.get_pulse_width_vs_materials())

