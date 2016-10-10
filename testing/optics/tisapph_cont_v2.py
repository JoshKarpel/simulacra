import os

import scipy.integrate as integ
import scipy.optimize as optim
import numpy.fft as nfft
import matplotlib.pyplot as plt

import compy as cp
from compy.units import *
import compy.optics.core as opt
import compy.optics.dispersion as disp


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


if __name__ == '__main__':
    with cp.utils.Logger(stdout_level = logging.DEBUG, file_logs = True, file_dir = OUT_DIR, file_level = logging.DEBUG) as logger:
        frequencies = np.linspace(100 * THz, 1000 * THz, 2 ** 20)
        materials = [opt.FS(length = 1 * cm)]

        opt.BK7().plot_index_vs_wavelength(200 * nm, 1600 * nm, save = True, target_dir = OUT_DIR)
        opt.FS().plot_index_vs_wavelength(200 * nm, 1600 * nm, save = True, target_dir = OUT_DIR)

        spec = disp.ContinuousAmplitudeSpectrumSpecification.from_power_spectrum_csv('TiSapph', frequencies, materials, 'tisapph_spectrum.txt', total_power = 150 * mW)
        sim = disp.ContinuousAmplitudeSpectrumSimulation(spec)

        sim.plot_power_vs_frequency(save = True, target_dir = OUT_DIR, x_scale = 'THz')
        sim.plot_power_vs_wavelength(save = True, target_dir = OUT_DIR, x_scale = 'nm')

        sim.run_simulation(plot_intermediate_electric_fields = True, target_dir = OUT_DIR)

        logger.info('\n' + sim.get_pulse_width_vs_materials())

