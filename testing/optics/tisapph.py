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
        lens_4mm = disp.Glass(name = 'lens_BK7_4mm', length = 4 * mm, b = disp.bk7_b, c = disp.bk7_c)
        pickoff = disp.Glass(name = 'pickoff_FS_Hin', length = np.sqrt(2) * 0.5 * inch, b = disp.fs_b, c = disp.fs_c)
        cavity_window = disp.Glass(name = 'window_FS_Qin', length = 0.25 * inch, b = disp.fs_b, c = disp.fs_c)
        cavity_mirror = disp.Glass(name = 'mirror_FS_Qin', length = 0.25 * inch, b = disp.fs_b, c = disp.fs_c)
        dichroic_mirror = disp.Glass(name = 'dichroic_mirror_FS_3.2mm', length = np.sqrt(2) * 3.2 * mm, b = disp.fs_b, c = disp.fs_c)
        bandblock_mirror = disp.Glass(name = 'bandblock_BK7_3mm', length = 3 * mm, b = disp.bk7_b, c = disp.bk7_c)
        beamsplitter = disp.Glass(name = 'beamsplitter_FS_5mm', length = np.sqrt(2) * 5 * mm, b = disp.fs_b, c = disp.fs_c)

        wavelength_min = .2 * um
        wavelength_max = 1.6 * um

        # cavity_mirror.plot_index_vs_wavelength(wavelength_min, wavelength_max, save = True, target_dir = OUT_DIR, img_format = 'pdf')
        # lens_4mm.plot_index_vs_wavelength(wavelength_min, wavelength_max, save = True, target_dir = OUT_DIR, img_format = 'pdf')

        f_min = c / (810 * nm)
        f_max = c / (770 * nm)
        print(f_min / THz, f_max / THz)
        num_modes = 100
        modes = [disp.Mode(frequency = f) for f in np.linspace(f_min, f_max, num_modes)]
        beam = disp.Beam(*modes)

        # beam.plot_field_vs_time(save = True, target_dir = OUT_DIR, name_postfix = 'pre_{}'.format(num_modes))

        d = .1

        with cp.utils.Timer() as timer:
            # beam.propagate(lens_4mm)
            # beam.propagate(lens_4mm)
            # beam.propagate(pickoff)
            # beam.propagate(cavity_window)
            # beam.propagate(cavity_mirror)
            # beam = disp.modulate_beam(beam, frequency_shift = 90 * THz, downshift_efficiency = 1e-6, upshift_efficiency = 1e-6)
            # beam.propagate(cavity_mirror)
            # beam.propagate(cavity_window)
            # beam.propagate(dichroic_mirror)
            # beam.propagate(bandblock_mirror)
            # beam = disp.bandblock_beam(beam, 700 * nm, 900 * nm, filter_by = 1e-6)
            # beam.propagate(lens_4mm)
            # beam.propagate(lens_4mm)
            # beam.propagate(lens_4mm)
            # beam.propagate(lens_4mm)
            # beam.propagate(beamsplitter)

            beam.propagate(disp.Glass(name = 'fs_flat', length = d * cm, b = disp.fs_b, c = disp.fs_c))

        print(timer)

        with cp.utils.Timer() as timer:
            beam.plot_field_vs_time(time_initial = -10 * psec, time_final = 10 * psec, save = True, target_dir = OUT_DIR, name_postfix = 'post_fs_{}modes_{}cm'.format(num_modes, d))

        print(timer)

