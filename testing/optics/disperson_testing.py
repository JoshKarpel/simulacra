import os

import compy as cp
from compy.units import *
import compy.optics.dispersion as disp


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


if __name__ == '__main__':
    with cp.utils.Logger(stdout_level = logging.DEBUG) as logger:
        bk = disp.Glass(name = 'BK7', length = 1 * cm)
        wavelength_min = .2 * um
        wavelength_max = 1.6 * um

        bk.plot_index_vs_wavelength(wavelength_min, wavelength_max, save = True, target_dir = OUT_DIR, img_format = 'pdf')

        f_min = c / (810 * nm)
        f_max = c / (770 * nm)
        print(f_min / THz, f_max / THz)
        modes = [disp.Mode(center_frequency = f) for f in np.linspace(f_min, f_max, 1000)]
        beam = disp.Beam(*modes)

        # print(beam)
        # print(repr(beam))

        beam.plot_field_vs_time(save = True, target_dir = OUT_DIR, name_postfix = 'pre')
        # beam.plot_fft(save = True, target_dir = OUT_DIR, name_postfix = 'pre')

        logger.info('prop')

        with cp.utils.Timer() as timer:
            logger.info('hello?')
            beam.propagate(bk)
            beam = disp.modulate_beam(beam)
            beam.propagate(bk)
            beam = disp.bandblock_beam(beam, 700 * nm, 900 * nm, filter_by = 1e-6)

        print(timer)

        # print(beam)
        # print(repr(beam))

        logger.info('done')

        with cp.utils.Timer() as timer:
            beam.plot_field_vs_time(save = True, target_dir = OUT_DIR, name_postfix = 'post')
            # beam.plot_fft(save = True, target_dir = OUT_DIR, name_postfix = 'post')

        print(timer)

