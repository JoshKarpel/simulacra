import os

import compy as cp
from compy.units import *
import compy.optics.dispersion as disp


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


if __name__ == '__main__':
    bk = disp.Glass(name = 'BK7', length = 1 * cm)
    wavelength_min = .2 * um
    wavelength_max = 1.6 * um

    bk.plot_index_vs_wavelength(wavelength_min, wavelength_max, save = True, target_dir = OUT_DIR, img_format = 'pdf')

    modes = [disp.Mode(center_frequency = n * 100 * THz) for n in range(1, 6)]
    beam = disp.Beam(*modes)

    print(beam)
    print(repr(beam))

    beam.propagate(bk)

    beam = disp.modulate_beam(beam)

    beam.propagate(bk)

    print(beam)
    print(repr(beam))

