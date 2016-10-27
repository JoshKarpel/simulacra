import logging
import os

import numpy as np

import compy as cp
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger(stdout_level = logging.DEBUG) as logger:
        with open('after_cavity.csv') as f:
            _, _, position, delay, signal = np.loadtxt(f, skiprows = 1, unpack = True, delimiter = ',')

            print(position)
            print(delay)
            print(signal)

            delay *= fsec

            fmts = ('png', 'svg')
            for fmt in fmts:
                cp.utils.xy_plot(delay, signal,
                                 x_label = r'Time Delay $\tau$', x_scale = 'fs',
                                 y_label = r'SHG Power (arb. units)',
                                 label_size = 20,
                                 name = 'after_cavity', img_format = fmt, target_dir = OUT_DIR)

                cp.utils.xy_plot(delay, signal,
                                 x_label = r'Time Delay $\tau$', x_scale = 'fs',
                                 y_label = r'SHG Power (arb. units)',
                                 label_size = 26,
                                 title_size = 26,
                                 unit_size = 16,
                                 title = r'Measured Interferometric Autocorrelation (No Modulation)',
                                 aspect_ratio = 1.8,
                                 name = 'after_cavity_with_title', img_format = fmt, target_dir = OUT_DIR)
