import logging
import os

import numpy as np

import compy as cp
import plots
from units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.LogManager(stdout_level = logging.DEBUG) as logger:
        with open('after_cavity.csv') as f:
            _, _, position, delay, signal = np.loadtxt(f, skiprows = 1, unpack = True, delimiter = ',')

            print(position)
            print(delay)
            print(signal)

            delay *= fsec

            fmts = ('png', 'svg')
            for fmt in fmts:
                plots.xy_plot('after_cavity',
                              delay, signal,
                              x_label = r'Time Delay $\tau$', x_scale = 'fs',
                              y_label = r'SHG Power (arb. units)',
                              font_size_axis_labels = 20,
                              img_format = fmt, target_dir = OUT_DIR)

                plots.xy_plot('after_cavity_with_title',
                              delay, signal,
                              x_label = r'Time Delay $\tau$', x_scale = 'fs',
                              y_label = r'SHG Power (arb. units)',
                              font_size_axis_labels = 26,
                              font_size_title = 26,
                              font_size_tick_labels = 16,
                              title = r'Measured Interferometric Autocorrelation (No Modulation)',
                              aspect_ratio = 1.8, img_format = fmt, target_dir = OUT_DIR)
