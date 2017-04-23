import logging
import os

import numpy as np
import scipy as sp

import compy as cp

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', stdout_logs = True, stdout_level = logging.DEBUG, file_dir = OUT_DIR, file_logs = False) as logger:
        x = np.linspace(0, 10, 500)
        t = np.linspace(0, 10, 1000)


        def f(x, t):
            return np.abs(np.sin(x)) ** t


        def ff(x, t):
            return np.sin(x * t)


        cp.plots.xyt_plot('xyt',
                          x, t, f, ff,
                          line_labels = (r'$ \left| \sin(x) \right|^t $', r'$ \sin(x t) $'),
                          title = 'Kerflagonblargh',
                          x_label = '$x$', y_label = '$\mathcal{H}$',
                          x_unit = 'cm', t_unit = 's',
                          fig_dpi_scale = 3,
                          progress_bar = False,
                          target_dir = OUT_DIR)
