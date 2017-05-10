import logging
import os

import numpy as np

import compy as cp
from compy.units import *

import matplotlib.pyplot as plt


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.LogManager('compy', stdout_logs = True, stdout_level = logging.DEBUG, file_dir = OUT_DIR, file_logs = False) as logger:
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)

        x_mesh, y_mesh = np.meshgrid(x, y, indexing = 'ij')

        t = np.linspace(0, 30, 900) * pi


        def z(x_mesh, y_mesh, t):
            return np.sin(x_mesh ** 2 + y_mesh ** 2 + t)


        for cmap_name in ('viridis', 'magma', 'inferno', 'plasma', 'seismic', 'PiYG', 'PRGn', 'Spectral'):
            cp.plots.xyzt_plot(f'xyzt_{cmap_name}',
                               x_mesh, y_mesh, t, z,
                               x_label = r'$x$', x_unit = 'cm',
                               y_label = r'$y$', y_unit = 'cm',
                               z_unit = 'rad', z_lower_limit = -1, z_upper_limit = 1,
                               t_unit = 's',
                               title = 'wiggle wobble bobble',
                               colormap = plt.get_cmap(cmap_name),
                               target_dir = OUT_DIR,
                               )
