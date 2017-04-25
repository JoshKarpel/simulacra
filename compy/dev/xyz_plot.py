import logging
import os

import numpy as np

import compy as cp
from compy.units import *

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', stdout_logs = True, stdout_level = logging.DEBUG, file_dir = OUT_DIR, file_logs = False) as logger:
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)

        x_mesh, y_mesh = np.meshgrid(x, y, indexing = 'ij')

        z_mesh = np.abs((x_mesh ** 2) + (y_mesh ** 2)) * np.sin(x_mesh * pi)

        for cmap_name in ('viridis', 'magma', 'inferno', 'plasma'):
            cp.plots.xyz_plot(f'xyz_{cmap_name}',
                              x_mesh, y_mesh, z_mesh,
                              colormap = plt.get_cmap(cmap_name),
                              target_dir = OUT_DIR
                              )
