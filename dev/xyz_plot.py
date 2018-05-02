import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *

import matplotlib.pyplot as plt


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', stdout_logs = True, stdout_level = logging.DEBUG, file_dir = OUT_DIR, file_logs = False) as logger:
        x = np.linspace(-5, 5, 200)
        y = np.linspace(-5, 5, 200)

        x_mesh, y_mesh = np.meshgrid(x, y, indexing = 'ij')

        z_mesh = np.sin(x_mesh * pi) * np.sin(y_mesh * pi)

        for cmap_name in ('viridis', 'magma', 'inferno', 'plasma', 'seismic', 'PiYG', 'PRGn', 'Spectral'):
            si.vis.xyz_plot(
                f'xyz_{cmap_name}',
                x_mesh, y_mesh, z_mesh,
                colormap = plt.get_cmap(cmap_name),
                contours = [.5],
                shading = si.vis.ColormapShader.FLAT,
                target_dir = OUT_DIR,
            )

        # for cmap_name in ('viridis', 'magma', 'inferno', 'plasma', 'seismic', 'PiYG', 'PRGn', 'Spectral'):
        #     si.vis.xyz_plot(
        #         f'xyz_{cmap_name}__log',
        #         x_mesh, y_mesh, z_mesh,
        #         colormap = plt.get_cmap(cmap_name),
        #         contours = [0],
        #         z_log_axis = True,
        #         aspect_ratio = .8,
        #         target_dir = OUT_DIR,
        #         sym_log_norm_epsilon = 1e-2,
        #     )
