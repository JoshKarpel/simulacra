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

        # z_mesh = 1j * np.sin(y_mesh)
        # z_mesh = np.zeros(np.shape(x_mesh))
        z_mesh = x_mesh + (1j * y_mesh)

        rich = si.plots.RichardsonColormap()
        for equator_mag in (.2, 1, 5):
            for shading in ('flat', 'gouraud'):
                si.plots.xyz_plot(f'richardson_xyz_eq={equator_mag}_{shading}',
                                  x_mesh, y_mesh, z_mesh,
                                  x_unit = 'rad', y_unit = 'rad',
                                  shading = shading,
                                  colormap = plt.get_cmap('richardson'),
                                  richardson_equator_magnitude = equator_mag,
                                  target_dir = OUT_DIR,
                                  show_colorbar = False,
                                  aspect_ratio = 1,
                                  )


        def z(x_mesh, y_mesh, t):
            return z_mesh * np.exp(1j * t)


        t = np.linspace(0, 10, 900) * pi

        for equator_mag in (.2, 1, 5):
            for shading in ('flat', 'gouraud'):
                si.plots.xyzt_plot(f'richardson_xyzt_eq={equator_mag}_{shading}',
                                   x_mesh, y_mesh, t, z,
                                   x_label = r'$x$', y_label = r'$y$',
                                   x_unit = 'rad', y_unit = 'rad',
                                   title = r'$(x + iy) e^{i t}$',
                                   shading = shading,
                                   colormap = plt.get_cmap('richardson'),
                                   richardson_equator_magnitude = equator_mag,
                                   target_dir = OUT_DIR,
                                   show_colorbar = False,
                                   aspect_ratio = 1,
                                   )


        def z2(x_mesh, y_mesh, t):
            return z_mesh * np.exp(1j * t) * np.sin(x_mesh ** 2 + y_mesh ** 2 + t)


        for equator_mag in (.2, 1, 5):
            for shading in ('flat', 'gouraud'):
                si.plots.xyzt_plot(f'richardson_xyzt2_eq={equator_mag}_{shading}',
                                   x_mesh, y_mesh, t, z2,
                                   x_label = r'$x$', y_label = r'$y$',
                                   x_unit = 'rad', y_unit = 'rad',
                                   title = r'$(x + iy) e^{i t} \sin(x^2 + y^2 + t)$',
                                   shading = shading,
                                   colormap = plt.get_cmap('richardson'),
                                   richardson_equator_magnitude = equator_mag,
                                   target_dir = OUT_DIR,
                                   show_colorbar = False,
                                   aspect_ratio = 1,
                                   )
