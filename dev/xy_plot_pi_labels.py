import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', stdout_logs = True, stdout_level = logging.DEBUG, file_dir = OUT_DIR, file_logs = False) as logger:
        x = np.linspace(0, 2 * u.pi)
        y = x

        si.vis.xy_plot(
            'pi_test',
            x,
            y,
            x_unit = 'rad',
            y_unit = 'rad',
            fig_dpi_scale = 6,
            img_format = 'png',
            target_dir = OUT_DIR,
        )
