import logging
import os
import sys

import numpy as np

import simulacra as si

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra", stdout_logs=True, stdout_level=logging.DEBUG
    ) as logger:
        x = np.linspace(0, 10, 1000)
        y = np.sin(x)

        times = np.linspace(0, 10, 1000)

        with si.vis.FigureManager(
            "easy", fig_dpi_scale=3, target_dir=OUT_DIR
        ) as figman:
            fig = figman.fig
            ax = fig.add_subplot(111)

            line, = ax.plot(x, y, animated=True)

            def update(t):
                line.set_ydata(np.sin(x + t))

            si.vis.animate(figman, update, times, artists=[line])
