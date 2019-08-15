import logging
import os

import simulacra as si
import matplotlib.pyplot as plt
import numpy as np
from simulacra.units import *


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra",
        stdout_logs=True,
        stdout_level=logging.DEBUG,
        file_dir=OUT_DIR,
        file_logs=False,
    ) as logger:
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)

        x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")

        t = np.linspace(0, 30, 900) * pi

        def z(x_mesh, y_mesh, t):
            return np.sin(x_mesh ** 2 + y_mesh ** 2 + t)

        for cmap_name in (
            "viridis",
            "magma",
            "inferno",
            "plasma",
            "seismic",
            "PiYG",
            "PRGn",
            "Spectral",
        ):
            for shading in ("flat", "gouraud"):
                si.vis.xyzt_plot(
                    f"xyzt_{cmap_name}_{shading}",
                    x_mesh,
                    y_mesh,
                    t,
                    z,
                    x_label=r"$x$",
                    x_unit="cm",
                    y_label=r"$y$",
                    y_unit="cm",
                    z_unit="rad",
                    z_lower_limit=-1,
                    z_upper_limit=1,
                    t_unit="s",
                    title="wiggle wobble bobble",
                    contours=[1],
                    aspect_ratio=0.8,
                    colormap=plt.get_cmap(cmap_name),
                    shading=shading,
                    target_dir=OUT_DIR,
                )
