import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u


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
        cmap_names = (
            "viridis",
            "magma",
            "inferno",
            "plasma",
            "seismic",
            "PiYG",
            "PRGn",
            "Spectral",
        )

        lim = 5
        points = 500

        x = np.linspace(-lim, lim, points)
        y = np.linspace(-lim, lim, points)

        x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")

        z_mesh = np.sin(x_mesh * u.pi) * np.sin(y_mesh * u.pi)

        shared_kwargs = dict(z_lower_limit=-1, z_upper_limit=1, target_dir=OUT_DIR)

        for cmap_name in cmap_names:
            si.vis.xyz_plot(
                f"xyz_{cmap_name}",
                x_mesh,
                y_mesh,
                z_mesh,
                colormap=cmap_name,
                contours=[0.5],
                **shared_kwargs,
            )

        for cmap_name in cmap_names:
            si.vis.xyz_plot(
                f"xyz_{cmap_name}__log",
                x_mesh,
                y_mesh,
                z_mesh,
                colormap=cmap_name,
                contours=[0],
                z_log_axis=True,
                sym_log_linear_threshold=1e-1,
                **shared_kwargs,
            )
