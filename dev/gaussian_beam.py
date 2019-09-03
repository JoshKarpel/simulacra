import logging

from pathlib import Path

import numpy as np

import simulacra as si
import simulacra.units as u

import matplotlib.pyplot as plt

FILE_NAME = Path(__file__).stem
OUT_DIR = Path(__file__).parent / "out" / FILE_NAME


def w(z, w_0, z_0):
    return w_0 * np.sqrt(1 + ((z / z_0) ** 2))


def R(z, z_0):
    return z * (1 + ((z_0 / z) ** 2))


def guoy_phase(z, z_0):
    return np.arctan(z / z_0)


def field(x, z, w_0, wavelength):
    z_0 = u.pi * (w_0 ** 2) / wavelength
    k = u.twopi / wavelength
    w_z = w(z, w_0, z_0)
    amplitude = (w_0 / w_z) * np.exp(-((x / w_z) ** 2))
    phase = np.exp(1j * k * (x ** 2) / (2 * R(z, z_0))) * np.exp(
        -1j * guoy_phase(z, z_0)
    )

    return amplitude * phase


if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra", stdout_logs=True, stdout_level=logging.DEBUG
    ) as logger:
        w_0 = 1 * u.um
        wavelength = 500 * u.nm

        x_lim = 10 * u.um
        z_lim = 50 * u.um

        pnts = 500

        x = np.linspace(-x_lim, x_lim, pnts)
        z = np.linspace(-z_lim, z_lim, pnts)

        x_mesh, z_mesh = np.meshgrid(x, z, indexing="ij")
        field_mesh = field(x_mesh, z_mesh, w_0=w_0, wavelength=wavelength)

        si.vis.xyz_plot(
            f"gaussian_beam",
            z_mesh,
            x_mesh,
            field_mesh,
            title=rf"Gaussian Beam w/ $\lambda = {wavelength / u.nm:.1f} \, \mathrm{{nm}}, \, w_0 = {w_0 / u.um:.1f} \, \mathrm{{\mu m}}$",
            x_label="z",
            y_label="x",
            x_unit="um",
            y_unit="um",
            colormap=plt.get_cmap("richardson"),
            richardson_equator_magnitude=0.1,
            target_dir=OUT_DIR,
        )
