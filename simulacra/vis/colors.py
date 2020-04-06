import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors

# named colors
WHITE = "#ffffff"
BLACK = "#000000"

BLUE = "#1f77b4"  # matplotlib C0
ORANGE = "#ff7f0e"  # matplotlib C1
GREEN = "#2ca02c"  # matplotlib C2
RED = "#d62728"  # matplotlib C3
PURPLE = "#9467bd"  # matplotlib C4
BROWN = "#8c564b"  # matplotlib C5
PINK = "#e377c2"  # matplotlib C6
GRAY = "#7f7f7f"  # matplotlib C7
YELLOW = "#bcbd22"  # matplotlib C8
TEAL = "#17becf"  # matplotlib C9


CMAP_TO_OPPOSITE = {}


def register_opposite_color(cmap_name, opposite_color):
    CMAP_TO_OPPOSITE[cmap_name] = opposite_color
    CMAP_TO_OPPOSITE[plt.get_cmap(cmap_name)] = opposite_color


register_opposite_color("viridis", RED)
register_opposite_color("plasma", GREEN)
register_opposite_color("inferno", GREEN)
register_opposite_color("magma", GREEN)


class RichardsonColormap(matplotlib.colors.Colormap):
    """
    A matplotlib Colormap subclass which implements the colormap described in J. L. Richardson, Comput. Phys. Commun. 63, 84 (1991).

    This colormap is appropriate for visualizing complex-valued functions in two dimensions.
    """

    def __init__(self):
        self.name = "richardson"
        self.N = 256

    def __call__(self, x, alpha=1, bytes=False):
        real, imag = np.real(x), np.imag(x)

        mag = np.sqrt((real ** 2) + (imag ** 2))
        z = (mag ** 2) - 1
        zplus = z + 2
        eta = np.where(np.greater_equal(z, 0), 1, -1)

        common = 0.5 + (eta * (0.5 - (mag / zplus)))
        real_term = real / (np.sqrt(6) * zplus)
        imag_term = imag / (np.sqrt(2) * zplus)

        rgba = np.ones(
            np.shape(x) + (4,)
        )  # create rgba array of shape shape as x, except in last dimension, where rgba values will be stored
        rgba[:, 0] = common + (2 * real_term)  # red
        rgba[:, 1] = common - real_term + imag_term  # green
        rgba[:, 2] = common - real_term - imag_term  # blue

        return rgba


# register cmap so that plt.get_cmap('richardson') can find it
matplotlib.cm.register_cmap(name="richardson", cmap=RichardsonColormap())
register_opposite_color("richardson", WHITE)

blue_black_red_cdit = {
    "red": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.1), (1.0, 1.0, 1.0)),
    "blue": ((0.0, 0.0, 1.0), (0.5, 0.1, 0.0), (1.0, 0.0, 0.0)),
    "green": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
}
blue_black_red_cmap = matplotlib.colors.LinearSegmentedColormap(
    "BlueBlackRed", blue_black_red_cdit
)
matplotlib.cm.register_cmap(name="BlueBlackRed", cmap=blue_black_red_cmap)


class RichardsonNormalization(matplotlib.colors.Normalize):
    """A matplotlib Normalize subclass which implements an appropriate normalization for :class:`RichardsonColormap`."""

    def __init__(self, equator_magnitude=1):
        self.equator_magnitude = np.abs(equator_magnitude)

    def __call__(self, x, **kwargs):
        return np.ma.masked_invalid(x / self.equator_magnitude, copy=False)

    def autoscale(self, *args):
        pass

    def autoscale_None(self, *args):
        pass


class AbsoluteRenormalize(matplotlib.colors.Normalize):
    def __init__(self):
        super().__init__(vmin=0, vmax=1, clip=False)

    def __call__(self, x, **kwargs):
        return np.ma.masked_invalid(np.abs(x) / np.nanmax(np.abs(x)), copy=False)

    def autoscale(self, *args):
        pass

    def autoscale_None(self, *args):
        pass
