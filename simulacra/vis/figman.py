import logging
from typing import Union, Optional

from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt

from .. import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _points_to_inches(points: Union[float, int]) -> Union[float, int]:
    """Convert the input from points to inches (~72 points per inch)."""
    return points / 72.27


def _inches_to_points(inches: Union[float, int]) -> Union[float, int]:
    """Convert the input from inches to points (~72 points per inch)."""
    return inches * 72.27


DEFAULT_LATEX_PAGE_WIDTH = _points_to_inches(350.0)
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
PPT_WIDESCREEN_WIDTH = 13.333
PPT_WIDESCREEN_HEIGHT = 7.5
PPT_WIDESCREEN_ASPECT_RATIO = 16 / 9


def get_figure(
    fig_width: Union[float, int] = DEFAULT_LATEX_PAGE_WIDTH,
    aspect_ratio: Union[float, int] = GOLDEN_RATIO,
    fig_height: Optional[Union[float, int]] = None,
    fig_scale: Union[float, int, str] = 1,
    fig_dpi_scale: Union[float, int] = 1,
) -> plt.Figure:
    """
    Get a matplotlib figure object with the desired dimensions.

    Parameters
    ----------
    fig_width
        The "base" width of a figure (e.g. a LaTeX page width), in inches.
    fig_scale
        The scale of the figure relative to the figure width.
    fig_dpi_scale
        Multiplier for the figure DPI (only important if saving to png-like formats).
    aspect_ratio
        The aspect ratio of the figure (width / height)
    fig_height
        If not `None`, overrides the aspect ratio. In inches.

    Returns
    -------
    figure :
        A matplotlib figure.
    """
    fig = plt.figure(
        figsize=_get_fig_dims(
            fig_width=fig_width,
            fig_height=fig_height,
            aspect_ratio=aspect_ratio,
            fig_scale=fig_scale,
        ),
        dpi=fig_dpi_scale * 100,
    )

    return fig


def _get_fig_dims(
    fig_width: Union[float, int] = DEFAULT_LATEX_PAGE_WIDTH,
    aspect_ratio: Union[float, int] = GOLDEN_RATIO,
    fig_height=None,
    fig_scale: Union[float, int] = 1,
):
    """
    Return the dimensions (width, height) for a figure based on the scale, width (in points), and aspect ratio.

    Primarily a helper function for get_figure.

    Parameters
    ----------
    fig_scale
        The scale of the figure relative to the figure width.
    aspect_ratio
        The aspect ratio of the figure (width / height)
    fig_height
        If not `None`, overrides the aspect ratio. In inches.
    fig_width
        The "base" width of a figure (e.g. a LaTeX page width). In inches.

    Returns
    -------
    tuple of floats
        Figure width and height in inches.
    """
    fig_width = fig_width * fig_scale  # width in inches
    if fig_height is None:
        fig_height = fig_width / aspect_ratio  # height in inches

    return fig_width, fig_height


def save_current_figure(
    name: str,
    target_dir: Optional[str] = None,
    img_format: str = "pdf",
    transparent: bool = True,
    tight_layout: bool = True,
) -> Path:
    """
    Save the current matplotlib figure as an image to a file.

    Parameters
    ----------
    name : :class:`str`
        The name to save the image with.
    target_dir
        The directory to save the figure to.
    img_format
        The image format to save to.
    transparent
        If available for the format, makes the background transparent (works for ``.png``, for example).
    tight_layout : :class:`bool`
        If ``True``, saves the figure with ``bbox_inches = 'tight'``.

    Returns
    -------
    path :
        The path the figure was saved to.
    """
    if target_dir is None:
        target_dir = Path.cwd()
    path = Path(target_dir) / f"{name}.{img_format}"
    utils.ensure_parents_exist(path)

    if tight_layout:
        plt.savefig(str(path), bbox_inches="tight", transparent=transparent)
    else:
        plt.savefig(str(path), transparent=transparent)

    logger.debug("Saved figure {} to {}".format(name, str(path)))

    return path


class FigureManager:
    """
    A class that manages a matplotlib figure: creating it, showing it, saving
    it, and cleaning it up.

    Attributes
    ----------
    elements : :class:`dict`
        A dictionary that can contain references to parts of the figure that you may want to reference later.
        The references need to be added manually - this just provides a convenient place to put them.
    """

    def __init__(
        self,
        name: str,
        fig_width: float = DEFAULT_LATEX_PAGE_WIDTH,
        aspect_ratio: float = GOLDEN_RATIO,
        fig_height: Optional[float] = None,
        fig_scale: float = 1,
        fig_dpi_scale: float = 6,
        tight_layout: bool = True,
        transparent: bool = True,
        close_before_enter: bool = True,
        close: bool = True,
        target_dir: Optional[Path] = None,
        img_format: str = "pdf",
        save: bool = True,
        show: bool = False,
    ):
        """
        Parameters
        ----------
        name
            The desired file name for the plot.
        fig_width
            The width of the figure, in inches.
        aspect_ratio
            The aspect ratio of the figure (height / width).
        fig_height
            If not ``None``, overrides the ``aspect_ratio`` to set the height of the figure.
        fig_scale
            A multiplier for the overall scale of the figure.
            This should generally stay at ``1`` so that font sizes and other similar settings make sense.
        fig_dpi_scale
            A multiplier for the base DPI of the figure.
            ``6`` is a good number for high-quality plots.
        tight_layout
            If ``True``, uses matplotlib's tight layout option before saving.
        transparent
            If ``True``, and the ``img_format`` allows for it, the background of the saved image will be transparent.
        close_before_enter
            If ``True``, close whatever matplotlib plot is open before trying to create the new figure.
        close
            If ``True``, close the figure after exiting the context manager.
        target_dir
            The directory to save the plot to.
        img_format
            The format for the plot.
            Accepts any matplotlib file format.
        save
            If ``True``, save the figure after exiting the context manager.
            This occurs before it might be shown.
        show
            If ``True``, show the figure after exiting the context manager.
        """
        self.name = name

        self.fig_scale = fig_scale
        self.fig_dpi_scale = fig_dpi_scale
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.aspect_ratio = aspect_ratio
        self.tight_layout = tight_layout
        self.transparent = transparent

        self.target_dir = Path(target_dir)
        self.img_format = img_format

        self.close_before_enter = close_before_enter
        self.close = close
        self._save = save
        self.show = show

        self.path = None

        self.elements = {}

    def save(self) -> Path:
        path = save_current_figure(
            name=self.name,
            target_dir=self.target_dir,
            img_format=self.img_format,
            tight_layout=self.tight_layout,
            transparent=self.transparent,
        )
        self.path = path
        return path

    def __enter__(self):
        if self.close_before_enter:
            plt.close()

        self.fig = get_figure(
            fig_width=self.fig_width,
            aspect_ratio=self.aspect_ratio,
            fig_height=self.fig_height,
            fig_scale=self.fig_scale,
            fig_dpi_scale=self.fig_dpi_scale,
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()

    def exit(self):
        if self._save:
            self.save()

        if self.show:
            plt.show()

        if self.close:
            self.fig.clear()
            plt.close(self.fig)
