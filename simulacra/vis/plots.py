import logging
from typing import Union, Optional, Iterable, Collection, Dict, Any

import itertools
import os
import fractions
import collections
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors

from .. import utils
from .. import units as u

from . import colors

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_TITLE_KWARGS = dict(fontsize=16)
DEFAULT_LABEL_KWARGS = dict(fontsize=14)

GRID_KWARGS = dict(linestyle="-", color="black", linewidth=0.5, alpha=0.4)

MINOR_GRID_KWARGS = GRID_KWARGS.copy()
MINOR_GRID_KWARGS["alpha"] -= 0.2

COLORMESH_GRID_KWARGS = dict(linestyle="-", linewidth=0.5, alpha=0.4)

LEGEND_KWARGS = dict(loc="best")

HVLINE_KWARGS = dict(linestyle="-", color="black")

CONTOUR_KWARGS = dict()

CONTOUR_LABEL_KWARGS = dict(inline=1, fontsize=8)

TITLE_OFFSET = 1.15


class ColormapShader(utils.StrEnum):
    FLAT = "flat"
    GOURAUD = "gouraud"


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


def xy_plot(
    name: str,
    x_data: np.ndarray,
    *y_data: np.ndarray,
    line_labels: Iterable[str] = (),
    line_kwargs: Iterable[dict] = (),
    x_unit: u.Unit = None,
    y_unit: u.Unit = None,
    x_log_axis: bool = False,
    y_log_axis: bool = False,
    x_lower_limit: Optional[float] = None,
    x_upper_limit: Optional[float] = None,
    y_lower_limit: Optional[float] = None,
    y_upper_limit: Optional[float] = None,
    y_pad: float = 0,
    y_log_pad: float = 1,
    sym_log_linear_threshold: float = 1e-3,
    vlines: Iterable[float] = (),
    vline_kwargs: Iterable[dict] = (),
    hlines: Iterable[float] = (),
    hline_kwargs: Iterable[dict] = (),
    x_extra_ticks: Optional[Collection[float]] = None,
    y_extra_ticks: Optional[Collection[float]] = None,
    x_extra_tick_labels: Optional[Collection[str]] = None,
    y_extra_tick_labels: Optional[Collection[str]] = None,
    title: Optional[str] = None,
    title_offset: float = TITLE_OFFSET,
    title_kwargs: Optional[dict] = None,
    x_label: Optional[str] = None,
    x_label_kwargs: Optional[dict] = None,
    y_label: Optional[str] = None,
    y_label_kwargs: Optional[dict] = None,
    font_size_tick_labels: float = 10,
    font_size_legend: float = 12,
    ticks_on_top: bool = True,
    ticks_on_right: bool = True,
    legend_on_right: bool = False,
    legend_kwargs: Optional[dict] = None,
    grid_kwargs: Optional[dict] = None,
    minor_grid_kwargs: Optional[dict] = None,
    equal_aspect: bool = False,
    figure_manager: Optional[FigureManager] = None,
    **kwargs,
) -> FigureManager:
    """
    Generate and save a generic x vs. y plot.

    This function is suitable for displaying any number of curves as long as
    they all share the same ``x_data``. If you need to display multiple curves
    with independent ``x_data``, see :function:`xxyy_plot``.

    Parameters
    ----------
    name
        The filename for the plot (not including path, which should be passed via they keyword argument ``target_dir``).
    x_data
        A single array that will be used as x-values for all the `y_data`.
    y_data
        Any number of arrays of the same length as `x_data`, each of which will appear as a line on the plot.
    line_labels
        Labels for each of the `y_data` lines.
    line_kwargs
        Keyword arguments for each of the `y_data` lines (a list of dictionaries).
    x_unit
        The unit for the x-axis. Can be a number or the name of a unit as string.
    y_unit
        The unit for the y-axis. Can be a number or the name of a unit as string.
    x_log_axis
        If ``True``, the x-axis will be log-scaled.
    y_log_axis
        If ``True``, the y-axis will be log-scaled.
    x_lower_limit
        The lower limit for the x-axis. If ``None``, set automatically from the `x_data`.
    x_upper_limit
        The upper limit for the x-axis. If ``None``, set automatically from the `x_data`.
    y_lower_limit
        The lower limit for the y-axis. If ``None``, set automatically from the `y_data`.
    y_upper_limit
        The upper limit for the y-axis. If ``None``, set automatically from the `y_data`.
    y_pad
        The linear padding factor for the y-axis. See :func:`calculate_axis_limits`.
    y_log_pad
        The logarithmic padding factor for the y-axis. See :func:`calculate_axis_limits`.
    sym_log_linear_threshold
        When the y-axis is in symmetric log-linear mode, this sets the threshold
        for switching from log to linear scaling.
    vlines
        A list of positions to draw vertical lines.
    vline_kwargs
        Keyword arguments for each of the `vlines` (a list of dictionaries).
    hlines
        A list of positions to draw horizontal lines.
    hline_kwargs
        Keyword arguments for each of the `hlines` (a list of dictionaries).
    x_extra_ticks
        Additional tick marks to display on the x-axis.
    y_extra_ticks
        Additional tick marks to display on the y-axis.
    x_extra_tick_labels
        Labels for the extra x ticks.
    y_extra_tick_labels
        Labels for the extra y ticks.
    title
        The text to display above the plot.
    title_offset
        How far to move the title vertically.
    x_label
        The label to display below the x-axis.
    y_label
        The label to display to the left of the y-axis.
    font_size_title
        The font size for the title.
    font_size_axis_labels
        The font size for the axis labels.
    font_size_tick_labels
        The font size for the tick labels.
    font_size_legend
        The font size for the legend.
    ticks_on_top
        If ``True``, axis ticks will be shown along the top side of the plot
        (in addition to the bottom).
    ticks_on_right
        If ``True``, axis ticks will be shown along the right side of the plot
        (in addition to the left).
    legend_on_right
        If ``True``, the legend will be displayed hanging on the right side of
        the plot.
    legend_kwargs
        Keyword arguments for the legend.
    grid_kwargs
        Keyword arguments for the major gridlines.
    minor_grid_kwargs
        Keyword arguments for the minor gridlines.
    equal_aspect
        If ``True``, the aspect ratio of the axes will be set to ``'equal'``.
    figure_manager
        An existing :class:`FigureManager` instance to use instead of creating
        a new one.
    kwargs
        Additional keyword arguments are passed to :class:`FigureManager`.

    Returns
    -------
    :class:`FigureManager`
        The :class:`FigureManager` that the xy-plot was constructed in.
    """
    legend_kwargs = collections.ChainMap(legend_kwargs or {}, LEGEND_KWARGS)
    grid_kwargs = collections.ChainMap(grid_kwargs or {}, GRID_KWARGS)
    minor_grid_kwargs = collections.ChainMap(minor_grid_kwargs or {}, MINOR_GRID_KWARGS)

    if figure_manager is None:
        figure_manager = FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = fig.add_subplot(111)
        if equal_aspect:
            ax.set_aspect("equal")

        fm.elements["axis"] = ax

        # ensure data is in numpy arrays
        x_data = np.array(x_data)
        y_data = [np.array(y) for y in y_data]
        line_labels = tuple(line_labels)
        line_kwargs = tuple(line_kwargs)

        x_unit_value, y_unit_value = u.get_unit_values(x_unit, y_unit)

        lines = []
        for y, label, kwargs in itertools.zip_longest(y_data, line_labels, line_kwargs):
            kwargs = kwargs or {}
            label = label or ""
            lines.append(
                plt.plot(
                    x_data / x_unit_value, y / y_unit_value, label=label, **kwargs
                )[0]
            )
        fm.elements["lines"] = lines

        vlines = _attach_h_or_v_lines(
            ax,
            direction="v",
            line_positions=vlines,
            line_kwargs=vline_kwargs,
            unit=x_unit,
        )
        fm.elements["vlines"] = vlines
        hlines = _attach_h_or_v_lines(
            ax,
            direction="h",
            line_positions=hlines,
            line_kwargs=hline_kwargs,
            unit=y_unit,
        )
        fm.elements["hlines"] = hlines

        x_lower_limit, x_upper_limit = _set_axis_limits_and_scale(
            ax,
            x_data,
            lower_limit=x_lower_limit,
            upper_limit=x_upper_limit,
            log=x_log_axis,
            unit=x_unit,
            direction="x",
            sym_log_linear_threshold=sym_log_linear_threshold,
        )
        y_lower_limit, y_upper_limit = _set_axis_limits_and_scale(
            ax,
            *y_data,
            lower_limit=y_lower_limit,
            upper_limit=y_upper_limit,
            log=y_log_axis,
            pad=y_pad,
            log_pad=y_log_pad,
            unit=y_unit,
            direction="y",
            sym_log_linear_threshold=sym_log_linear_threshold,
        )

        ax.tick_params(axis="both", which="major", labelsize=font_size_tick_labels)

        _set_title(
            ax, title_text=title, title_offset=title_offset, title_kwargs=title_kwargs
        )
        _set_axis_label(
            ax, which="x", label=x_label, unit=x_unit, label_kwargs=x_label_kwargs
        )
        _set_axis_label(
            ax, which="y", label=y_label, unit=y_unit, label_kwargs=y_label_kwargs
        )

        if len(line_labels) > 0 or "handles" in legend_kwargs:
            if not legend_on_right:
                legend = ax.legend(fontsize=font_size_legend, **legend_kwargs)
            if legend_on_right:
                legend_kwargs = collections.ChainMap(
                    dict(
                        loc="upper left",
                        bbox_to_anchor=(1.2, 1),
                        borderaxespad=0,
                        fontsize=font_size_legend,
                        ncol=1 + (len(line_labels) // 17),
                    ),
                    legend_kwargs,
                )
                legend = ax.legend(**legend_kwargs)

        # draw that figure so that the ticks exist, so that we can add more ticks
        fig.canvas.draw()

        for which, unit, limits in [
            ("x", x_unit, (x_lower_limit, x_upper_limit)),
            ("y", y_unit, (y_lower_limit, y_upper_limit)),
        ]:
            _maybe_set_pi_ticks_and_labels(ax, which, unit, *limits)

        for which, unit, extra_ticks, extra_tick_labels in [
            ("x", x_unit, x_extra_ticks, x_extra_tick_labels),
            ("y", y_unit, y_extra_ticks, y_extra_tick_labels),
        ]:
            _add_extra_ticks(
                ax,
                which=which,
                extra_ticks=extra_ticks,
                labels=extra_tick_labels,
                unit=unit,
            )

        ax.grid(True, which="major", **grid_kwargs)
        ax.minorticks_on()
        if x_log_axis:
            ax.grid(True, which="minor", axis="x", **minor_grid_kwargs)
        if y_log_axis:
            ax.grid(True, which="minor", axis="y", **minor_grid_kwargs)

        # set limits again to guarantee we don't see ticks outside the limits
        ax.set_xlim(x_lower_limit, x_upper_limit)
        ax.set_ylim(y_lower_limit, y_upper_limit)

        # set these AFTER adding extra tick labels so that we don't have to slice into the middle of the label lists above
        ax.tick_params(
            top=ticks_on_top,
            labeltop=ticks_on_top,
            right=ticks_on_right,
            labelright=ticks_on_right,
        )

    return fm


def xy_stackplot(
    name: str,
    x_data: np.ndarray,
    *y_data: np.ndarray,
    line_labels: Iterable[str] = (),
    line_kwargs: Iterable[dict] = (),
    x_unit: u.Unit = None,
    y_unit: u.Unit = None,
    x_log_axis: bool = False,
    y_log_axis: bool = False,
    x_lower_limit: Optional[float] = None,
    x_upper_limit: Optional[float] = None,
    y_lower_limit: Optional[float] = None,
    y_upper_limit: Optional[float] = None,
    y_pad: float = 0,
    y_log_pad: float = 1,
    vlines: Iterable[float] = (),
    vline_kwargs: Iterable[dict] = (),
    hlines: Iterable[float] = (),
    hline_kwargs: Iterable[dict] = (),
    x_extra_ticks: Optional[Collection[float]] = None,
    y_extra_ticks: Optional[Collection[float]] = None,
    x_extra_tick_labels: Optional[Collection[str]] = None,
    y_extra_tick_labels: Optional[Collection[str]] = None,
    title: Optional[str] = None,
    title_offset: float = TITLE_OFFSET,
    title_kwargs: Optional[dict] = None,
    x_label: Optional[str] = None,
    x_label_kwargs: Optional[dict] = None,
    y_label: Optional[str] = None,
    y_label_kwargs: Optional[dict] = None,
    font_size_tick_labels: float = 10,
    font_size_legend: float = 12,
    ticks_on_top: bool = True,
    ticks_on_right: bool = True,
    legend_on_right: bool = False,
    legend_kwargs: Optional[dict] = None,
    grid_kwargs: Optional[dict] = None,
    minor_grid_kwargs: Optional[dict] = None,
    equal_aspect: bool = False,
    figure_manager: Optional[FigureManager] = None,
    **kwargs,
) -> FigureManager:
    """
    Generate and save a generic x vs. y plot.

    Parameters
    ----------
    name
        The filename for the plot (not including path, which should be passed via they keyword argument ``target_dir``).
    x_data
        A single array that will be used as x-values for all the `y_data`.
    y_data
        Any number of arrays of the same length as `x_data`, each of which will appear as a line on the plot.
    line_labels
        Labels for each of the `y_data` lines.
    line_kwargs
        Keyword arguments for each of the `y_data` lines (a list of dictionaries).
    x_unit
        The unit for the x-axis. Can be a number or the name of a unit as string.
    y_unit
        The unit for the y-axis. Can be a number or the name of a unit as string.
    x_log_axis
        If ``True``, the x-axis will be log-scaled.
    y_log_axis
        If ``True``, the y-axis will be log-scaled.
    x_lower_limit
        The lower limit for the x-axis. If ``None``, set automatically from the `x_data`.
    x_upper_limit
        The upper limit for the x-axis. If ``None``, set automatically from the `x_data`.
    y_lower_limit
        The lower limit for the y-axis. If ``None``, set automatically from the `y_data`.
    y_upper_limit
        The upper limit for the y-axis. If ``None``, set automatically from the `y_data`.
    y_pad
        The linear padding factor for the y-axis. See :func:`calculate_axis_limits`.
    y_log_pad
        The logarithmic padding factor for the y-axis. See :func:`calculate_axis_limits`.
    vlines
        A list of positions to draw vertical lines.
    vline_kwargs
        Keyword arguments for each of the `vlines` (a list of dictionaries).
    hlines
        A list of positions to draw horizontal lines.
    hline_kwargs
        Keyword arguments for each of the `hlines` (a list of dictionaries).
    x_extra_ticks
        Additional tick marks to display on the x-axis.
    y_extra_ticks
        Additional tick marks to display on the y-axis.
    x_extra_tick_labels
        Labels for the extra x ticks.
    y_extra_tick_labels
        Labels for the extra y ticks.
    title
        The text to display above the plot.
    title_offset
        How far to move the title vertically.
    x_label
        The label to display below the x-axis.
    y_label
        The label to display to the left of the y-axis.
    font_size_title
        The font size for the title.
    font_size_axis_labels
        The font size for the axis labels.
    font_size_tick_labels
        The font size for the tick labels.
    font_size_legend
        The font size for the legend.
    ticks_on_top
        If ``True``, axis ticks will be shown along the top side of the plot (in addition to the bottom).
    ticks_on_right
        If ``True``, axis ticks will be shown along the right side of the plot (in addition to the left).
    legend_on_right
        If ``True``, the legend will be displayed hanging on the right side of the plot.
    legend_kwargs
        Keyword arguments for the legend.
    grid_kwargs
        Keyword arguments for the major gridlines.
    minor_grid_kwargs
        Keyword arguments for the minor gridlines.
    equal_aspect
        If ``True``, the aspect ratio of the axes will be set to ``'equal'``.
    figure_manager
        An existing :class:`FigureManager` instance to use instead of creating a new one.
    kwargs
        Additional keyword arguments are passed to :class:`FigureManager`.

    Returns
    -------
    :class:`FigureManager`
        The :class:`FigureManager` that the xy-plot was constructed in.
    """
    legend_kwargs = collections.ChainMap(legend_kwargs or {}, LEGEND_KWARGS)
    grid_kwargs = collections.ChainMap(grid_kwargs or {}, GRID_KWARGS)
    minor_grid_kwargs = collections.ChainMap(minor_grid_kwargs or {}, MINOR_GRID_KWARGS)

    if figure_manager is None:
        figure_manager = FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = fig.add_subplot(111)
        if equal_aspect:
            ax.set_aspect("equal")

        fm.elements["axis"] = ax

        # ensure data is in numpy arrays
        x_data = np.array(x_data)
        y_data = [np.array(y) for y in y_data]
        line_labels = tuple(line_labels)
        line_kwargs = tuple(line_kwargs)

        x_unit_value, y_unit_value = u.get_unit_values(x_unit, y_unit)

        x = x_data / x_unit_value
        ys = [
            y / y_unit_value for y, label in itertools.zip_longest(y_data, line_labels)
        ]
        line_labels = [
            label or "" for y, label in itertools.zip_longest(y_data, line_labels)
        ]

        stackplot = ax.stackplot(x, *ys, labels=line_labels)
        fm.elements["stackplot"] = stackplot

        vlines = _attach_h_or_v_lines(
            ax,
            direction="v",
            line_positions=vlines,
            line_kwargs=vline_kwargs,
            unit=x_unit,
        )
        hlines = _attach_h_or_v_lines(
            ax,
            direction="h",
            line_positions=hlines,
            line_kwargs=hline_kwargs,
            unit=y_unit,
        )
        fm.elements["vlines"] = vlines
        fm.elements["hlines"] = hlines

        x_lower_limit, x_upper_limit = _set_axis_limits_and_scale(
            ax,
            x_data,
            direction="x",
            lower_limit=x_lower_limit,
            upper_limit=x_upper_limit,
            log=x_log_axis,
            unit=x_unit,
        )
        y_lower_limit, y_upper_limit = _set_axis_limits_and_scale(
            ax,
            *y_data,
            direction="y",
            lower_limit=y_lower_limit,
            upper_limit=y_upper_limit,
            log=y_log_axis,
            pad=y_pad,
            log_pad=y_log_pad,
            unit=y_unit,
        )

        ax.tick_params(axis="both", which="major", labelsize=font_size_tick_labels)

        _set_title(
            ax, title_text=title, title_offset=title_offset, title_kwargs=title_kwargs
        )
        _set_axis_label(
            ax, which="x", label=x_label, unit=x_unit, label_kwargs=x_label_kwargs
        )
        _set_axis_label(
            ax, which="y", label=y_label, unit=y_unit, label_kwargs=y_label_kwargs
        )

        if len(line_labels) > 0 or "handles" in legend_kwargs:
            if not legend_on_right:
                legend = ax.legend(fontsize=font_size_legend, **legend_kwargs)
            if legend_on_right:
                legend_kwargs = collections.ChainMap(
                    legend_kwargs,
                    dict(
                        loc="upper left",
                        bbox_to_anchor=(1.15, 1),
                        borderaxespad=0,
                        fontsize=font_size_legend,
                        ncol=1 + (len(line_labels) // 17),
                    ),
                )
                legend = ax.legend(**legend_kwargs)

        # draw that figure so that the ticks exist, so that we can add more ticks
        fig.canvas.draw()

        for which, unit, limits in [
            ("x", x_unit, (x_lower_limit, x_upper_limit)),
            ("y", y_unit, (y_lower_limit, y_upper_limit)),
        ]:
            _maybe_set_pi_ticks_and_labels(ax, which, unit, *limits)

        _add_extra_ticks(
            ax,
            which="x",
            extra_ticks=x_extra_ticks,
            labels=x_extra_tick_labels,
            unit=x_unit,
        )
        _add_extra_ticks(
            ax,
            which="y",
            extra_ticks=y_extra_ticks,
            labels=y_extra_tick_labels,
            unit=y_unit,
        )

        ax.grid(True, which="major", **grid_kwargs)
        ax.minorticks_on()
        if x_log_axis:
            ax.grid(True, which="minor", axis="x", **minor_grid_kwargs)
        if y_log_axis:
            ax.grid(True, which="minor", axis="y", **minor_grid_kwargs)

        # set limits again to guarantee we don't see ticks outside the limits
        ax.set_xlim(x_lower_limit, x_upper_limit)
        ax.set_ylim(y_lower_limit, y_upper_limit)

        # set these AFTER adding extra tick labels so that we don't have to slice into the middle of the label lists above
        ax.tick_params(
            top=ticks_on_top,
            labeltop=ticks_on_top,
            right=ticks_on_right,
            labelright=ticks_on_right,
        )

    return fm


def xxyy_plot(
    name,
    x_data,
    y_data,
    line_labels=(),
    line_kwargs=(),
    x_unit=None,
    y_unit=None,
    x_log_axis=False,
    y_log_axis=False,
    x_lower_limit=None,
    x_upper_limit=None,
    y_lower_limit=None,
    y_upper_limit=None,
    y_pad=0,
    y_log_pad=1,
    vlines=(),
    vline_kwargs=(),
    hlines=(),
    hline_kwargs=(),
    x_extra_ticks=None,
    y_extra_ticks=None,
    x_extra_tick_labels=None,
    y_extra_tick_labels=None,
    title=None,
    title_offset=TITLE_OFFSET,
    title_kwargs: Optional[dict] = None,
    x_label=None,
    x_label_kwargs: Optional[dict] = None,
    y_label=None,
    y_label_kwargs: Optional[dict] = None,
    font_size_tick_labels=10,
    font_size_legend=12,
    ticks_on_top=True,
    ticks_on_right=True,
    legend_on_right=False,
    grid_kwargs=None,
    minor_grid_kwargs=None,
    legend_kwargs=None,
    figure_manager=None,
    **kwargs,
) -> FigureManager:
    legend_kwargs = collections.ChainMap(legend_kwargs or {}, LEGEND_KWARGS)
    grid_kwargs = collections.ChainMap(grid_kwargs or {}, GRID_KWARGS)
    minor_grid_kwargs = collections.ChainMap(minor_grid_kwargs or {}, MINOR_GRID_KWARGS)

    if figure_manager is None:
        figure_manager = FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = fig.add_subplot(111)

        fm.elements["axis"] = ax

        # ensure data is in numpy arrays
        x_data = [np.array(x) for x in x_data]
        y_data = [np.array(y) for y in y_data]
        line_labels = tuple(line_labels)
        line_kwargs = tuple(line_kwargs)

        x_unit_value, x_unit_tex = u.get_unit_value_and_latex_from_unit(x_unit)
        y_unit_value, y_unit_tex = u.get_unit_value_and_latex_from_unit(y_unit)

        lines = []
        for x, y, lab, kw in itertools.zip_longest(
            x_data, y_data, line_labels, line_kwargs
        ):
            kw = kw or {}
            lab = lab or ""
            lines.append(
                plt.plot(x / x_unit_value, y / y_unit_value, label=lab, **kw)[0]
            )
        fm.elements["lines"] = lines

        _attach_h_or_v_lines(
            ax,
            direction="v",
            line_positions=vlines,
            line_kwargs=vline_kwargs,
            unit=x_unit,
        )
        _attach_h_or_v_lines(
            ax,
            direction="h",
            line_positions=hlines,
            line_kwargs=hline_kwargs,
            unit=y_unit,
        )

        x_lower_limit, x_upper_limit = _set_axis_limits_and_scale(
            ax,
            *x_data,
            lower_limit=x_lower_limit,
            upper_limit=x_upper_limit,
            log=x_log_axis,
            pad=0,
            log_pad=1,
            unit=x_unit,
            direction="x",
        )
        y_lower_limit, y_upper_limit = _set_axis_limits_and_scale(
            ax,
            *y_data,
            lower_limit=y_lower_limit,
            upper_limit=y_upper_limit,
            log=y_log_axis,
            pad=y_pad,
            log_pad=y_log_pad,
            unit=y_unit,
            direction="y",
        )

        ax.tick_params(axis="both", which="major", labelsize=font_size_tick_labels)

        _set_title(
            ax, title_text=title, title_offset=title_offset, title_kwargs=title_kwargs
        )
        _set_axis_label(
            ax, which="x", label=x_label, unit=x_unit, label_kwargs=x_label_kwargs
        )
        _set_axis_label(
            ax, which="y", label=y_label, unit=y_unit, label_kwargs=y_label_kwargs
        )
        if len(line_labels) > 0 or "handles" in legend_kwargs:
            if not legend_on_right:
                legend = ax.legend(
                    **{**legend_kwargs, **dict(fontsize=font_size_legend)}
                )
            if legend_on_right:
                legend_kwargs["loc"] = "upper left"
                legend = ax.legend(
                    **{
                        **legend_kwargs,
                        **dict(
                            bbox_to_anchor=(1.15, 1),
                            borderaxespad=0.0,
                            fontsize=font_size_legend,
                            ncol=1 + (len(line_labels) // 17),
                        ),
                    }
                )

        # draw that figure so that the ticks exist, so that we can add more ticks
        fig.canvas.draw()

        for which, unit, limits in [
            ("x", x_unit, (x_lower_limit, x_upper_limit)),
            ("y", y_unit, (y_lower_limit, y_upper_limit)),
        ]:
            _maybe_set_pi_ticks_and_labels(ax, which, unit, *limits)

        for which, unit, extra_ticks, extra_tick_labels in [
            ("x", x_unit, x_extra_ticks, x_extra_tick_labels),
            ("y", y_unit, y_extra_ticks, y_extra_tick_labels),
        ]:
            _add_extra_ticks(
                ax,
                which=which,
                extra_ticks=extra_ticks,
                labels=extra_tick_labels,
                unit=unit,
            )

        ax.grid(True, which="major", **grid_kwargs)
        if x_log_axis:
            ax.grid(True, which="minor", axis="x", **minor_grid_kwargs)
        if y_log_axis:
            ax.grid(True, which="minor", axis="y", **minor_grid_kwargs)

        # set limits again to guarantee we don't see ticks oustide the limits
        ax.set_xlim(x_lower_limit, x_upper_limit)
        ax.set_ylim(y_lower_limit, y_upper_limit)

        # set these AFTER adding extra tick labels so that we don't have to slice into the middle of the label lists above
        ax.tick_params(
            top=ticks_on_top,
            labeltop=ticks_on_top,
            right=ticks_on_right,
            labelright=ticks_on_right,
        )

    return fm


def xyz_plot(
    name: str,
    x_mesh: np.array,
    y_mesh: np.array,
    z_mesh: np.array,
    x_unit: Optional[u.Unit] = None,
    y_unit: Optional[u.Unit] = None,
    z_unit: Optional[u.Unit] = None,
    x_log_axis: bool = False,
    y_log_axis: bool = False,
    z_log_axis: bool = False,
    x_lower_limit=None,
    x_upper_limit=None,
    y_lower_limit=None,
    y_upper_limit=None,
    z_lower_limit=None,
    z_upper_limit=None,
    z_pad=0,
    z_log_pad=1,
    x_extra_ticks=None,
    y_extra_ticks=None,
    x_extra_tick_labels=None,
    y_extra_tick_labels=None,
    title=None,
    title_offset=TITLE_OFFSET,
    title_kwargs: Optional[dict] = None,
    x_label=None,
    x_label_kwargs: Optional[dict] = None,
    y_label=None,
    y_label_kwargs: Optional[dict] = None,
    z_label=None,
    font_size_tick_labels=10,
    ticks_on_top=True,
    ticks_on_right=True,
    grids=True,
    grid_kwargs=None,
    minor_grids=True,
    minor_grid_kwargs=None,
    vlines=(),
    vline_kwargs=(),
    hlines=(),
    hline_kwargs=(),
    contours=(),
    contour_kwargs=None,
    show_contour_labels=True,
    contour_label_kwargs=None,
    lines=(),
    line_kwargs=(),
    colormap=plt.get_cmap("viridis"),
    shading="flat",
    show_colorbar=True,
    richardson_equator_magnitude=1,
    sym_log_norm_epsilon=1e-3,
    figure_manager=None,
    **kwargs,
) -> FigureManager:
    grid_kwargs = collections.ChainMap(grid_kwargs or {}, GRID_KWARGS)
    minor_grid_kwargs = collections.ChainMap(minor_grid_kwargs or {}, MINOR_GRID_KWARGS)

    contour_kwargs = collections.ChainMap(contour_kwargs or {}, CONTOUR_KWARGS)
    contour_label_kwargs = collections.ChainMap(
        contour_label_kwargs or {}, CONTOUR_LABEL_KWARGS
    )

    if figure_manager is None:
        figure_manager = FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = fig.add_subplot(111)

        # grid_color = colors.CMAP_TO_OPPOSITE.get(colormap, 'black')
        # grid_kwargs['color'] = grid_color
        # minor_grid_kwargs['color'] = grid_color

        plt.set_cmap(colormap)

        x_unit_value, y_unit_value, z_unit_value = u.get_unit_values(
            x_unit, y_unit, z_unit
        )
        z_unit_label = _get_unit_str_for_axis_label(z_unit)

        x_lower_limit, x_upper_limit = _set_axis_limits_and_scale(
            ax,
            x_mesh,
            direction="x",
            lower_limit=x_lower_limit,
            upper_limit=x_upper_limit,
            log=x_log_axis,
            pad=0,
            log_pad=1,
            unit=x_unit,
        )
        y_lower_limit, y_upper_limit = _set_axis_limits_and_scale(
            ax,
            y_mesh,
            direction="y",
            lower_limit=y_lower_limit,
            upper_limit=y_upper_limit,
            log=y_log_axis,
            pad=0,
            log_pad=1,
            unit=y_unit,
        )

        if isinstance(colormap, colors.RichardsonColormap):
            norm = colors.RichardsonNormalization(
                equator_magnitude=richardson_equator_magnitude
            )
        else:
            z_lower_limit, z_upper_limit = _calculate_axis_limits(
                z_mesh,
                lower_limit=z_lower_limit,
                upper_limit=z_upper_limit,
                log=z_log_axis,
                pad=z_pad,
                log_pad=z_log_pad,
            )

            norm_kwargs = dict(
                vmin=z_lower_limit / z_unit_value, vmax=z_upper_limit / z_unit_value
            )

            if z_log_axis:
                if z_lower_limit > 0:
                    norm = matplotlib.colors.LogNorm(**norm_kwargs)
                else:
                    norm = matplotlib.colors.SymLogNorm(
                        ((np.abs(z_lower_limit) + np.abs(z_upper_limit)) / 2)
                        * sym_log_norm_epsilon,
                        **norm_kwargs,
                    )
            else:
                norm = matplotlib.colors.Normalize(**norm_kwargs)

        colormesh = ax.pcolormesh(
            x_mesh / x_unit_value,
            y_mesh / y_unit_value,
            z_mesh / z_unit_value,
            shading=shading,
            norm=norm,
        )

        if len(contours) > 0:
            contours = ax.contour(
                x_mesh / x_unit_value,
                y_mesh / y_unit_value,
                z_mesh / z_unit_value,
                levels=np.array(contours) / z_unit_value,
                **contour_kwargs,
            )
            fm.elements["contours"] = contours

            if show_contour_labels:
                contour_labels = ax.clabel(contours, **contour_label_kwargs)
                fm.elements["contour_labels"] = contour_labels

        for (x, y), kw in itertools.zip_longest(lines, line_kwargs):
            kw = kw or {}
            plt.plot(x / x_unit_value, y / y_unit_value, **kw)

        _attach_h_or_v_lines(
            ax,
            direction="v",
            line_positions=vlines,
            line_kwargs=vline_kwargs,
            unit=x_unit,
        )
        _attach_h_or_v_lines(
            ax,
            direction="h",
            line_positions=hlines,
            line_kwargs=hline_kwargs,
            unit=y_unit,
        )

        ax.tick_params(axis="both", which="major", labelsize=font_size_tick_labels)

        _set_title(
            ax, title_text=title, title_offset=title_offset, title_kwargs=title_kwargs
        )
        _set_axis_label(
            ax, which="x", label=x_label, unit=x_unit, label_kwargs=x_label_kwargs
        )
        _set_axis_label(
            ax, which="y", label=y_label, unit=y_unit, label_kwargs=y_label_kwargs
        )

        if show_colorbar and colormap.name != "richardson":
            cbar = plt.colorbar(mappable=colormesh, ax=ax, pad=0.1)
            if z_label is not None:
                z_label = cbar.set_label(r"{}".format(z_label) + z_unit_label)

        # draw that figure so that the ticks exist, so that we can add more ticks
        fig.canvas.draw()

        for which, unit, limits in [
            ("x", x_unit, (x_lower_limit, x_upper_limit)),
            ("y", y_unit, (y_lower_limit, y_upper_limit)),
        ]:
            _maybe_set_pi_ticks_and_labels(ax, which, unit, *limits)

        for which, unit, extra_ticks, extra_tick_labels in [
            ("x", x_unit, x_extra_ticks, x_extra_tick_labels),
            ("y", y_unit, y_extra_ticks, y_extra_tick_labels),
        ]:
            _add_extra_ticks(
                ax,
                which=which,
                extra_ticks=extra_ticks,
                labels=extra_tick_labels,
                unit=unit,
            )

        if grids:
            ax.grid(grids, which="major", **grid_kwargs)
        if minor_grids:
            if x_log_axis:
                ax.grid(True, which="minor", axis="x", **minor_grid_kwargs)
            if y_log_axis:
                ax.grid(True, which="minor", axis="y", **minor_grid_kwargs)

        # set limits again to guarantee we don't see ticks oustide the limits
        ax.set_xlim(x_lower_limit, x_upper_limit)
        ax.set_ylim(y_lower_limit, y_upper_limit)

        # set these AFTER adding extra tick labels so that we don't have to slice into the middle of the label lists above
        ax.tick_params(
            top=ticks_on_top,
            labeltop=ticks_on_top,
            right=ticks_on_right,
            labelright=ticks_on_right,
        )

    return fm


def _get_pi_ticks_and_labels(
    lower_limit: Union[float, int] = 0,
    upper_limit: Union[float, int] = u.twopi,
    denom: int = 4,
):
    """
    NB: doesn't really work for large ranges.

    Parameters
    ----------
    lower_limit
    upper_limit
    denom

    Returns
    -------

    """
    low = int(np.floor(lower_limit / u.pi))
    high = int(np.ceil(upper_limit / u.pi))

    ticks = list(
        fractions.Fraction(n, denom) for n in range(low * denom, (high * denom) + 1)
    )
    labels = []
    for tick in ticks:
        if tick.numerator == 0:
            labels.append(r"$0$")
        elif tick.numerator == tick.denominator == 1:
            labels.append(r"$\pi$")
        elif tick.numerator == -1 and tick.denominator == 1:
            labels.append(r"$-\pi$")
        elif tick.denominator == 1:
            labels.append(fr"$ {tick.numerator} \pi $")
        else:
            if tick.numerator > 0:
                labels.append(
                    fr"$ \frac{{ {tick.numerator} }}{{ {tick.denominator} }} \pi $"
                )
            else:
                labels.append(
                    fr"$ -\frac{{ {abs(tick.numerator)} }}{{ {tick.denominator} }} \pi $"
                )

    decimation = max(len(ticks) // 8, 1)
    ticks, labels = ticks[::decimation], labels[::decimation]

    return list(float(tick) * u.pi for tick in ticks), list(labels)


def _maybe_set_pi_ticks_and_labels(
    ax, which: str, unit: u.Unit, lower_limit, upper_limit
):
    """
    This function handles the special case where an axis's units are in radians,
    in which case we should use a special set of axis ticks with fractions-of-pi
    labels.

    Parameters
    ----------
    ax
    which
    unit
    lower_limit
    upper_limit

    Returns
    -------

    """
    if unit == "rad":
        ticks, labels = _get_pi_ticks_and_labels(lower_limit, upper_limit)
        _set_axis_ticks_and_ticklabels(ax, which, ticks, labels)


def _set_axis_ticks_and_ticklabels(
    axis: plt.Axes,
    which: str,
    ticks: Iterable[Union[float, int]],
    labels: Iterable[str],
):
    """
    Set the ticks and labels for `axis` along `direction`.

    Parameters
    ----------
    axis
        The axis to act on.
    which : {``'x'``, ``'y'``, ``'z'``}
        Which axis to act on.
    ticks
        The tick positions.
    labels
        The tick labels.
    """
    getattr(axis, f"set_{which}ticks")(ticks)
    getattr(axis, f"set_{which}ticklabels")(labels)


def _calculate_axis_limits(
    *data: np.ndarray,
    lower_limit: Optional[Union[float, int]] = None,
    upper_limit: Optional[Union[float, int]] = None,
    log: bool = False,
    pad: Union[float, int] = 0,
    log_pad: Union[float, int] = 1,
) -> (float, float):
    """
    Calculate axis limits from datasets.

    Parameters
    ----------
    data
        The data that axis limits need to be constructed for.
    lower_limit, upper_limit
        Bypass automatic construction of this axis limit, and use the given value instead.
    log
        Set ``True`` if this axis direction is going to be log-scaled.
    pad
        The fraction of the data range to pad both sides of the range by.
    log_pad
        If `log` is ``True``, the limits will be padded by this value multiplicatively (down for lower limit, up for upper limit).

    Returns
    -------
    lower_limit, upper_limit
        The lower and upper limits, in the specified units.
    """
    if lower_limit is None:
        lower_limit = min(np.nanmin(d) for d in data if len(d) > 0)
    if upper_limit is None:
        upper_limit = max(np.nanmax(d) for d in data if len(d) > 0)

    if log:
        lower_limit /= log_pad
        upper_limit *= log_pad
    else:
        limit_range = np.abs(upper_limit - lower_limit)
        lower_limit -= pad * limit_range
        upper_limit += pad * limit_range

    return lower_limit, upper_limit


def _set_axis_limits_and_scale(
    axis: plt.Axes,
    *data: np.ndarray,
    direction: str,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
    log: bool = False,
    pad: float = 0,
    log_pad: float = 1,
    unit: Optional[u.Unit] = None,
    sym_log_linear_threshold: Optional[float] = None,
) -> (float, float):
    """

    Parameters
    ----------
    axis
    data
    lower_limit
    upper_limit
    log
    pad
    log_pad
    unit
    direction

    Returns
    -------
    lower_limit, upper_limit : tuple of floats
        The lower and upper limits, in the specified units.
    """
    unit_value, _ = u.get_unit_value_and_latex_from_unit(unit)

    lower_limit, upper_limit = _calculate_axis_limits(
        *data,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        log=log,
        pad=pad,
        log_pad=log_pad,
    )

    if log:
        if lower_limit > 0:
            getattr(axis, f"set_{direction}scale")("log")
        else:
            getattr(axis, f"set_{direction}scale")(
                "symlog", **{f"linthresh{direction}": sym_log_linear_threshold}
            )

    return getattr(axis, f"set_{direction}lim")(
        lower_limit / unit_value, upper_limit / unit_value
    )


def _set_title(
    axis: plt.Axes,
    title_text: Optional[str],
    title_offset: float = 1.1,
    title_kwargs: Optional[dict] = None,
) -> Optional[plt.Text]:
    title_kwargs = collections.ChainMap(title_kwargs or {}, DEFAULT_TITLE_KWARGS)
    if title_text is not None:
        title = axis.set_title(title_text, **title_kwargs)
        title.set_y(title_offset)
        return title


def _set_axis_label(
    axis: plt.Axes,
    which: str,
    label: Optional[str] = None,
    unit: Optional[u.Unit] = None,
    label_kwargs: Optional[dict] = None,
) -> Optional[plt.Text]:
    label_kwargs = collections.ChainMap(label_kwargs or {}, DEFAULT_LABEL_KWARGS)
    unit_label = _get_unit_str_for_axis_label(unit)
    if label is not None:
        ax_label = getattr(axis, f"set_{which}label")(
            label + unit_label, **label_kwargs
        )
        return ax_label


def _get_unit_str_for_axis_label(unit: u.Unit):
    """
    Get a LaTeX-formatted unit label for `unit`.

    Parameters
    ----------
    unit
        The unit to get the formatted string for.

    Returns
    -------
    :class:`str`
        The unit label.
    """
    _, unit_tex = u.get_unit_value_and_latex_from_unit(unit)

    if unit_tex != "":
        unit_label = fr" (${unit_tex}$)"
    else:
        unit_label = ""

    return unit_label


def _attach_h_or_v_lines(
    axis: plt.Axes,
    direction: str,
    line_positions: Optional[Iterable[float]] = None,
    line_kwargs: Optional[Iterable[Dict[str, Any]]] = None,
    unit: Optional[u.Unit] = None,
):
    """

    Parameters
    ----------
    axis
        The :class:`matplotlib.pyplot.Axes` to act on.
    direction : {``'h'``, ``'v'``}
        Which direction the lines are for (horizontal ``'h'`` or vertical ``'v'``).
    line_positions
        The positions of the lines to draw.
    line_kwargs
        The keyword arguments for each draw command.
    unit
        The unit for the lines.

    Returns
    -------
    lines :
        A list containing the lines that were drawn.
    """
    unit_value, _ = u.get_unit_value_and_latex_from_unit(unit)

    lines = []

    for position, kw in itertools.zip_longest(line_positions, line_kwargs):
        if kw is None:
            kw = {}
        kw = {**HVLINE_KWARGS, **kw}
        lines.append(getattr(axis, f"ax{direction}line")(position / unit_value, **kw))

    return lines


def _add_extra_ticks(ax, which: str, extra_ticks, labels, unit: u.Unit):
    unit_value = u.UNIT_NAME_TO_VALUE[unit]

    if extra_ticks is not None:
        # append the extra tick labels, scaled appropriately
        existing_ticks = list(getattr(ax, f"get_{which}ticks"))
        new_ticks = list(np.array(extra_ticks) / unit_value)
        ax.set_xticks(existing_ticks + new_ticks)

        # replace the last set of tick labels (the ones we just added) with the custom tick labels
        if labels is not None:
            x_tick_labels = list(ax.get_xticklabels())
            x_tick_labels[-len(extra_ticks) :] = labels
            ax.set_xticklabels(x_tick_labels)
