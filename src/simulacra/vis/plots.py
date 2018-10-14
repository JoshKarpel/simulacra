import itertools
import os
import logging
import fractions
import subprocess
import collections
from typing import Union, Optional, Iterable, Collection, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .. import utils
from .. import units as u

from . import colors

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

GRID_KWARGS = dict(
    linestyle = '-',
    color = 'black',
    linewidth = .5,
    alpha = 0.4
)

MINOR_GRID_KWARGS = GRID_KWARGS.copy()
MINOR_GRID_KWARGS['alpha'] -= .2

COLORMESH_GRID_KWARGS = dict(
    linestyle = '-',
    linewidth = .5,
    alpha = 0.4,
)

LEGEND_KWARGS = dict(
    loc = 'best',
)

HVLINE_KWARGS = dict(
    linestyle = '-',
    color = 'black',
)

T_TEXT_KWARGS = dict(
    fontsize = 12,
)

CONTOUR_KWARGS = dict(

)

CONTOUR_LABEL_KWARGS = dict(
    inline = 1,
    fontsize = 8,
)

TITLE_OFFSET = 1.15

FFMPEG_PROCESS_KWARGS = dict(
    stdin = subprocess.PIPE,
    stdout = subprocess.DEVNULL,
    stderr = subprocess.DEVNULL,
    bufsize = -1,
)


class ColormapShader(utils.StrEnum):
    FLAT = 'flat'
    GOURAUD = 'gouraud'


def points_to_inches(points: Union[float, int]) -> Union[float, int]:
    """Convert the input from points to inches (72 points per inch)."""
    return points / 72.27


def inches_to_points(inches: Union[float, int]) -> Union[float, int]:
    """Convert the input from inches to points (72 points per inch)."""
    return inches * 72.27


DEFAULT_LATEX_PAGE_WIDTH = points_to_inches(350.0)
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
PPT_WIDESCREEN_WIDTH = 13.333
PPT_WIDESCREEN_HEIGHT = 7.5
PPT_WIDESCREEN_ASPECT_RATIO = 16 / 9


def _get_fig_dims(
    fig_width: Union[float, int] = DEFAULT_LATEX_PAGE_WIDTH,
    aspect_ratio: Union[float, int] = GOLDEN_RATIO,
    fig_height = None,
    fig_scale: Union[float, int] = 1,
):
    """
    Return the dimensions (width, height) for a figure based on the scale, width (in points), and aspect ratio.

    Primarily a helper function for get_figure.

    Parameters
    ----------
    fig_scale : :class:`float`
        The scale of the figure relative to the figure width.
    aspect_ratio : :class:`float`
        The aspect ratio of the figure (width / height)
    fig_height : :class:`float`
        If not `None`, overrides the aspect ratio. In inches.
    fig_width : :class:`float`
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
    fig_width : :class:`float`
        The "base" width of a figure (e.g. a LaTeX page width), in inches.
    fig_scale : :class:`float`
        The scale of the figure relative to the figure width.
    fig_dpi_scale : :class:`float`
        Multiplier for the figure DPI (only important if saving to png-like formats).
    aspect_ratio : :class:`float`
        The aspect ratio of the figure (width / height)
    fig_height : :class:`float`
        If not `None`, overrides the aspect ratio. In inches.

    Returns
    -------
    figure
        A matplotlib figure.
    """
    fig = plt.figure(
        figsize = _get_fig_dims(
            fig_width = fig_width,
            fig_height = fig_height,
            aspect_ratio = aspect_ratio,
            fig_scale = fig_scale,
        ),
        dpi = fig_dpi_scale * 100,
    )

    return fig


def save_current_figure(
    name: str,
    target_dir: Optional[str] = None,
    img_format: str = 'pdf',
    transparent: bool = True,
    tight_layout: bool = True,
) -> str:
    """
    Save the current matplotlib figure as an image to a file.

    Parameters
    ----------
    name : :class:`str`
        The name to save the image with.
    name_postfix : :class:`str`
        An additional postfix for the name.
    target_dir
        The directory to save the figure to.
    img_format
        The image format to save to.
    transparent
        If available for the format, makes the background transparent (works for ``.png``, for example).
    colormap
        A matplotlib colormap to use.
    tight_layout : :class:`bool`
        If ``True``, saves the figure with ``bbox_inches = 'tight'``.

    Returns
    -------
    :class:`str`
        The path the figure was saved to.
    """
    if target_dir is None:
        target_dir = os.getcwd()
    path = os.path.join(target_dir, utils.strip_illegal_characters(f'{name}.{img_format}'))

    utils.ensure_parents_exist(path)

    if tight_layout:
        plt.savefig(path, bbox_inches = 'tight', transparent = transparent)
    else:
        plt.savefig(path, transparent = transparent)

    logger.debug('Saved matplotlib figure {} to {}'.format(name, path))

    return path


class FigureManager:
    """
    A class that manages a matplotlib figure: creating it, showing it, saving it, and cleaning it up.

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
        target_dir: Optional[str] = None,
        img_format: str = 'pdf',
        tight_layout: bool = True,
        transparent: bool = True,
        close_before_enter: bool = True,
        close: bool = True,
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
        target_dir
            The directory to save the plot to.
        img_format
            The format for the plot.
            Accepts any matplotlib file format.
        img_scale
            The scale for the image.
        tight_layout
            If ``True``, uses matplotlib's tight layout option before saving.
        transparent
            If ``True``, and the ``img_format`` allows for it, the background of the saved image will be transparent.
        close_before_enter
            If ``True``, close whatever matplotlib plot is open before trying to create the new figure.
        close
            If ``True``, close the figure after exiting the context manager.
        save
            If ``True``, save the figure after exiting the context manager.
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

        self.target_dir = target_dir
        self.img_format = img_format

        self.close_before_enter = close_before_enter
        self.close = close
        self._save = save
        self.show = show

        self.path = None

        self.elements = {}

    def save(self):
        path = save_current_figure(
            name = self.name,
            target_dir = self.target_dir,
            img_format = self.img_format,
            tight_layout = self.tight_layout,
            transparent = self.transparent,
        )

        self.path = path

    def __enter__(self):
        if self.close_before_enter:
            plt.close()

        self.fig = get_figure(
            fig_width = self.fig_width,
            aspect_ratio = self.aspect_ratio,
            fig_height = self.fig_height,
            fig_scale = self.fig_scale,
            fig_dpi_scale = self.fig_dpi_scale,
        )

        return self

    def cleanup(self):
        if self._save:
            self.save()

        if self.show:
            plt.show()

        if self.close:
            self.fig.clear()
            plt.close(self.fig)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def get_pi_ticks_and_labels(
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

    ticks = list(fractions.Fraction(n, denom) for n in range(low * denom, (high * denom) + 1))
    labels = []
    for tick in ticks:
        if tick.numerator == 0:
            labels.append(r'$0$')
        elif tick.numerator == tick.denominator == 1:
            labels.append(r'$\pi$')
        elif tick.numerator == -1 and tick.denominator == 1:
            labels.append(r'$-\pi$')
        elif tick.denominator == 1:
            labels.append(fr'$ {tick.numerator} \pi $')
        else:
            if tick.numerator > 0:
                labels.append(fr'$ \frac{{ {tick.numerator} }}{{ {tick.denominator} }} \pi $')
            else:
                labels.append(fr'$ -\frac{{ {abs(tick.numerator)} }}{{ {tick.denominator} }} \pi $')

    decimation = max(len(ticks) // 8, 1)
    ticks, labels = ticks[::decimation], labels[::decimation]

    return list(float(tick) * u.pi for tick in ticks), list(labels)


def set_axis_ticks_and_labels(
    axis: plt.Axes,
    ticks: Iterable[Union[float, int]],
    labels: Iterable[str],
    direction: str = 'x',
):
    """
    Set the ticks and labels for `axis` along `direction`.

    Parameters
    ----------
    axis
        The axis to act on.
    ticks
        The tick positions.
    labels
        The tick labels.
    direction : {``'x'``, ``'y'``, ``'z'``}
        Which axis to act on.
    """
    getattr(axis, f'set_{direction}ticks')(ticks)
    getattr(axis, f'set_{direction}ticklabels')(labels)


def calculate_axis_limits(
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


def set_axis_limits_and_scale(
    axis: plt.Axes,
    *data: np.ndarray,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
    log: bool = False,
    pad: float = 0,
    log_pad: float = 1,
    unit: Optional[u.Unit] = None,
    direction: str = 'x',
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

    lower_limit, upper_limit = calculate_axis_limits(*data, lower_limit = lower_limit, upper_limit = upper_limit, log = log, pad = pad, log_pad = log_pad)

    if log:
        getattr(axis, f'set_{direction}scale')('log')

    return getattr(axis, f'set_{direction}lim')(lower_limit / unit_value, upper_limit / unit_value)


def set_title_and_axis_labels(
    axis,
    title: str = '',
    x_label: str = '',
    x_unit_label: str = '',
    y_label: str = '',
    y_unit_label: str = '',
    title_offset = 1.1,
    font_size_title = 16,
    font_size_axis_labels = 14,
    title_kwargs = None,
    axis_label_kwargs = None,
    x_label_kwargs = None,
    y_label_kwargs = None,
):
    title_kwargs = title_kwargs or {}
    axis_label_kwargs = axis_label_kwargs or {}
    x_label_kwargs = x_label_kwargs or {}
    y_label_kwargs = y_label_kwargs or {}

    if title is not None:
        title = axis.set_title(title, fontsize = font_size_title, **title_kwargs)
        title.set_y(title_offset)
    if x_label is not None:
        x_label = axis.set_xlabel(x_label + x_unit_label, fontsize = font_size_axis_labels, **axis_label_kwargs, **x_label_kwargs)
    if y_label is not None:
        y_label = axis.set_ylabel(y_label + y_unit_label, fontsize = font_size_axis_labels, **axis_label_kwargs, **y_label_kwargs)

    return title, x_label, y_label


def get_unit_str_for_axis_label(unit: u.Unit):
    """
    Get a LaTeX-formatted unit label for `unit`.

    Parameters
    ----------
    unit

    Returns
    -------
    :class:`str`
        The unit label.
    """
    _, unit_tex = u.get_unit_value_and_latex_from_unit(unit)

    if unit_tex != '':
        unit_label = fr' (${unit_tex}$)'
    else:
        unit_label = ''

    return unit_label


def attach_h_or_v_lines(
    axis: plt.Axes,
    line_positions = (),
    line_kwargs = (),
    unit = None,
    direction = 'h',
):
    """

    Parameters
    ----------
    line_positions
    line_kwargs
    direction : {``'h'``, ``'v'``}

    Returns
    -------

    """
    unit_value, _ = u.get_unit_value_and_latex_from_unit(unit)

    lines = []

    for position, kw in itertools.zip_longest(line_positions, line_kwargs):
        if kw is None:
            kw = {}
        kw = {**HVLINE_KWARGS, **kw}
        lines.append(getattr(axis, f'ax{direction}line')(position / unit_value, **kw))

    return lines


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
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    font_size_title: float = 15,
    font_size_axis_labels: float = 15,
    font_size_tick_labels: float = 10,
    font_size_legend: float = 12,
    ticks_on_top: bool = True,
    ticks_on_right: bool = True,
    legend_on_right: bool = False,
    legend_kwargs: Optional[dict] = None,
    grid_kwargs: Optional[dict] = None,
    minor_grid_kwargs: Optional[dict] = None,
    equal_aspect: bool = False,
    save_csv: bool = False,
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
    save_csv : :class:`bool`
        If ``True``, the x and y data for the plot will be saved to a CSV file with the same name in the target directory.
    figure_manager
        An existing :class:`FigureManager` instance to use instead of creating a new one.
    kwargs
        Keyword arguments are passed to :class:`FigureManager`.

    Returns
    -------
    :class:`FigureManager`
        The :class:`FigureManager` that the xy-plot was constructed in.
    """
    if figure_manager is None:
        figure_manager = FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = plt.subplot(111)
        if equal_aspect:
            ax.set_aspect('equal')

        grid_kwargs = collections.ChainMap(grid_kwargs or {}, GRID_KWARGS)
        minor_grid_kwargs = collections.ChainMap(minor_grid_kwargs or {}, MINOR_GRID_KWARGS)
        legend_kwargs = collections.ChainMap(legend_kwargs or {}, LEGEND_KWARGS)

        # ensure data is in numpy arrays
        x_data = np.array(x_data)
        y_data = [np.array(y) for y in y_data]
        line_labels = tuple(line_labels)
        line_kwargs = tuple(line_kwargs)

        x_unit_value, _ = u.get_unit_value_and_latex_from_unit(x_unit)
        x_unit_label = get_unit_str_for_axis_label(x_unit)

        y_unit_value, _ = u.get_unit_value_and_latex_from_unit(y_unit)
        y_unit_label = get_unit_str_for_axis_label(y_unit)

        lines = []
        for y, label, kwargs in itertools.zip_longest(y_data, line_labels, line_kwargs):
            kwargs = kwargs or {}
            label = label or ''
            lines.append(plt.plot(x_data / x_unit_value, y / y_unit_value, label = label, **kwargs)[0])
        fm.elements['lines'] = lines

        vlines = attach_h_or_v_lines(ax, vlines, vline_kwargs, unit = x_unit, direction = 'v')
        fm.elements['vlines'] = vlines
        hlines = attach_h_or_v_lines(ax, hlines, hline_kwargs, unit = y_unit, direction = 'h')
        fm.elements['hlines'] = hlines

        x_lower_limit, x_upper_limit = set_axis_limits_and_scale(
            ax, x_data,
            lower_limit = x_lower_limit,
            upper_limit = x_upper_limit,
            log = x_log_axis,
            unit = x_unit,
            direction = 'x'
        )
        y_lower_limit, y_upper_limit = set_axis_limits_and_scale(
            ax, *y_data,
            lower_limit = y_lower_limit,
            upper_limit = y_upper_limit,
            log = y_log_axis,
            pad = y_pad,
            log_pad = y_log_pad,
            unit = y_unit,
            direction = 'y'
        )

        ax.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

        if title is not None:
            title = ax.set_title(r'{}'.format(title), fontsize = font_size_title)
            title.set_y(title_offset)  # move title up a little
        if x_label is not None:
            x_label = ax.set_xlabel(r'{}'.format(x_label) + x_unit_label, fontsize = font_size_axis_labels)
        if y_label is not None:
            y_label = ax.set_ylabel(r'{}'.format(y_label) + y_unit_label, fontsize = font_size_axis_labels)
        if len(line_labels) > 0 or 'handles' in legend_kwargs:
            if not legend_on_right:
                legend = ax.legend(fontsize = font_size_legend, **legend_kwargs)
            if legend_on_right:
                legend_kwargs = collections.ChainMap(legend_kwargs, dict(loc = 'upper left', bbox_to_anchor = (1.15, 1), borderaxespad = 0, fontsize = font_size_legend, ncol = 1 + (len(line_labels) // 17)))
                legend = ax.legend(**legend_kwargs)

        fig.canvas.draw()  # draw that figure so that the ticks exist, so that we can add more ticks

        for unit, direction in zip((x_unit, y_unit), ('x', 'y')):
            if unit == 'rad':
                ticks, labels = get_pi_ticks_and_labels(x_lower_limit, x_upper_limit)
                set_axis_ticks_and_labels(ax, ticks, labels, direction = direction)

        if x_extra_ticks is not None and x_extra_tick_labels is not None:
            ax.set_xticks(list(ax.get_xticks()) + list(np.array(x_extra_ticks) / x_unit_value))  # append the extra tick labels, scaled appropriately
            x_tick_labels = list(ax.get_xticklabels())
            x_tick_labels[-len(x_extra_ticks):] = x_extra_tick_labels  # replace the last set of tick labels (the ones we just added) with the custom tick labels
            ax.set_xticklabels(x_tick_labels)

        if y_extra_ticks is not None and y_extra_tick_labels is not None:
            ax.set_yticks(list(ax.get_yticks()) + list(np.array(y_extra_ticks) / y_unit_value))  # append the extra tick labels, scaled appropriately
            y_tick_labels = list(ax.get_yticklabels())
            y_tick_labels[-len(y_extra_ticks):] = y_extra_tick_labels  # replace the last set of tick labels (the ones we just added) with the custom tick labels
            ax.set_yticklabels(y_tick_labels)

        ax.grid(True, which = 'major', **grid_kwargs)
        ax.minorticks_on()
        if x_log_axis:
            ax.grid(True, which = 'minor', axis = 'x', **minor_grid_kwargs)
        if y_log_axis:
            ax.grid(True, which = 'minor', axis = 'y', **minor_grid_kwargs)

        # set limits again to guarantee we don't see ticks outside the limits
        ax.set_xlim(x_lower_limit, x_upper_limit)
        ax.set_ylim(y_lower_limit, y_upper_limit)

        # set these AFTER adding extra tick labels so that we don't have to slice into the middle of the label lists above
        ax.tick_params(labeltop = ticks_on_top, labelright = ticks_on_right)

    if save_csv:
        path = fm.path
        csv_path = os.path.splitext(path)[0] + '.csv'
        np.savetxt(csv_path, (x_data, *y_data), delimiter = ',')

        logger.debug('Saved figure data from {} to {}'.format(name, csv_path))

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
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    font_size_title: float = 15,
    font_size_axis_labels: float = 15,
    font_size_tick_labels: float = 10,
    font_size_legend: float = 12,
    ticks_on_top: bool = True,
    ticks_on_right: bool = True,
    legend_on_right: bool = False,
    legend_kwargs: Optional[dict] = None,
    grid_kwargs: Optional[dict] = None,
    minor_grid_kwargs: Optional[dict] = None,
    equal_aspect: bool = False,
    save_csv: bool = False,
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
    save_csv : :class:`bool`
        If ``True``, the x and y data for the plot will be saved to a CSV file with the same name in the target directory.
    figure_manager
        An existing :class:`FigureManager` instance to use instead of creating a new one.
    kwargs
        Keyword arguments are passed to :class:`FigureManager`.

    Returns
    -------
    :class:`FigureManager`
        The :class:`FigureManager` that the xy-plot was constructed in.
    """
    if figure_manager is None:
        figure_manager = FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = plt.subplot(111)
        if equal_aspect:
            ax.set_aspect('equal')

        grid_kwargs = collections.ChainMap(grid_kwargs or {}, GRID_KWARGS)
        minor_grid_kwargs = collections.ChainMap(minor_grid_kwargs or {}, MINOR_GRID_KWARGS)
        legend_kwargs = collections.ChainMap(legend_kwargs or {}, LEGEND_KWARGS)

        # ensure data is in numpy arrays
        x_data = np.array(x_data)
        y_data = [np.array(y) for y in y_data]
        line_labels = tuple(line_labels)
        line_kwargs = tuple(line_kwargs)

        x_unit_value, _ = u.get_unit_value_and_latex_from_unit(x_unit)
        x_unit_label = get_unit_str_for_axis_label(x_unit)

        y_unit_value, _ = u.get_unit_value_and_latex_from_unit(y_unit)
        y_unit_label = get_unit_str_for_axis_label(y_unit)

        x = x_data / x_unit_value
        ys = [y / y_unit_value for y, label in itertools.zip_longest(y_data, line_labels)]
        line_labels = [label or '' for y, label in itertools.zip_longest(y_data, line_labels)]

        ax.stackplot(
            x,
            *ys,
            labels = line_labels,
        )

        vlines = attach_h_or_v_lines(ax, vlines, vline_kwargs, unit = x_unit, direction = 'v')
        fm.elements['vlines'] = vlines
        hlines = attach_h_or_v_lines(ax, hlines, hline_kwargs, unit = y_unit, direction = 'h')
        fm.elements['hlines'] = hlines

        x_lower_limit, x_upper_limit = set_axis_limits_and_scale(
            ax, x_data,
            lower_limit = x_lower_limit,
            upper_limit = x_upper_limit,
            log = x_log_axis,
            unit = x_unit,
            direction = 'x'
        )
        y_lower_limit, y_upper_limit = set_axis_limits_and_scale(
            ax, *y_data,
            lower_limit = y_lower_limit,
            upper_limit = y_upper_limit,
            log = y_log_axis,
            pad = y_pad,
            log_pad = y_log_pad,
            unit = y_unit,
            direction = 'y'
        )

        ax.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

        if title is not None:
            title = ax.set_title(r'{}'.format(title), fontsize = font_size_title)
            title.set_y(title_offset)  # move title up a little
        if x_label is not None:
            x_label = ax.set_xlabel(r'{}'.format(x_label) + x_unit_label, fontsize = font_size_axis_labels)
        if y_label is not None:
            y_label = ax.set_ylabel(r'{}'.format(y_label) + y_unit_label, fontsize = font_size_axis_labels)
        if len(line_labels) > 0 or 'handles' in legend_kwargs:
            if not legend_on_right:
                legend = ax.legend(fontsize = font_size_legend, **legend_kwargs)
            if legend_on_right:
                legend_kwargs = collections.ChainMap(legend_kwargs, dict(loc = 'upper left', bbox_to_anchor = (1.15, 1), borderaxespad = 0, fontsize = font_size_legend, ncol = 1 + (len(line_labels) // 17)))
                legend = ax.legend(**legend_kwargs)

        fig.canvas.draw()  # draw that figure so that the ticks exist, so that we can add more ticks

        for unit, direction in zip((x_unit, y_unit), ('x', 'y')):
            if unit == 'rad':
                ticks, labels = get_pi_ticks_and_labels(x_lower_limit, x_upper_limit)
                set_axis_ticks_and_labels(ax, ticks, labels, direction = direction)

        if x_extra_ticks is not None and x_extra_tick_labels is not None:
            ax.set_xticks(list(ax.get_xticks()) + list(np.array(x_extra_ticks) / x_unit_value))  # append the extra tick labels, scaled appropriately
            x_tick_labels = list(ax.get_xticklabels())
            x_tick_labels[-len(x_extra_ticks):] = x_extra_tick_labels  # replace the last set of tick labels (the ones we just added) with the custom tick labels
            ax.set_xticklabels(x_tick_labels)

        if y_extra_ticks is not None and y_extra_tick_labels is not None:
            ax.set_yticks(list(ax.get_yticks()) + list(np.array(y_extra_ticks) / y_unit_value))  # append the extra tick labels, scaled appropriately
            y_tick_labels = list(ax.get_yticklabels())
            y_tick_labels[-len(y_extra_ticks):] = y_extra_tick_labels  # replace the last set of tick labels (the ones we just added) with the custom tick labels
            ax.set_yticklabels(y_tick_labels)

        ax.grid(True, which = 'major', **grid_kwargs)
        ax.minorticks_on()
        if x_log_axis:
            ax.grid(True, which = 'minor', axis = 'x', **minor_grid_kwargs)
        if y_log_axis:
            ax.grid(True, which = 'minor', axis = 'y', **minor_grid_kwargs)

        # set limits again to guarantee we don't see ticks outside the limits
        ax.set_xlim(x_lower_limit, x_upper_limit)
        ax.set_ylim(y_lower_limit, y_upper_limit)

        # set these AFTER adding extra tick labels so that we don't have to slice into the middle of the label lists above
        ax.tick_params(labeltop = ticks_on_top, labelright = ticks_on_right)

    if save_csv:
        path = fm.path
        csv_path = os.path.splitext(path)[0] + '.csv'
        np.savetxt(csv_path, (x_data, *y_data), delimiter = ',')

        logger.debug('Saved figure data from {} to {}'.format(name, csv_path))

    return fm


def xxyy_plot(
    name,
    x_data,
    y_data,
    line_labels = (),
    line_kwargs = (),
    x_unit = None,
    y_unit = None,
    x_log_axis = False,
    y_log_axis = False,
    x_lower_limit = None,
    x_upper_limit = None,
    y_lower_limit = None,
    y_upper_limit = None,
    y_pad = 0,
    y_log_pad = 1,
    vlines = (),
    vline_kwargs = (),
    hlines = (),
    hline_kwargs = (),
    x_extra_ticks = None,
    y_extra_ticks = None,
    x_extra_tick_labels = None,
    y_extra_tick_labels = None,
    title = None,
    title_offset = TITLE_OFFSET,
    x_label = None,
    y_label = None,
    font_size_title = 15,
    font_size_axis_labels = 15,
    font_size_tick_labels = 10,
    font_size_legend = 12,
    ticks_on_top = True,
    ticks_on_right = True,
    legend_on_right = False,
    grid_kwargs = None,
    minor_grid_kwargs = None,
    legend_kwargs = None,
    save_csv = False,
    figure_manager = None,
    **kwargs,
) -> FigureManager:
    if figure_manager is None:
        figure_manager = FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = plt.subplot(111)

        fm.elements = {}

        grid_kwargs = collections.ChainMap(grid_kwargs or {}, GRID_KWARGS)
        minor_grid_kwargs = collections.ChainMap(minor_grid_kwargs or {}, MINOR_GRID_KWARGS)
        legend_kwargs = collections.ChainMap(legend_kwargs or {}, LEGEND_KWARGS)

        # ensure data is in numpy arrays
        x_data = [np.array(x) for x in x_data]
        y_data = [np.array(y) for y in y_data]
        line_labels = tuple(line_labels)
        line_kwargs = tuple(line_kwargs)

        x_unit_value, x_unit_tex = u.get_unit_value_and_latex_from_unit(x_unit)
        x_unit_label = get_unit_str_for_axis_label(x_unit)

        y_unit_value, y_unit_tex = u.get_unit_value_and_latex_from_unit(y_unit)
        y_unit_label = get_unit_str_for_axis_label(y_unit)

        lines = []
        for x, y, lab, kw in itertools.zip_longest(x_data, y_data, line_labels, line_kwargs):
            kw = kw or {}
            lab = lab or ''
            lines.append(plt.plot(x / x_unit_value, y / y_unit_value, label = lab, **kw)[0])
        fm.elements['lines'] = lines

        attach_h_or_v_lines(ax, vlines, vline_kwargs, unit = x_unit, direction = 'v')
        attach_h_or_v_lines(ax, hlines, hline_kwargs, unit = y_unit, direction = 'h')

        x_lower_limit, x_upper_limit = set_axis_limits_and_scale(
            ax, *x_data,
            lower_limit = x_lower_limit,
            upper_limit = x_upper_limit,
            log = x_log_axis,
            pad = 0,
            log_pad = 1,
            unit = x_unit,
            direction = 'x'
        )
        y_lower_limit, y_upper_limit = set_axis_limits_and_scale(
            ax, *y_data,
            lower_limit = y_lower_limit,
            upper_limit = y_upper_limit,
            log = y_log_axis,
            pad = y_pad,
            log_pad = y_log_pad,
            unit = y_unit,
            direction = 'y'
        )

        ax.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

        # make title, axis labels, and legend
        if title is not None:
            title = ax.set_title(r'{}'.format(title), fontsize = font_size_title)
            title.set_y(title_offset)  # move title up a little
        if x_label is not None:
            x_label = ax.set_xlabel(r'{}'.format(x_label) + x_unit_label, fontsize = font_size_axis_labels)
        if y_label is not None:
            y_label = ax.set_ylabel(r'{}'.format(y_label) + y_unit_label, fontsize = font_size_axis_labels)
        if len(line_labels) > 0 or 'handles' in legend_kwargs:
            if not legend_on_right:
                legend = ax.legend(fontsize = font_size_legend, **legend_kwargs)
            if legend_on_right:
                legend_kwargs['loc'] = 'upper left'
                legend = ax.legend(bbox_to_anchor = (1.15, 1), borderaxespad = 0., fontsize = font_size_legend, ncol = 1 + (len(line_labels) // 17), **legend_kwargs)

        fig.canvas.draw()  # draw that figure so that the ticks exist, so that we can add more ticks

        for unit, direction in zip((x_unit, y_unit), ('x', 'y')):
            if unit == 'rad':
                ticks, labels = get_pi_ticks_and_labels(x_lower_limit, x_upper_limit)
                set_axis_ticks_and_labels(ax, ticks, labels, direction = direction)

        if x_extra_ticks is not None and x_extra_tick_labels is not None:
            ax.set_xticks(list(ax.get_xticks()) + list(np.array(x_extra_ticks) / x_unit_value))  # append the extra tick labels, scaled appropriately
            x_tick_labels = list(ax.get_xticklabels())
            x_tick_labels[-len(x_extra_ticks):] = x_extra_tick_labels  # replace the last set of tick labels (the ones we just added) with the custom tick labels
            ax.set_xticklabels(x_tick_labels)

        if y_extra_ticks is not None and y_extra_tick_labels is not None:
            ax.set_yticks(list(ax.get_yticks()) + list(np.array(y_extra_ticks) / y_unit_value))  # append the extra tick labels, scaled appropriately
            y_tick_labels = list(ax.get_yticklabels())
            y_tick_labels[-len(y_extra_ticks):] = y_extra_tick_labels  # replace the last set of tick labels (the ones we just added) with the custom tick labels
            ax.set_yticklabels(y_tick_labels)

        ax.grid(True, which = 'major', **grid_kwargs)
        if x_log_axis:
            ax.grid(True, which = 'minor', axis = 'x', **minor_grid_kwargs)
        if y_log_axis:
            ax.grid(True, which = 'minor', axis = 'y', **minor_grid_kwargs)

        # set limits again to guarantee we don't see ticks oustide the limits
        ax.set_xlim(x_lower_limit, x_upper_limit)
        ax.set_ylim(y_lower_limit, y_upper_limit)

        # set these AFTER adding extra tick labels so that we don't have to slice into the middle of the label lists above
        ax.tick_params(labeltop = ticks_on_top, labelright = ticks_on_right)

    path = fm.path

    if save_csv:
        csv_path = os.path.splitext(path)[0] + '.csv'
        np.savetxt(csv_path, (x_data, *y_data), delimiter = ',')

        logger.debug('Saved figure data from {} to {}'.format(name, csv_path))

    return fm


def xyz_plot(
    name,
    x_mesh,
    y_mesh,
    z_mesh,
    x_unit = None,
    y_unit = None,
    z_unit = None,
    x_log_axis = False,
    y_log_axis = False,
    z_log_axis = False,
    x_lower_limit = None,
    x_upper_limit = None,
    y_lower_limit = None,
    y_upper_limit = None,
    z_lower_limit = None,
    z_upper_limit = None,
    z_pad = 0,
    z_log_pad = 1,
    x_extra_ticks = None,
    y_extra_ticks = None, x_extra_tick_labels = None, y_extra_tick_labels = None,
    title = None,
    x_label = None,
    y_label = None,
    z_label = None,
    font_size_title = 15,
    font_size_axis_labels = 15,
    font_size_tick_labels = 10,
    ticks_on_top = True,
    ticks_on_right = True,
    grid_kwargs = None,
    minor_grid_kwargs = None,
    contours = (),
    contour_kwargs = None,
    show_contour_labels = True,
    contour_label_kwargs = None,
    save_csv = False,
    colormap = plt.get_cmap('viridis'),
    shading = 'flat', show_colorbar = True,
    richardson_equator_magnitude = 1,
    sym_log_norm_epsilon = 1e-3,
    figure_manager = None,
    **kwargs,
) -> FigureManager:
    if figure_manager is None:
        figure_manager = FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = plt.subplot(111)

        grid_kwargs = collections.ChainMap(grid_kwargs or {}, GRID_KWARGS)
        minor_grid_kwargs = collections.ChainMap(minor_grid_kwargs or {}, MINOR_GRID_KWARGS)

        contour_kwargs = collections.ChainMap(contour_kwargs or {}, CONTOUR_KWARGS)
        contour_label_kwargs = collections.ChainMap(contour_label_kwargs or {}, CONTOUR_LABEL_KWARGS)

        grid_color = colors.CMAP_TO_OPPOSITE.get(colormap, 'black')
        grid_kwargs['color'] = grid_color
        minor_grid_kwargs['color'] = grid_color

        plt.set_cmap(colormap)

        x_unit_value, x_unit_name = u.get_unit_value_and_latex_from_unit(x_unit)
        x_unit_label = get_unit_str_for_axis_label(x_unit)

        y_unit_value, y_unit_name = u.get_unit_value_and_latex_from_unit(y_unit)
        y_unit_label = get_unit_str_for_axis_label(y_unit)

        z_unit_value, z_unit_name = u.get_unit_value_and_latex_from_unit(z_unit)
        z_unit_label = get_unit_str_for_axis_label(z_unit)

        x_lower_limit, x_upper_limit = set_axis_limits_and_scale(
            ax, x_mesh,
            direction = 'x',
            lower_limit = x_lower_limit,
            upper_limit = x_upper_limit,
            log = x_log_axis,
            pad = 0,
            log_pad = 1,
            unit = x_unit,
        )
        y_lower_limit, y_upper_limit = set_axis_limits_and_scale(
            ax, y_mesh,
            direction = 'y',
            lower_limit = y_lower_limit,
            upper_limit = y_upper_limit,
            log = y_log_axis,
            pad = 0,
            log_pad = 1,
            unit = y_unit,
        )

        if isinstance(colormap, colors.RichardsonColormap):
            norm = colors.RichardsonNormalization(
                equator_magnitude = richardson_equator_magnitude
            )
        else:
            z_lower_limit, z_upper_limit = calculate_axis_limits(
                z_mesh,
                lower_limit = z_lower_limit,
                upper_limit = z_upper_limit,
                log = z_log_axis,
                pad = z_pad,
                log_pad = z_log_pad,
            )

            norm_kwargs = dict(
                vmin = z_lower_limit / z_unit_value,
                vmax = z_upper_limit / z_unit_value,
            )

            if z_log_axis:
                if z_lower_limit > 0:
                    norm = matplotlib.colors.LogNorm(
                        **norm_kwargs,
                    )
                else:
                    norm = matplotlib.colors.SymLogNorm(
                        ((np.abs(z_lower_limit) + np.abs(z_upper_limit)) / 2) * sym_log_norm_epsilon,
                        **norm_kwargs,
                    )
            else:
                norm = matplotlib.colors.Normalize(
                    **norm_kwargs,
                )

        colormesh = ax.pcolormesh(
            x_mesh / x_unit_value,
            y_mesh / y_unit_value,
            z_mesh / z_unit_value,
            shading = shading,
            norm = norm
        )

        if len(contours) > 0:
            contour = ax.contour(
                x_mesh / x_unit_value,
                y_mesh / y_unit_value,
                z_mesh / z_unit_value,
                levels = np.array(sorted(contours)) / z_unit_value,
                **contour_kwargs,
            )
            if show_contour_labels:
                ax.clabel(contour, **contour_label_kwargs)

        ax.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

        # make title, axis labels, and legend
        if title is not None:
            title = ax.set_title(r'{}'.format(title), fontsize = font_size_title)
            title.set_y(TITLE_OFFSET)  # move title up a little
        if x_label is not None:
            x_label = ax.set_xlabel(r'{}'.format(x_label) + x_unit_label, fontsize = font_size_axis_labels)
        if y_label is not None:
            y_label = ax.set_ylabel(r'{}'.format(y_label) + y_unit_label, fontsize = font_size_axis_labels)

        if show_colorbar and colormap.name != 'richardson':
            cbar = plt.colorbar(mappable = colormesh, ax = ax, pad = 0.1)
            if z_label is not None:
                z_label = cbar.set_label(r'{}'.format(z_label) + z_unit_label, fontsize = font_size_axis_labels)

        fig.canvas.draw()  # draw that figure so that the ticks exist, so that we can add more ticks

        if x_unit == 'rad':
            ticks, labels = get_pi_ticks_and_labels(x_lower_limit, x_upper_limit)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
        if y_unit == 'rad':
            ticks, labels = get_pi_ticks_and_labels(y_lower_limit, y_upper_limit)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)

        if x_extra_ticks is not None and x_extra_tick_labels is not None:
            ax.set_xticks(list(ax.get_xticks()) + list(np.array(x_extra_ticks) / x_unit))  # append the extra tick labels, scaled appropriately
            x_tick_labels = list(ax.get_xticklabels())
            x_tick_labels[-len(x_extra_ticks):] = x_extra_tick_labels  # replace the last set of tick labels (the ones we just added) with the custom tick labels
            ax.set_xticklabels(x_tick_labels)

        if y_extra_ticks is not None and y_extra_tick_labels is not None:
            ax.set_yticks(list(ax.get_yticks()) + list(np.array(y_extra_ticks) / y_unit_value))  # append the extra tick labels, scaled appropriately
            y_tick_labels = list(ax.get_yticklabels())
            y_tick_labels[-len(y_extra_ticks):] = y_extra_tick_labels  # replace the last set of tick labels (the ones we just added) with the custom tick labels
            ax.set_yticklabels(y_tick_labels)

        ax.grid(True, which = 'major', **grid_kwargs)
        if x_log_axis:
            ax.grid(True, which = 'minor', axis = 'x', **minor_grid_kwargs)
        if y_log_axis:
            ax.grid(True, which = 'minor', axis = 'y', **minor_grid_kwargs)

        # set limits again to guarantee we don't see ticks oustide the limits
        ax.set_xlim(x_lower_limit, x_upper_limit)
        ax.set_ylim(y_lower_limit, y_upper_limit)

        # set these AFTER adding extra tick labels so that we don't have to slice into the middle of the label lists above
        ax.tick_params(labeltop = ticks_on_top, labelright = ticks_on_right)

    path = fm.path

    if save_csv:
        raise NotImplementedError
        # csv_path = os.path.splitext(path)[0] + '.csv'
        # np.savetxt(csv_path, (x_data, *y_data), delimiter = ',')
        #
        # logger.debug('Saved figure data from {} to {}'.format(name, csv_path))

    return fm
