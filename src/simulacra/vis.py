"""
Simulacra visualization sub-package.


Copyright 2017 Josh Karpel

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import itertools
import os
import logging
import fractions
import subprocess
import sys
from typing import Union, Optional, Iterable

import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import utils
from . import core
from .units import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# named colors
WHITE = '#ffffff'
BLACK = '#000000'

BLUE = '#1f77b4'  # matplotlib C0
ORANGE = '#ff7f0e'  # matplotlib C1
GREEN = '#2ca02c'  # matplotlib C2
RED = '#d62728'  # matplotlib C3
PURPLE = '#9467bd'  # matplotlib C4
BROWN = '#8c564b'  # matplotlib C5
PINK = '#e377c2'  # matplotlib C6
GRAY = '#7f7f7f'  # matplotlib C7
YELLOW = '#bcbd22'  # matplotlib C8
TEAL = '#17becf'  # matplotlib C9

# colors opposite common colormaps
COLOR_OPPOSITE_PLASMA = GREEN
COLOR_OPPOSITE_INFERNO = GREEN
COLOR_OPPOSITE_MAGMA = GREEN
COLOR_OPPOSITE_VIRIDIS = RED

CMAP_TO_OPPOSITE = {
    plt.get_cmap('viridis'): COLOR_OPPOSITE_VIRIDIS,
    'viridis': COLOR_OPPOSITE_VIRIDIS,
    plt.get_cmap('plasma'): COLOR_OPPOSITE_PLASMA,
    'plasma': COLOR_OPPOSITE_PLASMA,
    plt.get_cmap('inferno'): COLOR_OPPOSITE_INFERNO,
    'inferno': COLOR_OPPOSITE_INFERNO,
    plt.get_cmap('magma'): COLOR_OPPOSITE_MAGMA,
    'magma': COLOR_OPPOSITE_MAGMA,
}

GRID_KWARGS = dict(
    linestyle = '-',
    color = 'black',
    linewidth = .25,
    alpha = 0.4
)

MINOR_GRID_KWARGS = GRID_KWARGS.copy()
MINOR_GRID_KWARGS['alpha'] -= .1

COLORMESH_GRID_KWARGS = dict(
    linestyle = '-',
    linewidth = .25,
    alpha = 0.4,
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


def _get_fig_dims(fig_width: Union[float, int] = DEFAULT_LATEX_PAGE_WIDTH,
                  aspect_ratio: Union[float, int] = GOLDEN_RATIO,
                  fig_height = None,
                  fig_scale: Union[float, int] = 1):
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


def get_figure(fig_width: Union[float, int] = DEFAULT_LATEX_PAGE_WIDTH,
               aspect_ratio: Union[float, int] = GOLDEN_RATIO,
               fig_height = None,
               fig_scale: Union[float, int, str] = 1,
               fig_dpi_scale: Union[float, int] = 1):
    """
    Get a matplotlib figure object with the desired scale relative to a full-text-width LaTeX page.

    Special scales:
    ``scale = 'full'`` -> ``scale = 0.95``
    ``scale = 'half'`` -> ``scale = 0.475``

    Parameters
    ----------
    fig_scale : :class:`float`
        The scale of the figure relative to the figure width.
    fig_dpi_scale : :class:`float`
        Multiplier for the figure DPI (only important if saving to png-like formats).
    aspect_ratio : :class:`float`
        The aspect ratio of the figure (width / height)
    fig_height : :class:`float`
        If not `None`, overrides the aspect ratio. In inches.
    fig_width : :class:`float`
        The "base" width of a figure (e.g. a LaTeX page width). In inches.

    Returns
    -------
    figure
        A matplotlib figure.
    """
    fig = plt.figure(figsize = _get_fig_dims(fig_width = fig_width, fig_height = fig_height, aspect_ratio = aspect_ratio, fig_scale = fig_scale), dpi = fig_dpi_scale * 100)

    return fig


def save_current_figure(name: str,
                        name_postfix: str = '',
                        target_dir: Optional[str] = None,
                        img_format: str = 'pdf',
                        transparent: bool = True,
                        tight_layout: bool = True,
                        **kwargs):
    """
    Save the current matplotlib figure as an image to a file.

    Parameters
    ----------
    name : :class:`str`
        The name to save the image with.
    name_postfix : :class:`str`
        An additional postfix for the name.
    target_dir
        The directory to save the image to.
    img_format
        The image format to save to.
    transparent
        If available for the format, makes the background transparent (works for ``.png``, for example).
    colormap
        A matplotlib colormap to use.
    tight_layout : :class:`bool`
        If ``True``, saves the figure with ``bbox_inches = 'tight'``.
    kwargs
        This function absorbs keyword arguments silently.

    Returns
    -------
    :class:`str`
        The path the figure was saved to.
    """
    if target_dir is None:
        target_dir = os.getcwd()
    path = os.path.join(target_dir, utils.strip_illegal_characters('{}{}.{}'.format(name, name_postfix, img_format)))

    utils.ensure_dir_exists(path)

    if tight_layout:
        plt.savefig(path, bbox_inches = 'tight', transparent = transparent)
    else:
        plt.savefig(path, transparent = transparent)

    logger.debug('Saved matplotlib figure {} to {}'.format(name, path))

    return path


class FigureManager:
    """
    A class that manages a matplotlib figure: creating it, showing it, saving it, and cleaning it up.
    """

    def __init__(self, name: str,
                 name_postfix: str = '',
                 fig_width = DEFAULT_LATEX_PAGE_WIDTH,
                 aspect_ratio = GOLDEN_RATIO,
                 fig_height = None,
                 fig_scale = 1,
                 fig_dpi_scale = 1,
                 target_dir: Optional[str] = None,
                 img_format: str = 'pdf',
                 tight_layout: bool = True,
                 close_before_enter: bool = True,
                 close_after_exit: bool = True,
                 save_on_exit: bool = True,
                 show: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        name
            The desired file name for the plot.
        name_postfix
            An additional postfix for the file name.
        fig_scale
            The relative width of the figure.
        fig_dpi_scale
            The relative point density of the figure (i.e., 2 is twice as much dpi as normal).
        fig_width_pts
            The width of the figure, in points (72 points per inch).
        aspect_ratio
            The aspect ratio of the figure (height / width).
        target_dir
            The directory to save the plot to.
        img_format
            The format for the plot. Accepts any matplotlib file format.
        img_scale
            The scale for the image.
        tight_layout
            If ``True``, uses matplotlib's tight layout option before saving.
        close_before_enter
            If ``True``, close whatever matplotlib plot is open before trying to create the new figure.
        close_after_exit
            If ``True``, close the figure after exiting the context manager.
        save_on_exit
            If ``True``, save the figure after exiting the context manager.
        show
            If ``True``, show the figure after exiting the context manager.
        kwargs
            Keyword arguments are silently absorbed.
        """
        self.name = name
        self.name_postfix = name_postfix

        self.fig_scale = fig_scale
        self.fig_dpi_scale = fig_dpi_scale
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.aspect_ratio = aspect_ratio
        self.tight_layout = tight_layout

        self.target_dir = target_dir
        self.img_format = img_format

        if len(kwargs) > 0:
            logger.debug('FigureManager for figure {} absorbed extraneous kwargs: {}'.format(self.name, kwargs))

        self.close_before_enter = close_before_enter
        self.close_after_exit = close_after_exit

        self.save_on_exit = save_on_exit
        self.show = show

        self.path = None

    def save(self):
        path = save_current_figure(
            name = self.name,
            name_postfix = self.name_postfix,
            target_dir = self.target_dir,
            img_format = self.img_format,
            tight_layout = self.tight_layout
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
        if self.save_on_exit:
            self.save()

        if self.show:
            plt.show()

        if self.close_after_exit:
            self.fig.clear()
            plt.close(self.fig)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def get_pi_ticks_and_labels(lower_limit: Union[float, int] = 0, upper_limit: Union[float, int] = twopi, denom: int = 4):
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
    low = int(np.floor(lower_limit / pi))
    high = int(np.ceil(upper_limit / pi))

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

    return list(float(tick) * pi for tick in ticks), list(labels)


def set_axis_ticks_and_labels(axis, ticks: Iterable[Union[float, int]], labels: Iterable[str], direction: str = 'x'):
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


def get_axis_limits(*data,
                    lower_limit: Optional[Union[float, int]] = None,
                    upper_limit: Optional[Union[float, int]] = None,
                    log: bool = False,
                    pad: Union[float, int] = 0,
                    log_pad: Union[float, int] = 1):
    """
    Calculate axis limits from datasets.

    Parameters
    ----------
    data : any number of numpy arrays
        The data that axis limits need to be constructed for.
    lower_limit, upper_limit : :class:`float`
        Bypass automatic construction of this axis limit, and use the given value instead.
    log : :class:`bool`
        Set ``True`` if this axis direction is going to be log-scaled.
    pad : :class:`float`
        The fraction of the data range to pad both sides of the range by.
    log_pad : :class:`float`
        If `log` is ``True``, the limits will be padded by this value multiplicatively (down for lower limit, up for upper limit).

    Returns
    -------
    lower_limit, upper_limit : tuple of floats
        The lower and upper limits, in the specified units.
    """
    if lower_limit is None:
        lower_limit = min(np.nanmin(d) for d in data)
    if upper_limit is None:
        upper_limit = max(np.nanmax(d) for d in data)

    if not log:
        limit_range = np.abs(upper_limit - lower_limit)
        lower_limit -= pad * limit_range
        upper_limit += pad * limit_range

    if log:
        lower_limit /= log_pad
        upper_limit *= log_pad

    return lower_limit, upper_limit


def set_axis_limits(axis, *data, lower_limit = None, upper_limit = None, log = False, pad = 0, log_pad = 1, unit = None, direction = 'x'):
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
    unit_value, _ = get_unit_value_and_latex_from_unit(unit)

    lower_limit, upper_limit = get_axis_limits(*data, lower_limit = lower_limit, upper_limit = upper_limit, log = log, pad = pad, log_pad = log_pad)

    if log:
        getattr(axis, f'set_{direction}scale')('log')

    return getattr(axis, f'set_{direction}lim')(lower_limit / unit_value, upper_limit / unit_value)


def get_unit_label(unit):
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
    _, unit_tex = get_unit_value_and_latex_from_unit(unit)

    if unit_tex != '':
        unit_label = fr' (${unit_tex}$)'
    else:
        unit_label = ''

    return unit_label


def attach_hv_lines(axis, line_positions = (), line_kwargs = (), unit = None, direction = 'h'):
    """

    Parameters
    ----------
    line_positions
    line_kwargs
    direction : {``'h'``, ``'v'``}

    Returns
    -------

    """
    unit_value, _ = get_unit_value_and_latex_from_unit(unit)

    for position, kw in itertools.zip_longest(line_positions, line_kwargs):
        if kw is None:
            kw = {}
        kw = {**HVLINE_KWARGS, **kw}
        getattr(axis, f'ax{direction}line')(position / unit_value, **kw)


def xy_plot(name,
            x_data, *y_data,
            line_labels = (), line_kwargs = (),
            figure_manager = None,
            x_unit = None, y_unit = None,
            x_log_axis = False, y_log_axis = False,
            x_lower_limit = None, x_upper_limit = None, y_lower_limit = None, y_upper_limit = None, y_pad = 0.05, y_log_pad = 10,
            vlines = (), vline_kwargs = (), hlines = (), hline_kwargs = (),
            x_extra_ticks = None, y_extra_ticks = None, x_extra_tick_labels = None, y_extra_tick_labels = None,
            title = None, x_label = None, y_label = None,
            font_size_title = 15, font_size_axis_labels = 15, font_size_tick_labels = 10, font_size_legend = 12,
            ticks_on_top = True, ticks_on_right = True, legend_on_right = False,
            grid_kwargs = None, minor_grid_kwargs = None,
            save_csv = False,
            **kwargs):
    """
    Generate and save a generic x vs. y plot.

    Parameters
    ----------
    name : :class:`str`
        The filename for the plot (not including path, which should be passed via keyword argument `target_dir`.
    x_data
        A single array that will be used as x-values for all the `y_data`.
    y_data
        Any number of arrays of the same length as `x_data`, each of which will appear as a line on the plot.
    line_labels
        Labels for each of the `y_data` lines.
    line_kwargs
        Keyword arguments for each of the `y_data` lines (a list of dictionaries).
    figure_manager
        An existing :class:`FigureManager` instance to use instead of creating a new one.
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
        The linear padding factor for the y-axis. See :func:`get_axis_limits`.
    y_log_pad
        The logarithmic padding factor for the y-axis. See :func:`get_axis_limits`.
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
    title : :class:`str`
        The text to display above the plot.
    x_label : :class:`str`
        The label to display below the x-axis.
    y_label : :class:`str`
        The label to display to the left of the y-axis.
    font_size_title : :class:`float`
        The font size for the title.
    font_size_axis_labels : :class:`float`
        The font size for the axis labels.
    font_size_tick_labels : :class:`float`
        The font size for the tick labels.
    font_size_legend : :class:`float`
        The font size for the legend.
    ticks_on_top : :class:`bool`
        If ``True``, axis ticks will be shown along the top side of the plot (in addition to the bottom).
    ticks_on_right : :class:`bool`
        If ``True``, axis ticks will be shown along the right side of the plot (in addition to the left).
    legend_on_right : :class:`bool`
        If ``True``, the legend will be displayed hanging on the right side of the plot.
    grid_kwargs : :class:`dict`
        Keyword arguments for the major gridlines.
    minor_grid_kwargs : :class:`dict`
        Keyword arguments for the minor gridlines.
    save_csv : :class:`bool`
        If ``True``, the x and y data for the plot will be saved to a CSV file with the same name in the target directory.
    kwargs
        Keyword arguments are passed to :class:`FigureManager`.

    Returns
    -------
    :class:`FigureManager`
        The :class:`FigureManager` that the xy-plot was constructed in.
    """
    # set up figure and axis
    if figure_manager is None:
        figure_manager = FigureManager(name, **kwargs)
    with figure_manager as fm:
        fm.elements = {}

        fig = fm.fig
        ax = plt.subplot(111)

        if grid_kwargs is None:
            grid_kwargs = {}
        if minor_grid_kwargs is None:
            minor_grid_kwargs = {}

        grid_kwargs = {**GRID_KWARGS, **grid_kwargs}
        minor_grid_kwargs = {**MINOR_GRID_KWARGS, **minor_grid_kwargs}

        # ensure data is in numpy arrays
        x_data = np.array(x_data)
        y_data = [np.array(y) for y in y_data]
        line_labels = tuple(line_labels)
        line_kwargs = tuple(line_kwargs)

        x_unit_value, x_unit_tex = get_unit_value_and_latex_from_unit(x_unit)
        x_unit_label = get_unit_label(x_unit)

        y_unit_value, y_unit_tex = get_unit_value_and_latex_from_unit(y_unit)
        y_unit_label = get_unit_label(y_unit)

        lines = []
        for y, lab, kw in itertools.zip_longest(y_data, line_labels, line_kwargs):
            if kw is None:
                kw = {}
            lines.append(plt.plot(x_data / x_unit_value, y / y_unit_value, label = lab, **kw)[0])
        fm.elements['lines'] = lines

        attach_hv_lines(ax, vlines, vline_kwargs, unit = x_unit, direction = 'v')
        attach_hv_lines(ax, hlines, hline_kwargs, unit = y_unit, direction = 'h')

        x_lower_limit, x_upper_limit = set_axis_limits(ax, x_data,
                                                       lower_limit = x_lower_limit, upper_limit = x_upper_limit,
                                                       log = x_log_axis,
                                                       pad = 0, log_pad = 1,
                                                       unit = x_unit, direction = 'x')
        y_lower_limit, y_upper_limit = set_axis_limits(ax, *y_data,
                                                       lower_limit = y_lower_limit, upper_limit = y_upper_limit,
                                                       log = y_log_axis,
                                                       pad = y_pad, log_pad = y_log_pad,
                                                       unit = y_unit, direction = 'y')

        ax.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

        # make title, axis labels, and legend
        if title is not None:
            title = ax.set_title(r'{}'.format(title), fontsize = font_size_title)
            title.set_y(TITLE_OFFSET)  # move title up a little
        if x_label is not None:
            x_label = ax.set_xlabel(r'{}'.format(x_label) + x_unit_label, fontsize = font_size_axis_labels)
        if y_label is not None:
            y_label = ax.set_ylabel(r'{}'.format(y_label) + y_unit_label, fontsize = font_size_axis_labels)
        if len(line_labels) > 0:
            if not legend_on_right:
                legend = ax.legend(loc = 'best', fontsize = font_size_legend)
            if legend_on_right:
                legend = ax.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left', borderaxespad = 0., fontsize = font_size_legend, ncol = 1 + (len(line_labels) // 17))

        fig.canvas.draw()  # draw that figure so that the ticks exist, so that we can add more ticks

        if x_unit == 'rad':
            ticks, labels = get_pi_ticks_and_labels(x_lower_limit, x_upper_limit)
            set_axis_ticks_and_labels(ax, ticks, labels, direction = 'x')
        if y_unit == 'rad':
            ticks, labels = get_pi_ticks_and_labels(y_lower_limit, y_upper_limit)
            set_axis_ticks_and_labels(ax, ticks, labels, direction = 'y')

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


def xxyy_plot(name,
              x_data, y_data,
              line_labels = (), line_kwargs = (),
              figure_manager = None,
              x_unit = None, y_unit = None,
              x_log_axis = False, y_log_axis = False,
              x_lower_limit = None, x_upper_limit = None, y_lower_limit = None, y_upper_limit = None, y_pad = .05, y_log_pad = 10,
              vlines = (), vline_kwargs = (), hlines = (), hline_kwargs = (),
              x_extra_ticks = None, y_extra_ticks = None, x_extra_tick_labels = None, y_extra_tick_labels = None,
              title = None, x_label = None, y_label = None,
              font_size_title = 15, font_size_axis_labels = 15, font_size_tick_labels = 10, font_size_legend = 12,
              ticks_on_top = True, ticks_on_right = True, legend_on_right = False,
              grid_kwargs = None, minor_grid_kwargs = None,
              save_csv = False,
              **kwargs):
    # set up figure and axis
    if figure_manager is None:
        figure_manager = FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = plt.subplot(111)

        if grid_kwargs is None:
            grid_kwargs = {}
        if minor_grid_kwargs is None:
            minor_grid_kwargs = {}

        grid_kwargs = {**GRID_KWARGS, **grid_kwargs}
        minor_grid_kwargs = {**MINOR_GRID_KWARGS, **minor_grid_kwargs}

        # ensure data is in numpy arrays
        x_data = [np.array(x) for x in x_data]
        y_data = [np.array(y) for y in y_data]
        line_labels = tuple(line_labels)
        line_kwargs = tuple(line_kwargs)

        x_unit_value, x_unit_tex = get_unit_value_and_latex_from_unit(x_unit)
        x_unit_label = get_unit_label(x_unit)

        y_unit_value, y_unit_tex = get_unit_value_and_latex_from_unit(y_unit)
        y_unit_label = get_unit_label(y_unit)

        lines = []
        for x, y, lab, kw in itertools.zip_longest(x_data, y_data, line_labels, line_kwargs):
            if kw is None:
                kw = {}
            lines.append(plt.plot(x / x_unit_value, y / y_unit_value, label = lab, **kw)[0])

        attach_hv_lines(ax, vlines, vline_kwargs, unit = x_unit, direction = 'v')
        attach_hv_lines(ax, hlines, hline_kwargs, unit = y_unit, direction = 'h')

        x_lower_limit, x_upper_limit = set_axis_limits(ax, *x_data,
                                                       lower_limit = x_lower_limit, upper_limit = x_upper_limit,
                                                       log = x_log_axis,
                                                       pad = 0, log_pad = 1,
                                                       unit = x_unit, direction = 'x')
        y_lower_limit, y_upper_limit = set_axis_limits(ax, *y_data,
                                                       lower_limit = y_lower_limit, upper_limit = y_upper_limit,
                                                       log = y_log_axis,
                                                       pad = y_pad, log_pad = y_log_pad,
                                                       unit = y_unit, direction = 'y')

        ax.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

        # make title, axis labels, and legend
        if title is not None:
            title = ax.set_title(r'{}'.format(title), fontsize = font_size_title)
            title.set_y(TITLE_OFFSET)  # move title up a little
        if x_label is not None:
            x_label = ax.set_xlabel(r'{}'.format(x_label) + x_unit_label, fontsize = font_size_axis_labels)
        if y_label is not None:
            y_label = ax.set_ylabel(r'{}'.format(y_label) + y_unit_label, fontsize = font_size_axis_labels)
        if len(line_labels) > 0:
            if not legend_on_right:
                legend = ax.legend(loc = 'best', fontsize = font_size_legend)
            if legend_on_right:
                legend = ax.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left', borderaxespad = 0., fontsize = font_size_legend, ncol = 1 + (len(line_labels) // 17))

        fig.canvas.draw()  # draw that figure so that the ticks exist, so that we can add more ticks

        if x_unit == 'rad':
            ticks, labels = get_pi_ticks_and_labels(x_lower_limit, x_upper_limit)
            set_axis_ticks_and_labels(ax, ticks, labels, direction = 'x')
        if y_unit == 'rad':
            ticks, labels = get_pi_ticks_and_labels(y_lower_limit, y_upper_limit)
            set_axis_ticks_and_labels(ax, ticks, labels, direction = 'y')

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


def xyz_plot(name,
             x_mesh, y_mesh, z_mesh,
             figure_manager = None,
             x_unit = None, y_unit = None, z_unit = None,
             x_log_axis = False, y_log_axis = False, z_log_axis = False,
             x_lower_limit = None, x_upper_limit = None, y_lower_limit = None, y_upper_limit = None, z_lower_limit = None, z_upper_limit = None,
             z_pad = 0, z_log_pad = 1,
             x_extra_ticks = None, y_extra_ticks = None, x_extra_tick_labels = None, y_extra_tick_labels = None,
             z_label = None, x_label = None, y_label = None,
             font_size_title = 15, font_size_axis_labels = 15, font_size_tick_labels = 10,
             ticks_on_top = True, ticks_on_right = True,
             grid_kwargs = None, minor_grid_kwargs = None,
             contours = (), contour_kwargs = None, show_contour_labels = True, contour_label_kwargs = None,
             save_csv = False,
             colormap = plt.get_cmap('viridis'),
             shading = 'gouraud', show_colorbar = True,
             richardson_equator_magnitude = 1,
             sym_log_norm_epsilon = 1e-3,
             **kwargs):
    # set up figure and axis
    if figure_manager is None:
        figure_manager = FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = plt.subplot(111)

        if grid_kwargs is None:
            grid_kwargs = {}
        if minor_grid_kwargs is None:
            minor_grid_kwargs = {}

        if contour_kwargs is None:
            contour_kwargs = {}
        if contour_label_kwargs is None:
            contour_label_kwargs = {}

        grid_color = CMAP_TO_OPPOSITE.get(colormap, 'black')
        grid_kwargs['color'] = grid_color
        minor_grid_kwargs['color'] = grid_color
        grid_kwargs = {**GRID_KWARGS, **grid_kwargs}
        minor_grid_kwargs = {**MINOR_GRID_KWARGS, **minor_grid_kwargs}

        contour_kwargs = {**CONTOUR_KWARGS, **contour_kwargs}
        contour_label_kwargs = {**CONTOUR_LABEL_KWARGS, **contour_label_kwargs}

        plt.set_cmap(colormap)

        x_unit_value, x_unit_name = get_unit_value_and_latex_from_unit(x_unit)
        x_unit_label = get_unit_label(x_unit)

        y_unit_value, y_unit_name = get_unit_value_and_latex_from_unit(y_unit)
        y_unit_label = get_unit_label(y_unit)

        z_unit_value, z_unit_name = get_unit_value_and_latex_from_unit(z_unit)
        z_unit_label = get_unit_label(z_unit)

        x_lower_limit, x_upper_limit = set_axis_limits(ax, x_mesh,
                                                       lower_limit = x_lower_limit, upper_limit = x_upper_limit,
                                                       log = x_log_axis,
                                                       pad = 0, log_pad = 1,
                                                       unit = x_unit, direction = 'x')
        y_lower_limit, y_upper_limit = set_axis_limits(ax, y_mesh,
                                                       lower_limit = y_lower_limit, upper_limit = y_upper_limit,
                                                       log = y_log_axis,
                                                       pad = 0, log_pad = 1,
                                                       unit = y_unit, direction = 'y')

        if not isinstance(colormap, RichardsonColormap):
            z_lower_limit, z_upper_limit = get_axis_limits(z_mesh,
                                                           lower_limit = z_lower_limit, upper_limit = z_upper_limit,
                                                           log = z_log_axis,
                                                           pad = z_pad, log_pad = z_log_pad)
            if z_log_axis:
                if z_lower_limit > 0:
                    norm = matplotlib.colors.LogNorm(vmin = z_lower_limit / z_unit_value, vmax = z_upper_limit / z_unit_value)
                else:
                    norm = matplotlib.colors.SymLogNorm(((np.abs(z_lower_limit) + np.abs(z_upper_limit)) / 2) * sym_log_norm_epsilon)
            else:
                norm = matplotlib.colors.Normalize(vmin = z_lower_limit / z_unit_value, vmax = z_upper_limit / z_unit_value)
        else:
            norm = RichardsonNormalization(equator_magnitude = richardson_equator_magnitude)

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
        if z_label is not None:
            z_label = ax.set_title(r'{}'.format(z_label), fontsize = font_size_title)
            z_label.set_y(TITLE_OFFSET)  # move title up a little
        if x_label is not None:
            x_label = ax.set_xlabel(r'{}'.format(x_label) + x_unit_label, fontsize = font_size_axis_labels)
        if y_label is not None:
            y_label = ax.set_ylabel(r'{}'.format(y_label) + y_unit_label, fontsize = font_size_axis_labels)

        if show_colorbar and colormap.name != 'richardson':
            plt.colorbar(mappable = colormesh, ax = ax, pad = 0.1)

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


def xyt_plot(name,
             x_data, t_data, *y_funcs, y_func_kwargs = (),
             line_labels = (), line_kwargs = (),
             figure_manager = None,
             x_unit = None, y_unit = None, t_unit = None,
             t_fmt_string = r'$t = {} \; {}$', t_text_kwargs = None,
             x_log_axis = False, y_log_axis = False,
             x_lower_limit = None, x_upper_limit = None, y_lower_limit = None, y_upper_limit = None,
             vlines = (), vline_kwargs = (), hlines = (), hline_kwargs = (),
             x_extra_ticks = None, y_extra_ticks = None, x_extra_tick_labels = None, y_extra_tick_labels = None,
             title = None, x_label = None, y_label = None,
             font_size_title = 15, font_size_axis_labels = 15, font_size_tick_labels = 10, font_size_legend = 12,
             ticks_on_top = True, ticks_on_right = True, legend_on_right = False,
             grid_kwargs = None, minor_grid_kwargs = None,
             length = 30,
             fig_dpi_scale = 3,
             save_csv = False,
             progress_bar = True,
             **kwargs):
    # set up figure and axis
    if figure_manager is None:
        figure_manager = FigureManager(name, save_on_exit = False, fig_dpi_scale = fig_dpi_scale, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        # plt.close()
        # fig = plt.figure(figsize = (10, 10))
        ax = fig.add_axes([.15, .15, .75, .7])

        if grid_kwargs is None:
            grid_kwargs = {}
        if minor_grid_kwargs is None:
            minor_grid_kwargs = {}

        grid_kwargs = {**GRID_KWARGS, **grid_kwargs}
        minor_grid_kwargs = {**MINOR_GRID_KWARGS, **minor_grid_kwargs}

        # ensure data is in numpy arrays
        x_data = np.array(x_data)
        line_labels = tuple(line_labels)
        line_kwargs = tuple(line_kwargs)

        _y_func_kwargs = []
        for y_func, y_func_kwargs in itertools.zip_longest(y_funcs, y_func_kwargs):
            if y_func_kwargs is not None:
                _y_func_kwargs.append(y_func_kwargs)
            else:
                _y_func_kwargs.append({})

        y_func_kwargs = _y_func_kwargs

        x_unit_value, x_unit_tex = get_unit_value_and_latex_from_unit(x_unit)
        x_unit_label = get_unit_label(x_unit)

        y_unit_value, y_unit_tex = get_unit_value_and_latex_from_unit(y_unit)
        y_unit_label = get_unit_label(y_unit)

        t_unit_value, t_unit_tex = get_unit_value_and_latex_from_unit(t_unit)
        t_unit_label = t_unit_tex

        attach_hv_lines(ax, vlines, vline_kwargs, unit = x_unit, direction = 'v')
        attach_hv_lines(ax, hlines, hline_kwargs, unit = y_unit, direction = 'h')

        x_lower_limit, x_upper_limit = set_axis_limits(ax, x_data,
                                                       lower_limit = x_lower_limit, upper_limit = x_upper_limit,
                                                       log = x_log_axis,
                                                       pad = 0, log_pad = 1,
                                                       unit = x_unit, direction = 'x')
        y_lower_limit, y_upper_limit = set_axis_limits(ax, *(y_func(x_data, t, **y_kwargs) for y_func, y_kwargs in zip(y_funcs, y_func_kwargs) for t in t_data),
                                                       lower_limit = y_lower_limit, upper_limit = y_upper_limit,
                                                       log = y_log_axis,
                                                       pad = 0.05, log_pad = 10,
                                                       unit = y_unit, direction = 'y')

        ax.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

        # make title, axis labels, and legend
        if title is not None:
            title = ax.set_title(r'{}'.format(title), fontsize = font_size_title)
            title.set_y(TITLE_OFFSET)  # move title up a little
        if x_label is not None:
            x_label = ax.set_xlabel(r'{}'.format(x_label) + x_unit_label, fontsize = font_size_axis_labels)
        if y_label is not None:
            y_label = ax.set_ylabel(r'{}'.format(y_label) + y_unit_label, fontsize = font_size_axis_labels)

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
            ax.set_xticks(list(ax.get_xticks()) + list(np.array(x_extra_ticks) / x_unit_value))  # append the extra tick labels, scaled appropriately
            x_tick_labels = list(ax.get_xticklabels())
            x_tick_labels[-len(x_extra_ticks):] = x_extra_tick_labels  # replace the last set of tick labels (the ones we just added) with the custom tick labels
            ax.set_xticklabels(x_tick_labels)

        if y_extra_ticks is not None and y_extra_tick_labels is not None:
            ax.set_yticks(list(ax.get_yticks()) + list(np.array(y_extra_ticks) / y_unit_value))  # append the extra tick labels, scaled appropriately
            y_tick_labels = list(ax.get_yticklabels())
            y_tick_labels[-len(y_extra_ticks):] = y_extra_tick_labels  # replace the last set of tick labels (the ones we just added) with the custom tick labels
            ax.set_yticklabels(y_tick_labels)

        # set limits again to guarantee we don't see ticks oustide the limits
        ax.set_xlim(x_lower_limit, x_upper_limit)
        ax.set_ylim(y_lower_limit, y_upper_limit)

        # set these AFTER adding extra tick labels so that we don't have to slice into the middle of the label lists above
        ax.tick_params(labeltop = ticks_on_top, labelright = ticks_on_right)

        # zip together each set of y data with its plotting options
        lines = []
        for y_func, y_kwargs, lab, kw in itertools.zip_longest(y_funcs, y_func_kwargs, line_labels, line_kwargs):
            if y_kwargs is None:
                y_kwargs = {}
            if kw is None:  # means there are no kwargs for this y data
                kw = {}
            lines.append(plt.plot(x_data / x_unit_value, np.array(y_func(x_data, t_data[0], **y_kwargs)) / y_unit_value, label = lab, **kw, animated = True)[0])

        if len(line_labels) > 0:
            if not legend_on_right:
                legend = ax.legend(loc = 'upper right', fontsize = font_size_legend)
            if legend_on_right:
                legend = ax.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left', borderaxespad = 0., fontsize = font_size_legend, ncol = 1 + (len(line_labels) // 17))

        if t_text_kwargs is None:
            t_text_kwargs = {}

        t_text_kwargs = {**T_TEXT_KWARGS, **t_text_kwargs}

        t_str = t_fmt_string.format(uround(t_data[0], t_unit, digits = 3), t_unit_label)
        t_text = plt.figtext(.7, .05, t_str, **t_text_kwargs, animated = True)

        ax.grid(True, which = 'major', **grid_kwargs)
        if x_log_axis:
            ax.grid(True, which = 'minor', axis = 'x', **minor_grid_kwargs)
        if y_log_axis:
            ax.grid(True, which = 'minor', axis = 'y', **minor_grid_kwargs)

        # do animation

        frames = len(t_data)
        fps = int(frames / length)

        path = f"{os.path.join(kwargs['target_dir'], name)}.mp4"
        utils.ensure_dir_exists(path)

        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(fig.bbox)
        canvas_width, canvas_height = fig.canvas.get_width_height()

        cmd = ("ffmpeg",
               '-y',
               '-r', f'{fps}',  # choose fps
               '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
               '-pix_fmt', 'argb',  # pixel format
               '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
               '-vcodec', 'mpeg4',  # output encoding
               '-q:v', '1',  # maximum quality
               path)

        if progress_bar:
            t_iter = tqdm(t_data)
        else:
            t_iter = t_data

        with utils.SubprocessManager(cmd, **FFMPEG_PROCESS_KWARGS) as ffmpeg:
            for t in t_iter:
                fig.canvas.restore_region(background)

                # update and redraw y lines
                for line, y_func, y_kwargs in zip(lines, y_funcs, y_func_kwargs):
                    line.set_ydata(np.array(y_func(x_data, t, **y_kwargs)) / y_unit_value)
                    fig.draw_artist(line)

                # update and redraw t strings
                t_text.set_text(t_fmt_string.format(uround(t, t_unit, digits = 3), t_unit_label))
                fig.draw_artist(t_text)

                for artist in itertools.chain(ax.xaxis.get_gridlines(), ax.yaxis.get_gridlines()):
                    fig.draw_artist(artist)

                fig.canvas.blit(fig.bbox)

                ffmpeg.stdin.write(fig.canvas.tostring_argb())

                if not progress_bar:
                    logger.debug(f'Wrote frame for t = {uround(t, t_unit, 3)} {t_unit} to ffmpeg')

    if save_csv:
        raise NotImplementedError
        # csv_path = os.path.splitext(path)[0] + '.csv'
        # np.savetxt(csv_path, (x_data, *(y_func(x_data, t, **y_kwargs) for y_func, y_kwargs in zip(y_funcs, y_func_kwargs) for t in t_data)), delimiter = ',')
        #
        # logger.debug('Saved figure data from {} to {}'.format(name, csv_path))

    return fm


def xyzt_plot(name,
              x_mesh, y_mesh, t_data, z_func, z_func_kwargs = None,
              figure_manager = None,
              x_unit = None, y_unit = None, t_unit = None, z_unit = None,
              t_fmt_string = r'$t = {} \; {}$', t_text_kwargs = None,
              x_log_axis = False, y_log_axis = False, z_log_axis = False,
              x_lower_limit = None, x_upper_limit = None, y_lower_limit = None, y_upper_limit = None, z_lower_limit = None, z_upper_limit = None,
              vlines = (), vline_kwargs = (), hlines = (), hline_kwargs = (),
              x_extra_ticks = None, y_extra_ticks = None, x_extra_tick_labels = None, y_extra_tick_labels = None,
              title = None, x_label = None, y_label = None,
              font_size_title = 15, font_size_axis_labels = 15, font_size_tick_labels = 10,
              ticks_on_top = True, ticks_on_right = True,
              grid_kwargs = None, minor_grid_kwargs = None,
              length = 30,
              colormap = plt.get_cmap('viridis'),
              shading = 'gouarud',
              richardson_equator_magnitude = 1,
              sym_log_norm_epsilon = 1e-3,
              show_colorbar = True,
              save_csv = False,
              progress_bar = True,
              **kwargs):
    # set up figure and axis
    if figure_manager is None:
        figure_manager = FigureManager(name, save_on_exit = False, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = fig.add_axes([.15, .15, .75, .7])

        if grid_kwargs is None:
            grid_kwargs = {}
        if minor_grid_kwargs is None:
            minor_grid_kwargs = {}

        grid_color = CMAP_TO_OPPOSITE.get(colormap, 'black')
        grid_kwargs['color'] = grid_color
        minor_grid_kwargs['color'] = grid_color
        grid_kwargs = {**GRID_KWARGS, **grid_kwargs}
        minor_grid_kwargs = {**MINOR_GRID_KWARGS, **minor_grid_kwargs}

        if z_func_kwargs is None:
            z_func_kwargs = {}

        plt.set_cmap(colormap)

        x_unit_value, x_unit_tex = get_unit_value_and_latex_from_unit(x_unit)
        x_unit_label = get_unit_label(x_unit)

        y_unit_value, y_unit_tex = get_unit_value_and_latex_from_unit(y_unit)
        y_unit_label = get_unit_label(y_unit)

        z_unit_value, z_unit_name = get_unit_value_and_latex_from_unit(z_unit)
        z_unit_label = get_unit_label(z_unit)

        t_unit_value, t_unit_tex = get_unit_value_and_latex_from_unit(t_unit)
        t_unit_label = t_unit_tex

        attach_hv_lines(ax, vlines, vline_kwargs, unit = x_unit, direction = 'v')
        attach_hv_lines(ax, hlines, hline_kwargs, unit = y_unit, direction = 'h')

        x_lower_limit, x_upper_limit = set_axis_limits(ax, x_mesh,
                                                       lower_limit = x_lower_limit, upper_limit = x_upper_limit,
                                                       log = x_log_axis,
                                                       pad = 0, log_pad = 1,
                                                       unit = x_unit, direction = 'x')
        y_lower_limit, y_upper_limit = set_axis_limits(ax, y_mesh,
                                                       lower_limit = y_lower_limit, upper_limit = y_upper_limit,
                                                       log = y_log_axis,
                                                       pad = 0, log_pad = 1,
                                                       unit = y_unit, direction = 'y')

        if not isinstance(colormap, RichardsonColormap):
            z_lower_limit, z_upper_limit = get_axis_limits(*(z_func(x_mesh, y_mesh, t, **z_func_kwargs) for t in t_data),
                                                           lower_limit = z_lower_limit, upper_limit = z_upper_limit,
                                                           log = z_log_axis,
                                                           pad = 0, log_pad = 10)
            if z_log_axis:
                if z_lower_limit > 0:
                    norm = matplotlib.colors.LogNorm(vmin = z_lower_limit / z_unit_value, vmax = z_upper_limit / z_unit_value)
                else:
                    norm = matplotlib.colors.SymLogNorm(((np.abs(z_lower_limit) + np.abs(z_upper_limit)) / 2) * sym_log_norm_epsilon)
            else:
                norm = matplotlib.colors.Normalize(vmin = z_lower_limit / z_unit_value, vmax = z_upper_limit / z_unit_value)
        else:
            norm = RichardsonNormalization(equator_magnitude = richardson_equator_magnitude)

        ax.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

        # make title, axis labels, and legend
        if title is not None:
            title = ax.set_title(r'{}'.format(title), fontsize = font_size_title)
            title.set_y(TITLE_OFFSET)  # move title up a little
        if x_label is not None:
            x_label = ax.set_xlabel(r'{}'.format(x_label) + x_unit_label, fontsize = font_size_axis_labels)
        if y_label is not None:
            y_label = ax.set_ylabel(r'{}'.format(y_label) + y_unit_label, fontsize = font_size_axis_labels)

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
            ax.set_xticks(list(ax.get_xticks()) + list(np.array(x_extra_ticks) / x_unit_value))  # append the extra tick labels, scaled appropriately
            x_tick_labels = list(ax.get_xticklabels())
            x_tick_labels[-len(x_extra_ticks):] = x_extra_tick_labels  # replace the last set of tick labels (the ones we just added) with the custom tick labels
            ax.set_xticklabels(x_tick_labels)

        if y_extra_ticks is not None and y_extra_tick_labels is not None:
            ax.set_yticks(list(ax.get_yticks()) + list(np.array(y_extra_ticks) / y_unit_value))  # append the extra tick labels, scaled appropriately
            y_tick_labels = list(ax.get_yticklabels())
            y_tick_labels[-len(y_extra_ticks):] = y_extra_tick_labels  # replace the last set of tick labels (the ones we just added) with the custom tick labels
            ax.set_yticklabels(y_tick_labels)

        # set limits again to guarantee we don't see ticks oustide the limits
        ax.set_xlim(x_lower_limit, x_upper_limit)
        ax.set_ylim(y_lower_limit, y_upper_limit)

        # set these AFTER adding extra tick labels so that we don't have to slice into the middle of the label lists above
        ax.tick_params(labeltop = ticks_on_top, labelright = ticks_on_right)

        ax.grid(True, which = 'major', **grid_kwargs)
        if x_log_axis:
            ax.grid(True, which = 'minor', axis = 'x', **minor_grid_kwargs)
        if y_log_axis:
            ax.grid(True, which = 'minor', axis = 'y', **minor_grid_kwargs)

        colormesh = ax.pcolormesh(
            x_mesh / x_unit_value,
            y_mesh / y_unit_value,
            z_func(x_mesh, y_mesh, t_data[0], **z_func_kwargs) / z_unit_value,
            shading = shading,
            norm = norm,
            animated = True,
        )

        if show_colorbar and colormap.name != 'richardson':
            plt.colorbar(mappable = colormesh, ax = ax, pad = 0.1)

        if t_text_kwargs is None:
            t_text_kwargs = {}

        t_text_kwargs = {**T_TEXT_KWARGS, **t_text_kwargs}

        t_str = t_fmt_string.format(uround(t_data[0], t_unit, digits = 3), t_unit_label)
        t_text = plt.figtext(.7, .05, t_str, **t_text_kwargs, animated = True)

        # do animation

        frames = len(t_data)
        fps = int(frames / length)

        path = f"{os.path.join(kwargs['target_dir'], name)}.mp4"
        utils.ensure_dir_exists(path)

        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(fig.bbox)
        canvas_width, canvas_height = fig.canvas.get_width_height()

        cmd = ("ffmpeg",
               '-y',
               '-r', f'{fps}',  # choose fps
               '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
               '-pix_fmt', 'argb',  # pixel format
               '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
               '-vcodec', 'mpeg4',  # output encoding
               '-q:v', '1',  # maximum quality
               path)

        if progress_bar:
            t_iter = tqdm(t_data)
        else:
            t_iter = t_data

        with utils.SubprocessManager(cmd, **FFMPEG_PROCESS_KWARGS) as ffmpeg:
            for t in t_iter:
                fig.canvas.restore_region(background)

                z = z_func(x_mesh, y_mesh, t, **z_func_kwargs)

                if shading == 'flat':
                    z = z[:-1, :-1]

                colormesh.set_array(z.ravel())
                fig.draw_artist(colormesh)

                # update and redraw t strings
                t_text.set_text(t_fmt_string.format(uround(t, t_unit, digits = 3), t_unit_label))
                fig.draw_artist(t_text)

                for artist in itertools.chain(ax.xaxis.get_gridlines(), ax.yaxis.get_gridlines()):
                    fig.draw_artist(artist)

                fig.canvas.blit(fig.bbox)

                ffmpeg.stdin.write(fig.canvas.tostring_argb())

                if not progress_bar:
                    logger.debug(f'Wrote frame for t = {uround(t, t_unit, 3)} {t_unit} to ffmpeg')

    if save_csv:
        raise NotImplementedError
        # csv_path = os.path.splitext(path)[0] + '.csv'
        # np.savetxt(csv_path, (x_data, *y_data), delimiter = ',')
        #
        # logger.debug('Saved figure data from {} to {}'.format(name, csv_path))

    return fm


def animate(figure_manager, update_function, update_function_arguments,
            artists = (),
            length = 30,
            progress_bar = True):
    fig = figure_manager.fig

    path = os.path.join(figure_manager.target_dir, figure_manager.name + figure_manager.name_postfix + '.mp4')

    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    canvas_width, canvas_height = fig.canvas.get_width_height()

    fps = int(len(update_function_arguments) / length)

    cmd = ("ffmpeg",
           '-y',
           '-r', f'{fps}',  # choose fps
           '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
           '-pix_fmt', 'argb',  # pixel format
           '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
           '-vcodec', 'mpeg4',  # output encoding
           '-q:v', '1',  # maximum quality
           path)

    if progress_bar:
        update_function_arguments = tqdm(update_function_arguments)

    with utils.SubprocessManager(cmd, **FFMPEG_PROCESS_KWARGS) as ffmpeg:
        for arg in update_function_arguments:
            fig.canvas.restore_region(background)

            update_function(arg)

            for artist in artists:
                fig.draw_artist(artist)

            fig.canvas.blit(fig.bbox)

            ffmpeg.stdin.write(fig.canvas.tostring_argb())

            if not progress_bar:
                logger.debug(f'Wrote frame for t = {uround(t, t_unit, 3)} {t_unit} to ffmpeg')


class AxisManager:
    """
    A superclass that manages a matplotlib axis for an Animator.
    """

    def __init__(self):
        self.redraw = []

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    def initialize(self, simulation):
        """Hook method for initializing the AxisManager."""
        self.sim = simulation
        self.spec = simulation.spec

        self.initialize_axis()

        logger.debug(f'Initialized {self}')

    def assign_axis(self, axis):
        self.axis = axis

        logger.debug(f'Assigned {self} to {axis}')

    def initialize_axis(self):
        logger.debug(f'Initialized {self}')

    def update_axis(self):
        """Hook method for updating the AxisManager's internal state."""
        logger.debug(f'Updated axis for {self}')

    def info(self):
        info = core.Info(header = self.__class__.__name__)

        return info


class Animator:
    """
    A superclass that handles sending frames to ffmpeg to create animations.

    To actually make an animation there are three hook methods that need to be overwritten: _initialize_figure, _update_data, and _redraw_frame.

    An Animator will generally contain a single matplotlib figure with some animation code of its own in addition to a list of :class:`AxisManagers ~<AxisManager>` that handle axes on the figure.

    For this class to function correctly :code:`ffmpeg` must be visible on the system path.
    """

    def __init__(self, postfix = '', target_dir = None,
                 length = 60, fps = 30,
                 colormap = plt.cm.get_cmap('inferno')):
        """
        Parameters
        ----------
        postfix : :class:`str`
            Postfix for the file name of the resulting animation.
        target_dir : :class:`str`
            Directory to place the animation (and work in).
        length : :class:`float`
            The length of the animation.
        fps : :class:`float`
            The FPS of the animation.
        colormap
            The colormap to use for the animation.
        """
        if target_dir is None:
            target_dir = os.getcwd()
        self.target_dir = target_dir

        postfix = utils.strip_illegal_characters(postfix)
        self.postfix = postfix

        self.length = int(length)
        self.fps = fps
        self.colormap = colormap

        self.axis_managers = []
        self.redraw = []

        self.sim = None
        self.spec = None
        self.fig = None

    def __str__(self):
        return '{}(postfix = "{}")'.format(self.__class__.__name__, self.postfix)

    def __repr__(self):
        return '{}(postfix = {})'.format(self.__class__.__name__, self.postfix)

    def initialize(self, simulation):
        """
        Initialize the Animation by setting the Simulation and Specification, determining the target path for output, determining fps and decimation, and setting up the ffmpeg subprocess.

        _initialize_figure() is called during the execution of this method. It should assign a matplotlib figure object to self.fig.

        The simulation should have an attribute available_animation_frames that returns an int describing how many raw frames might be available for use by the animation.

        :param simulation: a Simulation for the AxisManager to collect data from
        """
        self.sim = simulation
        self.spec = simulation.spec

        self.file_name = '{}_{}.mp4'.format(self.sim.file_name, self.postfix)
        self.file_path = os.path.join(self.target_dir, self.file_name)
        utils.ensure_dir_exists(self.file_path)
        try:
            os.remove(self.file_path)  # ffmpeg complains if you try to overwrite an existing file, so remove it first
        except FileNotFoundError:
            pass

        ideal_frame_count = self.length * self.fps
        self.decimation = int(self.sim.available_animation_frames / ideal_frame_count)  # determine ideal decimation from number of available frames in the simulation
        if self.decimation < 1:
            self.decimation = 1  # if there aren't enough frames available
        self.fps = (self.sim.available_animation_frames / self.decimation) / self.length

        self._initialize_figure()  # call figure initialization hook

        # AXES MUST BE ASSIGNED DURING FIGURE INITIALIZATION
        for ax in self.axis_managers:
            logger.debug(f'Initializing axis {ax} for {self}')
            ax.initialize(simulation)

        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        canvas_width, canvas_height = self.fig.canvas.get_width_height()
        self.cmd = ("ffmpeg",
                    '-y',
                    '-r', '{}'.format(self.fps),  # choose fps
                    '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
                    '-pix_fmt', 'argb',  # pixel format
                    '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
                    '-vcodec', 'mpeg4',  # output encoding
                    '-q:v', '1',  # maximum quality
                    self.file_path)

        self.ffmpeg = subprocess.Popen(self.cmd, **FFMPEG_PROCESS_KWARGS)

        logger.info('Initialized {}'.format(self))

    def cleanup(self):
        """
        Cleanup method for the Animator's ffmpeg subprocess.

        Should always be called via a try...finally clause (namely, in the finally) in Simulation.run_simulation.
        """
        self.ffmpeg.communicate()
        logger.info('Cleaned up {}'.format(self))

    def _initialize_figure(self):
        """
        Hook for a method to initialize the Animator's figure.

        Make sure that any plot element that will be mutated during the animation is created using the animation = True keyword argument and has a reference in self.redraw.
        """
        logger.debug('Initialized figure for {}'.format(self))

    def _update_data(self):
        """Hook for a method to update the data for each animated figure element."""
        logger.debug('{} updating data from {} {}'.format(self, self.sim.__class__.__name__, self.sim.name))

        for ax in self.axis_managers:
            ax.update_axis()

        logger.debug('{} updated data from {} {}'.format(self, self.sim.__class__.__name__, self.sim.name))

    def _redraw_frame(self):
        """Redraw the figure frame."""
        logger.debug('Redrawing frame for {}'.format(self))

        plt.set_cmap(self.colormap)  # make sure the colormap is correct, in case other figures have been created somewhere

        self.fig.canvas.restore_region(self.background)  # copy the static background back onto the figure

        self._update_data()  # get data from the Simulation and update any plot elements that need to be redrawn

        # draw everything that needs to be redrawn (any plot elements that will be mutated during the animation should be added to self.redraw)
        for rd in itertools.chain(self.redraw, *(ax.redraw for ax in self.axis_managers)):
            self.fig.draw_artist(rd)

        self.fig.canvas.blit(self.fig.bbox)  # blit the canvas, finalizing all of the draw_artists

        logger.debug('Redrew frame for {}'.format(self))

    def send_frame_to_ffmpeg(self):
        """Redraw anything that needs to be redrawn, then write the figure to an RGB string and send it to ffmpeg."""
        logger.debug('{} sending frame to ffpmeg from {} {}'.format(self, self.sim.__class__.__name__, self.sim.name))

        self._redraw_frame()

        self.ffmpeg.stdin.write(self.fig.canvas.tostring_argb())

        logger.debug('{} sent frame to ffpmeg from {} {}'.format(self, self.sim.__class__.__name__, self.sim.name))

    def info(self):
        info = core.Info(header = f'{self.__class__.__name__}: {self.postfix}')

        info.add_field('Length', f'{self.length} s')
        info.add_field('FPS', f'{self.fps}')

        for axis_manager in self.axis_managers:
            info.add_info(axis_manager.info())

        return info


class RichardsonColormap(matplotlib.colors.Colormap):
    """
    A matplotlib Colormap subclass which implements the colormap described in J. L. Richardson, Comput. Phys. Commun. 63, 84 (1991).

    This colormap is appropriate for visualizing complex-valued functions in two dimensions.
    """

    def __init__(self):
        self.name = 'richardson'
        self.N = 256

    def __call__(self, x, alpha = 1, bytes = False):
        real, imag = np.real(x), np.imag(x)

        mag = np.sqrt((real ** 2) + (imag ** 2))
        z = (mag ** 2) - 1
        zplus = z + 2
        eta = np.where(np.greater_equal(z, 0), 1, -1)

        common = .5 + (eta * (.5 - (mag / zplus)))
        real_term = real / (np.sqrt(6) * zplus)
        imag_term = imag / (np.sqrt(2) * zplus)

        rgba = np.ones(np.shape(x) + (4,))  # create rgba array of shape shape as x, except in last dimension, where rgba values will be stored
        rgba[:, 0] = common + (2 * real_term)  # red
        rgba[:, 1] = common - real_term + imag_term  # green
        rgba[:, 2] = common - real_term - imag_term  # blue

        return rgba


matplotlib.cm.register_cmap(name = 'richardson', cmap = RichardsonColormap())  # register cmap so that plt.get_cmap('richardson') can find it
CMAP_TO_OPPOSITE[plt.get_cmap('richardson')] = WHITE
CMAP_TO_OPPOSITE['richardson'] = WHITE


class RichardsonNormalization(matplotlib.colors.Normalize):
    """A matplotlib Normalize subclass which implements an appropriate normalization for :class:`RichardsonColormap`."""

    def __init__(self, equator_magnitude = 1):
        self.equator_magnitude = np.abs(equator_magnitude)

    def __call__(self, x, **kwargs):
        return ma.masked_invalid(x / self.equator_magnitude, copy = False)

    def autoscale(self, *args):
        pass

    def autoscale_None(self, *args):
        pass


class AbsoluteRenormalize(matplotlib.colors.Normalize):
    def __init__(self):
        self.vmin = 0
        self.vmax = 1

    def __call__(self, x, **kwargs):
        return ma.masked_invalid(np.abs(x) / np.nanmax(np.abs(x)), copy = False)

    def autoscale(self, *args):
        pass

    def autoscale_None(self, *args):
        pass
