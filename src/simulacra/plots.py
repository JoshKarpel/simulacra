import itertools
import os
import logging
import fractions
import subprocess

import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import utils
from .units import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

RED = '#d62728'
BLUE = '#1f77b4'
ORANGE = '#ff7f0e'
GREEN = '#2ca02c'

COLOR_OPPOSITE_PLASMA = GREEN
COLOR_OPPOSITE_INFERNO = GREEN
COLOR_OPPOSITE_MAGMA = GREEN
COLOR_OPPOSITE_VIRIDIS = RED

CMAP_TO_OPPOSITE = {
    plt.get_cmap('viridis'): COLOR_OPPOSITE_VIRIDIS,
    plt.get_cmap('plasma'): COLOR_OPPOSITE_PLASMA,
    plt.get_cmap('inferno'): COLOR_OPPOSITE_INFERNO,
    plt.get_cmap('magma'): COLOR_OPPOSITE_MAGMA,
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

TITLE_OFFSET = 1.1

FFMPEG_PROCESS_KWARGS = dict(
        stdin = subprocess.PIPE,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        bufsize = -1,
)


def _get_fig_dims(fig_scale, aspect_ratio = (np.sqrt(5.0) - 1.0) / 2.0, fig_width_pts = 498.66258):
    """
    Return the dimensions (width, height) for a figure based on the scale, width (in points), and aspect ratio.

    Primarily a helper function for get_figure.
    
    Parameters
    ----------
    fig_scale : :class:`float`
        The scale of the figure relative to the figure width.
    aspect_ratio : :class:`float`
        The aspect ratio of the figure (width / height)
    fig_width_pts : :class:`float`
        The "base" width of a figure (e.g. a LaTeX page width).

    Returns
    -------
    tuple of floats
        Figure width and height.
    """
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch

    fig_width = fig_width_pts * inches_per_pt * fig_scale  # width in inches
    fig_height = fig_width * aspect_ratio  # height in inches

    return fig_width, fig_height


def get_figure(fig_scale = 0.95, fig_dpi_scale = 1, aspect_ratio = (np.sqrt(5.0) - 1.0) / 2.0, fig_width_pts = 498.66258):
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
    fig_width_pts : :class:`float`
        The "base" width of a figure (e.g. a LaTeX page width).

    Returns
    -------
    figure
        A matplotlib figure.
    """
    if fig_scale == 'full':
        fig_scale = 0.95
    elif fig_scale == 'half':
        fig_scale = .475

    fig = plt.figure(figsize = _get_fig_dims(fig_scale, fig_width_pts = fig_width_pts, aspect_ratio = aspect_ratio), dpi = fig_dpi_scale * 100)

    return fig


def save_current_figure(name,
                        name_postfix = '',
                        target_dir = None,
                        img_format = 'pdf',
                        transparent = True,
                        colormap = plt.cm.get_cmap('inferno'),
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
    kwargs
        This function absorbs keyword arguments silently.

    Returns
    -------
    :class:`str`
        The path the figure was saved to.
    """
    plt.set_cmap(colormap)

    if target_dir is None:
        target_dir = os.getcwd()
    path = os.path.join(target_dir, '{}{}.{}'.format(name, name_postfix, img_format))

    utils.ensure_dir_exists(path)

    plt.savefig(path, dpi = plt.gcf().dpi, bbox_inches = 'tight', transparent = transparent)

    logger.debug('Saved matplotlib figure {} to {}'.format(name, path))

    return path


class FigureManager:
    """
    A class that manages a matplotlib figure: creating it, showing it, saving it, and cleaning it up.
    """

    def __init__(self, name, name_postfix = '',
                 fig_scale = 0.95, fig_dpi_scale = 1, fig_width_pts = 498.66258, aspect_ratio = (np.sqrt(5.0) - 1.0) / 2.0,
                 target_dir = None, img_format = 'pdf', img_scale = 1,
                 close_before_enter = True, close_after_exit = True,
                 save_on_exit = True, show = False,
                 **kwargs):
        """
        Initialize a :code:`FigureManager`.
        
        Saving occurs before showing.
        
        :param name: the name of the file
        :type name: str
        :param name_postfix: a postfix for the filename, added after :code:`name`
        :type name: str
        :param target_dir: the directory to save the file to
        :type target_dir: str
        :param fig_scale: the scale of the figure in LaTeX pagewidths
        :type fig_scale: float
        :param fig_width_pts: width of a LaTeX pagewidth in points
        :type float
        :param aspect_ratio: the aspect ratio of the image
        :type aspect_ratio: float
        :param img_format: the format the save the image in
        :type img_format: str
        :param img_scale: the scale to save the image at
        :type img_scale: float
        :param close_before_enter: close any existing images before creating the new figure
        :type close_before_enter: bool
        :param close_after_exit: close the figure after save/show
        :type close_after_exit: bool
        :param save_on_exit: if True, save the image using :func:`save_current_figure`
        :type save_on_exit: bool
        :param show: if True, show the image
        :type save: bool
        :param kwargs: kwargs are absorbed
        """
        self.name = name
        self.name_postfix = name_postfix

        self.fig_scale = fig_scale
        self.fig_dpi_scale = fig_dpi_scale
        self.fig_width_pts = fig_width_pts
        self.aspect_ratio = aspect_ratio

        self.target_dir = target_dir
        self.img_format = img_format
        self.img_scale = img_scale

        if len(kwargs) > 0:
            logger.debug('FigureManager for figure {} absorbed extraneous kwargs: {}'.format(self.name, kwargs))

        self.close_before = close_before_enter
        self.close_after = close_after_exit

        self.save_on_exit = save_on_exit
        self.show = show

        self.path = None

    def save(self):
        path = save_current_figure(name = self.name, name_postfix = self.name_postfix, target_dir = self.target_dir, img_format = self.img_format, dpi_scale = self.img_scale)

        self.path = path

    def __enter__(self):
        if self.close_before:
            plt.close()

        self.fig = get_figure(fig_scale = self.fig_scale, fig_dpi_scale = self.fig_dpi_scale, fig_width_pts = self.fig_width_pts, aspect_ratio = self.aspect_ratio)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.save_on_exit:
            self.save()

        if self.show:
            plt.show()

        if self.close_after:
            self.fig.clear()
            plt.close(self.fig)


def get_pi_ticks_and_labels(lower_limit = 0, upper_limit = twopi, denom = 4):
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


def set_axis_ticks_and_labels(axis, ticks, labels, direction = 'x'):
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


def get_axis_limits(*data, lower_limit = None, upper_limit = None, log = False, pad = 0, log_pad = 1):
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
            x_lower_limit = None, x_upper_limit = None, y_lower_limit = None, y_upper_limit = None,
            vlines = (), vline_kwargs = (), hlines = (), hline_kwargs = (),
            x_extra_ticks = None, y_extra_ticks = None, x_extra_tick_labels = None, y_extra_tick_labels = None,
            title = None, x_label = None, y_label = None,
            font_size_title = 15, font_size_axis_labels = 15, font_size_tick_labels = 10, font_size_legend = 12,
            ticks_on_top = True, ticks_on_right = True, legend_on_right = False,
            grid_kwargs = None, minor_grid_kwargs = None,
            save_csv = False,
            **kwargs):
    """
    Generate and save a generic x-y plot.

    :param name: filename for the plot
    :param x_data: a single array to plot the y data against
    :param y_data: any number of arrays of the same length as x_data to plot
    :param line_labels: the labels for the lines
    :param line_kwargs: other keyword arguments for each line's .plot() call (None for default)
    :param x_unit: a number to scale the x_data by. Can be a string corresponding to a key in the unit name/value dict.
    :param y_unit: a number to scale the y_data by. Can be a string corresponding to a key in the unit name/value dict.
    :param x_log_axis: if True, log the x axis
    :param y_log_axis: if True, log the y axis
    :param x_lower_limit: lower limit for the x axis, defaults to np.nanmin(x_data)
    :param x_upper_limit: upper limit for the x axis, defaults to np.nanmax(x_data)
    :param y_lower_limit: lower limit for the y axis, defaults to min(np.nanmin(y_data))
    :param y_upper_limit: upper limit for the y axis, defaults to min(np.nanmin(y_data))
    :param vlines: a list of x positions to place vertical lines
    :param vline_kwargs: a list of kwargs for each vline (None for default)
    :param hlines: a list of y positions to place horizontal lines
    :param hline_kwargs: a list of kwargs for each hline (None for default)
    :param x_extra_ticks:
    :param x_extra_tick_labels:
    :param y_extra_ticks:
    :param y_extra_tick_labels:
    :param title: a title for the plot
    :param x_label: a label for the x axis
    :param y_label: a label for the y axis
    :param font_size_title: font size for the title
    :param font_size_axis_labels: font size for the axis labels
    :param font_size_tick_labels: font size for the tick labels
    :param font_size_legend: font size for the legend
    :param ticks_on_top:
    :param ticks_on_right:
    :param legend_on_right:
    :param grid_kwargs:
    :param save_csv: if True, save x_data and y_data to a CSV file
    :param kwargs: kwargs are passed to a FigureManager context manager
    :return: the path the plot was saved to
    """
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

        attach_hv_lines(ax, vlines, vline_kwargs, unit = x_unit, direction = 'v')
        attach_hv_lines(ax, hlines, hline_kwargs, unit = y_unit, direction = 'h')

        x_lower_limit, x_upper_limit = set_axis_limits(ax, x_data,
                                                       lower_limit = x_lower_limit, upper_limit = x_upper_limit,
                                                       log = x_log_axis,
                                                       pad = 0, log_pad = 1,
                                                       unit = x_unit, direction = 'x')
        y_lower_limit, y_upper_limit = set_axis_limits(ax, y_data,
                                                       lower_limit = y_lower_limit, upper_limit = y_upper_limit,
                                                       log = y_log_axis,
                                                       pad = 0.05, log_pad = 10,
                                                       unit = y_unit, direction = 'y')

        ax.grid(True, which = 'major', **grid_kwargs)
        if x_log_axis:
            ax.grid(True, which = 'minor', axis = 'x', **minor_grid_kwargs)
        if y_log_axis:
            ax.grid(True, which = 'minor', axis = 'y', **minor_grid_kwargs)

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

    return path


def xyz_plot(name,
             x_mesh, y_mesh, z_mesh,
             figure_manager = None,
             x_unit = None, y_unit = None, z_unit = None,
             x_log_axis = False, y_log_axis = False, z_log_axis = False,
             x_lower_limit = None, x_upper_limit = None, y_lower_limit = None, y_upper_limit = None, z_lower_limit = None, z_upper_limit = None,
             x_extra_ticks = None, y_extra_ticks = None, x_extra_tick_labels = None, y_extra_tick_labels = None,
             z_label = None, x_label = None, y_label = None,
             font_size_title = 15, font_size_axis_labels = 15, font_size_tick_labels = 10,
             ticks_on_top = True, ticks_on_right = True,
             grid_kwargs = None, minor_grid_kwargs = None,
             save_csv = False,
             colormap = plt.get_cmap('viridis'), shading = 'gouraud', show_colorbar = True,
             richardson_equator_magnitude = 1,
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

        grid_color = CMAP_TO_OPPOSITE.get(colormap, 'black')
        grid_kwargs['color'] = grid_color
        minor_grid_kwargs['color'] = grid_color
        grid_kwargs = {**GRID_KWARGS, **grid_kwargs}
        minor_grid_kwargs = {**MINOR_GRID_KWARGS, **minor_grid_kwargs}

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
                                                           pad = 0, log_pad = 10)
            if z_log_axis:
                norm = matplotlib.colors.LogNorm(vmin = z_lower_limit / z_unit_value, vmax = z_upper_limit / z_unit_value)
            else:
                norm = matplotlib.colors.Normalize(vmin = z_lower_limit / z_unit_value, vmax = z_upper_limit / z_unit_value)
        else:
            norm = RichardsonNormalization(equator_magnitude = richardson_equator_magnitude)

        colormesh = ax.pcolormesh(x_mesh / x_unit_value,
                                  y_mesh / y_unit_value,
                                  z_mesh / z_unit_value,
                                  shading = shading,
                                  norm = norm)

        ax.grid(True, which = 'major', **grid_kwargs)
        if x_log_axis:
            ax.grid(True, which = 'minor', axis = 'x', **minor_grid_kwargs)
        if y_log_axis:
            ax.grid(True, which = 'minor', axis = 'y', **minor_grid_kwargs)

        ax.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

        # make title, axis labels, and legend
        if z_label is not None:
            z_label = ax.set_title(r'{}'.format(z_label), fontsize = font_size_title)
            z_label.set_y(TITLE_OFFSET)  # move title up a little
        if x_label is not None:
            x_label = ax.set_xlabel(r'{}'.format(x_label) + x_unit_label, fontsize = font_size_axis_labels)
        if y_label is not None:
            y_label = ax.set_ylabel(r'{}'.format(y_label) + y_unit_label, fontsize = font_size_axis_labels)

        if show_colorbar:
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

    return path


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
    """
    
    :param name: filename for the plot
    :param x_data: a single array to ploy the y values against
    :param t_data: a single array whose values will be animated over
    :param y_funcs: functions of the form ``f(x, t, **kwargs)``, where kwargs are from `y_func_kwargs`
    :param y_func_kwargs: keyword arguments for the y_funcs
    :param line_labels: the labels for the lines
    :param line_kwargs: other keyword arguments for each line's .plot() call (None for default)
    :param figure_manager: 
    :param x_unit: 
    :param y_unit: 
    :param t_unit: 
    :param t_fmt_string: 
    :param t_text_kwargs: 
    :param x_log_axis: 
    :param y_log_axis: 
    :param x_lower_limit: 
    :param x_upper_limit: 
    :param y_lower_limit: 
    :param y_upper_limit: 
    :param vlines: 
    :param vline_kwargs: 
    :param hlines: 
    :param hline_kwargs: 
    :param x_extra_ticks: 
    :param y_extra_ticks: 
    :param x_extra_tick_labels: 
    :param y_extra_tick_labels: 
    :param title: 
    :param x_label: 
    :param y_label: 
    :param font_size_title: 
    :param font_size_axis_labels: 
    :param font_size_tick_labels: 
    :param font_size_legend: 
    :param ticks_on_top: 
    :param ticks_on_right: 
    :param legend_on_right: 
    :param grid_kwargs: 
    :param length: 
    :param save_csv: 
    :param kwargs: 
    :return: 
    """
    # set up figure and axis
    if figure_manager is None:
        figure_manager = FigureManager(name, save_on_exit = False, fig_dpi_scale = fig_dpi_scale, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
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

        ax.grid(True, which = 'major', **grid_kwargs)
        if x_log_axis:
            ax.grid(True, which = 'minor', axis = 'x', **minor_grid_kwargs)
        if y_log_axis:
            ax.grid(True, which = 'minor', axis = 'y', **minor_grid_kwargs)

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

        # do animation

        frames = len(t_data)
        fps = int(frames / length)

        path = f"{os.path.join(kwargs['target_dir'], name)}.mp4"
        utils.ensure_dir_exists(path)

        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(fig.bbox)
        canvas_width, canvas_height = fig.canvas.get_width_height()

        cmds = ("ffmpeg",
                '-y',
                '-r', '{}'.format(fps),  # choose fps
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

        with utils.SubprocessManager(cmds, **FFMPEG_PROCESS_KWARGS) as ffmpeg:
            for t in t_iter:
                fig.canvas.restore_region(background)

                # update and redraw y lines
                for line, y_func, y_kwargs in zip(lines, y_funcs, y_func_kwargs):
                    line.set_ydata(np.array(y_func(x_data, t, **y_kwargs)) / y_unit_value)
                    fig.draw_artist(line)

                # update and redraw t strings
                t_text.set_text(t_fmt_string.format(uround(t, t_unit, digits = 3), t_unit_label))
                fig.draw_artist(t_text)

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

    return path


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
              fig_dpi_scale = 3,
              colormap = plt.get_cmap('viridis'), shading = 'gouarud', richardson_equator_magnitude = 1,
              show_colorbar = True,
              save_csv = False,
              progress_bar = True,
              **kwargs):
    # set up figure and axis
    if figure_manager is None:
        figure_manager = FigureManager(name, save_on_exit = False, fig_dpi_scale = fig_dpi_scale, **kwargs)
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
                norm = matplotlib.colors.LogNorm(vmin = z_lower_limit / z_unit_value, vmax = z_upper_limit / z_unit_value)
            else:
                norm = matplotlib.colors.Normalize(vmin = z_lower_limit / z_unit_value, vmax = z_upper_limit / z_unit_value)
        else:
            norm = RichardsonNormalization(equator_magnitude = richardson_equator_magnitude)

        ax.grid(True, which = 'major', **grid_kwargs)
        if x_log_axis:
            ax.grid(True, which = 'minor', axis = 'x', **minor_grid_kwargs)
        if y_log_axis:
            ax.grid(True, which = 'minor', axis = 'y', **minor_grid_kwargs)

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

        colormesh = ax.pcolormesh(x_mesh / x_unit_value,
                                  y_mesh / y_unit_value,
                                  z_func(x_mesh, y_mesh, t_data[0], **z_func_kwargs) / z_unit_value,
                                  shading = shading,
                                  norm = norm)

        if show_colorbar:
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

        cmds = ("ffmpeg",
                '-y',
                '-r', '{}'.format(fps),  # choose fps
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

        with utils.SubprocessManager(cmds, **FFMPEG_PROCESS_KWARGS) as ffmpeg:
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

    return path


class RichardsonColormap(matplotlib.colors.Colormap):
    def __init__(self):
        self.name = 'richardson'
        self.N = 256

    def __call__(self, x, alpha = 1, bytes = False):
        real, imag = np.real(x), np.imag(x)

        mag = np.sqrt((real ** 2) + (imag ** 2))
        z = (mag ** 2) - 1
        zplus = z + 2
        eta = np.where(np.greater_equal(z, 0), 1, -1)

        common = .5 + (eta * (.5 - (mag / zplus)))  # common term to rgb
        real_term = real / (np.sqrt(6) * zplus)
        imag_term = imag / (np.sqrt(2) * zplus)

        rgba = np.ones(np.shape(x) + (4,))  # create rgba array of shape shape as x, except in last dimension, where rgba values will be stored
        rgba[:, 0] = common + (2 * real_term)  # red
        rgba[:, 1] = common - real_term + imag_term  # green
        rgba[:, 2] = common - real_term - imag_term  # blue

        return rgba


matplotlib.cm.register_cmap(name = 'richardson', cmap = RichardsonColormap())  # register cmap so that plt.get_cmap('richardson') can find it


class RichardsonNormalization(matplotlib.colors.Normalize):
    def __init__(self, equator_magnitude = 1):
        self.equator_magnitude = equator_magnitude

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
