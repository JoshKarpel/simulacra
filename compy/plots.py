import itertools as it
import os
import logging
import fractions
import subprocess

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm

from . import core, utils
from .units import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# default kwargs for various purposes
GRID_KWARGS = {
    'linestyle': '-',
    'color': 'black',
    'linewidth': .25,
    'alpha': 0.4
}


def _get_fig_dims(fig_scale, fig_width_pts = 498.66258, aspect_ratio = (np.sqrt(5.0) - 1.0) / 2.0):
    """
    Return the dimensions (width, height) for a figure based on the scale, width (in points), and aspect ratio.

    Helper function for get_figure.

    :param fig_scale: the scale of the figure
    :type fig_scale: float
    :param fig_width_pts: get this from LaTeX using \the\textwidth
    :type fig_width_pts: float
    :param aspect_ratio: height = width * ratio, defaults to golden ratio
    :type aspect_ratio: float
    :return: (fig_width, fig_height)
    """
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch

    fig_width = fig_width_pts * inches_per_pt * fig_scale  # width in inches
    fig_height = fig_width * aspect_ratio  # height in inches

    return fig_width, fig_height


def get_figure(fig_scale = 0.95, fig_dpi_scale = 1, fig_width_pts = 498.66258, aspect_ratio = (np.sqrt(5.0) - 1.0) / 2.0):
    """
    Get a matplotlib figure object with the desired scale relative to a full-text-width LaTeX page.

    Special scales:
    scale = 'full' -> scale = 0.95
    scale = 'half' -> scale = 0.475

    :param fig_scale: the scale of the figure
    :type fig_scale: float
    :param fig_width_pts: get this from LaTeX using \the\textwidth
    :type fig_width_pts: float
    :param aspect_ratio: height = width * ratio, defaults to golden ratio
    :type aspect_ratio: float
    :return: a matplotlib figure with the desired dimensions
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
                        dpi_scale = 1,
                        transparent = True,
                        colormap = plt.cm.get_cmap('inferno'),
                        **kwargs):
    """
    Save the current matplotlib figure to a file with the given name to the given folder.
    
    :param name: the name of the file
    :type name: str
    :param name_postfix: a postfix for the filename, added after :code:`name`
    :type name: str
    :param target_dir: the directory to save the file to
    :type target_dir: str
    :param img_format: the format the save the image in
    :type img_format: str
    :param dpi_scale: the scale to save the image at
    :type dpi_scale: float
    :param transparent: whether to make the background of the image transparent (if the format supports it)
    :type transparent: bool
    :param colormap: a colormap to switch to before saving the image
    :param kwargs: absorbs kwargs silently
    :return: the path the image was saved to
    """
    plt.set_cmap(colormap)

    if target_dir is None:
        target_dir = os.getcwd()
    path = os.path.join(target_dir, '{}{}.{}'.format(name, name_postfix, img_format))

    utils.ensure_dir_exists(path)

    plt.savefig(path, dpi = dpi_scale * plt.gcf().dpi, bbox_inches = 'tight', transparent = transparent)

    logger.debug('Saved matplotlib figure {} to {}'.format(name, path))

    return path


def get_pi_ticks_and_labels(lower_limit = 0, upper_limit = twopi, denom = 4):
    """
    
    :param lower_limit: 
    :param upper_limit: 
    :param num: 
    :return: ticks, labels
    """

    low = int(np.floor(lower_limit / pi))
    high = int(np.ceil(upper_limit / pi))

    ticks = list(fractions.Fraction(n, denom) for n in range((high * denom) + 1))
    labels = []
    for tick in ticks:
        if tick.numerator == 0:
            labels.append(r'$0$')
        elif tick.numerator == tick.denominator == 1:
            labels.append(r'$\pi$')
        elif tick.denominator == 1:
            labels.append(fr'$ {tick.numerator} \pi $')
        else:
            labels.append(fr'$ \frac{{ {tick.numerator} }}{{ {tick.denominator} }} \pi $')

    return list(float(tick) * pi for tick in ticks), list(labels)


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


def set_axis_ticks_and_labels(axis, ticks, labels, direction = 'x'):
    if direction == 'x':
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels)
    elif direction == 'y':
        axis.set_yticks(ticks)
        axis.set_yticklabels(labels)
    elif direction == 'z':
        axis.set_zticks(ticks)
        axis.set_zticklabels(labels)
    else:
        raise ValueError(f"Invalid direction specifier to set_axis_ticks_and_labels. Got {direction}, should have gotten one of 'x', 'y', or 'z'")


def xy_plot(name,
            x_data, *y_data,
            line_labels = (), line_kwargs = (),
            figure_manager = None,
            x_unit = 1, y_unit = 1,
            x_log_axis = False, y_log_axis = False,
            x_lower_limit = None, x_upper_limit = None, y_lower_limit = None, y_upper_limit = None,
            vlines = (), vline_kwargs = (), hlines = (), hline_kwargs = (),
            x_extra_ticks = None, y_extra_ticks = None, x_extra_tick_labels = None, y_extra_tick_labels = None,
            title = None, x_label = None, y_label = None,
            font_size_title = 15, font_size_axis_labels = 15, font_size_tick_labels = 10, font_size_legend = 12,
            ticks_on_top = True, ticks_on_right = True, legend_on_right = False,
            grid_kwargs = None,
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

        # ensure data is in numpy arrays
        x_data = np.array(x_data)
        y_data = [np.array(y) for y in y_data]
        line_labels = tuple(line_labels)
        line_kwargs = tuple(line_kwargs)

        x_unit_value, x_unit_name = unit_value_and_name_from_unit(x_unit)
        if x_unit_name != '':
            x_unit_label = r' (${}$)'.format(x_unit_name)
        else:
            x_unit_label = ''

        y_unit_value, y_unit_name = unit_value_and_name_from_unit(y_unit)
        if y_unit_name != '':
            y_unit_label = r' (${}$)'.format(y_unit_name)
        else:
            y_unit_label = ''

        # zip together each set of y data with its plotting options
        lines = []
        for y, lab, kw in it.zip_longest(y_data, line_labels, line_kwargs):
            if kw is None:  # means there are no kwargs for this y data
                kw = {}
            lines.append(plt.plot(x_data / x_unit_value, y / y_unit_value, label = lab, **kw)[0])

        # make any horizontal and vertical lines
        for vl, vkw in it.zip_longest(vlines, vline_kwargs):
            if vkw is None:
                vkw = {}
            kw = {'color': 'black', 'linestyle': '-'}
            kw.update(vkw)
            ax.axvline(x = vl / x_unit_value, **kw)
        for hl, hkw in it.zip_longest(hlines, hline_kwargs):
            if hkw is None:
                hkw = {}
            kw = {'color': 'black', 'linestyle': '-'}
            kw.update(hkw)
            ax.axhline(y = hl / y_unit_value, **kw)

        if grid_kwargs is not None:
            grid_kwargs = GRID_KWARGS.update(grid_kwargs)
        else:
            grid_kwargs = GRID_KWARGS

        if x_log_axis:
            ax.set_xscale('log')
        if y_log_axis:
            ax.set_yscale('log')
            minor_grid_kwargs = grid_kwargs.copy()
            minor_grid_kwargs['alpha'] -= .1
            ax.grid(True, which = 'minor', **minor_grid_kwargs)

        if x_lower_limit is None:
            x_lower_limit = np.nanmin(x_data)
        if x_upper_limit is None:
            x_upper_limit = np.nanmax(x_data)

        # TODO: THIS CODE IS TOTALLY FUKT

        if y_lower_limit is None and y_upper_limit is None:
            y_lower_limit = min([np.nanmin(y) for y in y_data])
            y_upper_limit = max([np.nanmax(y) for y in y_data])
            y_range = np.abs(y_upper_limit - y_lower_limit)
            y_lower_limit -= .05 * y_range
            y_upper_limit += .05 * y_range
            if y_log_axis:
                y_lower_limit /= 10
                y_upper_limit *= 10

        # neither of these trigger if both y limits are None
        if y_lower_limit is None:
            y_lower_limit = min([np.nanmin(y) for y in y_data])
            if y_log_axis:
                y_lower_limit /= 10
        if y_upper_limit is None:
            y_upper_limit = max([np.nanmax(y) for y in y_data])
            if y_log_axis:
                y_upper_limit *= 10

        ax.set_xlim(left = x_lower_limit / x_unit_value, right = x_upper_limit / x_unit_value)
        ax.set_ylim(bottom = y_lower_limit / y_unit_value, top = y_upper_limit / y_unit_value)

        ax.grid(True, which = 'major', **grid_kwargs)

        ax.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

        # make title, axis labels, and legend
        if title is not None:
            title = ax.set_title(r'{}'.format(title), fontsize = font_size_title)
            title.set_y(1.06)  # move title up a little
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

        if x_unit_name == 'rad':
            ticks, labels = get_pi_ticks_and_labels(x_lower_limit, x_upper_limit)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
        if y_unit_name == 'rad':
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
        ax.set_xlim(left = x_lower_limit / x_unit_value, right = x_upper_limit / x_unit_value)
        ax.set_ylim(bottom = y_lower_limit / y_unit_value, top = y_upper_limit / y_unit_value)

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
             x_unit = 1, y_unit = 1, z_unit = 1,
             x_log_axis = False, y_log_axis = False, z_log_axis = False,
             x_lower_limit = None, x_upper_limit = None, y_lower_limit = None, y_upper_limit = None, z_lower_limit = None, z_upper_limit = None,
             x_extra_ticks = None, y_extra_ticks = None, x_extra_tick_labels = None, y_extra_tick_labels = None,
             z_label = None, x_label = None, y_label = None,
             font_size_title = 15, font_size_axis_labels = 15, font_size_tick_labels = 10,
             ticks_on_top = True, ticks_on_right = True,
             grid_kwargs = None,
             save_csv = False,
             **kwargs):
    # set up figure and axis
    if figure_manager is None:
        figure_manager = FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = plt.subplot(111)

        x_unit_value, x_unit_name = unit_value_and_name_from_unit(x_unit)
        if x_unit_name != '':
            x_unit_label = r' (${}$)'.format(x_unit_name)
        else:
            x_unit_label = ''

        y_unit_value, y_unit_name = unit_value_and_name_from_unit(y_unit)
        if y_unit_name != '':
            y_unit_label = r' (${}$)'.format(y_unit_name)
        else:
            y_unit_label = ''

        z_unit_value, z_unit_name = unit_value_and_name_from_unit(z_unit)
        if z_unit_name != '':
            z_unit_label = r' (${}$)'.format(z_unit_name)
        else:
            z_unit_label = ''

        if z_lower_limit is None:
            z_lower_limit = np.nanmin(z_mesh)
            if z_log_axis:
                z_lower_limit /= 10
        if z_upper_limit is None:
            z_upper_limit = np.nanmax(z_mesh)
            if z_log_axis:
                z_upper_limit *= 10

        if z_log_axis:
            norm = matplotlib.colors.LogNorm(vmin = z_lower_limit / z_unit_value, vmax = z_upper_limit / z_unit_value)
        else:
            norm = matplotlib.colors.Normalize(vmin = z_lower_limit / z_unit_value, vmax = z_upper_limit / z_unit_value)

        colormesh = ax.pcolormesh(x_mesh / x_unit_value,
                                  y_mesh / y_unit_value,
                                  z_mesh / z_unit_value,
                                  shading = 'gouraud',
                                  norm = norm)

        if grid_kwargs is not None:
            grid_kwargs = GRID_KWARGS.update(grid_kwargs)
        else:
            grid_kwargs = GRID_KWARGS

        if x_log_axis:
            ax.set_xscale('log')
            minor_grid_kwargs = grid_kwargs.copy()
            minor_grid_kwargs['alpha'] -= .1
            ax.grid(True, which = 'minor', **minor_grid_kwargs)
        if y_log_axis:
            ax.set_yscale('log')
            minor_grid_kwargs = grid_kwargs.copy()
            minor_grid_kwargs['alpha'] -= .1
            ax.grid(True, which = 'minor', **minor_grid_kwargs)

        # set axis limits
        if x_lower_limit is None:
            x_lower_limit = np.nanmin(x_mesh)
        if x_upper_limit is None:
            x_upper_limit = np.nanmax(x_mesh)

        if y_lower_limit is None:
            y_lower_limit = np.nanmin(y_mesh)
        if y_upper_limit is None:
            y_upper_limit = np.nanmax(y_mesh)

        ax.set_xlim(left = x_lower_limit / x_unit_value, right = x_upper_limit / x_unit_value)
        ax.set_ylim(bottom = y_lower_limit / y_unit_value, top = y_upper_limit / y_unit_value)

        ax.grid(True, which = 'major', **grid_kwargs)

        ax.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

        # make title, axis labels, and legend
        if z_label is not None:
            z_label = ax.set_title(r'{}'.format(z_label), fontsize = font_size_title)
            z_label.set_y(1.075)  # move title up a little
        if x_label is not None:
            x_label = ax.set_xlabel(r'{}'.format(x_label) + x_unit_label, fontsize = font_size_axis_labels)
        if y_label is not None:
            y_label = ax.set_ylabel(r'{}'.format(y_label) + y_unit_label, fontsize = font_size_axis_labels)

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
             x_unit = 1, y_unit = 1, t_unit = 1,
             t_fmt_string = r'$t = {}$', t_text_kwargs = None,
             x_log_axis = False, y_log_axis = False,
             x_lower_limit = None, x_upper_limit = None, y_lower_limit = None, y_upper_limit = None,
             vlines = (), vline_kwargs = (), hlines = (), hline_kwargs = (),
             x_extra_ticks = None, y_extra_ticks = None, x_extra_tick_labels = None, y_extra_tick_labels = None,
             title = None, x_label = None, y_label = None,
             font_size_title = 15, font_size_axis_labels = 15, font_size_tick_labels = 10, font_size_legend = 12,
             ticks_on_top = True, ticks_on_right = True, legend_on_right = False,
             grid_kwargs = None,
             length = 30,
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
        figure_manager = FigureManager(name, save_on_exit = False, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = fig.add_axes([.15, .15, .75, .7])

        # ensure data is in numpy arrays
        x_data = np.array(x_data)
        line_labels = tuple(line_labels)
        line_kwargs = tuple(line_kwargs)

        _y_func_kwargs = []
        for y_func, y_func_kwargs in it.zip_longest(y_funcs, y_func_kwargs):
            if y_func_kwargs is not None:
                _y_func_kwargs.append(y_func_kwargs)
            else:
                _y_func_kwargs.append({})

        y_func_kwargs = _y_func_kwargs

        x_unit_value, x_unit_name = unit_value_and_name_from_unit(x_unit)
        if x_unit_name != '':
            x_unit_label = r' (${}$)'.format(x_unit_name)
        else:
            x_unit_label = ''

        y_unit_value, y_unit_name = unit_value_and_name_from_unit(y_unit)
        if y_unit_name != '':
            y_unit_label = r' (${}$)'.format(y_unit_name)
        else:
            y_unit_label = ''

        t_unit_value, t_unit_name = unit_value_and_name_from_unit(t_unit)

        # make any horizontal and vertical lines
        for vl, vkw in it.zip_longest(vlines, vline_kwargs):
            if vkw is None:
                vkw = {}
            kw = {'color': 'black', 'linestyle': '-'}
            kw.update(vkw)
            ax.axvline(x = vl / x_unit_value, **kw)
        for hl, hkw in it.zip_longest(hlines, hline_kwargs):
            if hkw is None:
                hkw = {}
            kw = {'color': 'black', 'linestyle': '-'}
            kw.update(hkw)
            ax.axhline(y = hl / y_unit_value, **kw)

        if grid_kwargs is not None:
            grid_kwargs = GRID_KWARGS.update(grid_kwargs)
        else:
            grid_kwargs = GRID_KWARGS

        if x_log_axis:
            ax.set_xscale('log')
        if y_log_axis:
            ax.set_yscale('log')
            minor_grid_kwargs = grid_kwargs.copy()
            minor_grid_kwargs['alpha'] -= .1
            ax.grid(True, which = 'minor', **minor_grid_kwargs)

        # set axis limits
        if x_lower_limit is None:
            x_lower_limit = np.nanmin(x_data)
        if x_upper_limit is None:
            x_upper_limit = np.nanmax(x_data)

        if y_lower_limit is None and y_upper_limit is None:
            y_lower_limit = min(np.nanmin(y_func(x_data, t, **y_kwargs)) for y_func, y_kwargs in zip(y_funcs, y_func_kwargs) for t in t_data)
            y_upper_limit = max(np.nanmax(y_func(x_data, t, **y_kwargs)) for y_func, y_kwargs in zip(y_funcs, y_func_kwargs) for t in t_data)
            y_range = np.abs(y_upper_limit - y_lower_limit)
            y_lower_limit -= .05 * y_range
            y_upper_limit += .05 * y_range

        if y_log_axis:
            y_lower_limit /= 10
            y_upper_limit *= 10

        # neither of these trigger if both y limits are None
        if y_lower_limit is None:
            y_lower_limit = min(np.nanmin(y_func(x_data, t, **y_kwargs)) for y_func, y_kwargs in zip(y_funcs, y_func_kwargs) for t in t_data)
            if y_log_axis:
                y_lower_limit /= 10
        if y_upper_limit is None:
            y_upper_limit = max(np.nanmax(y_func(x_data, t, **y_kwargs)) for y_func, y_kwargs in zip(y_funcs, y_func_kwargs) for t in t_data)
            if y_log_axis:
                y_upper_limit *= 10

        ax.set_xlim(left = x_lower_limit / x_unit_value, right = x_upper_limit / x_unit_value)
        ax.set_ylim(bottom = y_lower_limit / y_unit_value, top = y_upper_limit / y_unit_value)

        ax.grid(True, which = 'major', **grid_kwargs)

        ax.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

        # make title, axis labels, and legend
        if title is not None:
            title = ax.set_title(r'{}'.format(title), fontsize = font_size_title)
            title.set_y(1.075)  # move title up a little
        if x_label is not None:
            x_label = ax.set_xlabel(r'{}'.format(x_label) + x_unit_label, fontsize = font_size_axis_labels)
        if y_label is not None:
            y_label = ax.set_ylabel(r'{}'.format(y_label) + y_unit_label, fontsize = font_size_axis_labels)

        fig.canvas.draw()  # draw that figure so that the ticks exist, so that we can add more ticks

        if x_unit_name == 'rad':
            ticks, labels = get_pi_ticks_and_labels(x_lower_limit, x_upper_limit)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
        if y_unit_name == 'rad':
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
        ax.set_xlim(left = x_lower_limit / x_unit_value, right = x_upper_limit / x_unit_value)
        ax.set_ylim(bottom = y_lower_limit / y_unit_value, top = y_upper_limit / y_unit_value)

        # set these AFTER adding extra tick labels so that we don't have to slice into the middle of the label lists above
        ax.tick_params(labeltop = ticks_on_top, labelright = ticks_on_right)

        # zip together each set of y data with its plotting options
        lines = []
        for y_func, y_kwargs, lab, kw in it.zip_longest(y_funcs, y_func_kwargs, line_labels, line_kwargs):
            if y_kwargs is None:
                y_kwargs = {}
            if kw is None:  # means there are no kwargs for this y data
                kw = {}
            lines.append(plt.plot(x_data / x_unit_value, y_func(x_data, t_data[0], **y_kwargs) / y_unit_value, label = lab, **kw, animated = True)[0])

        if len(line_labels) > 0:
            if not legend_on_right:
                legend = ax.legend(loc = 'upper right', fontsize = font_size_legend)
            if legend_on_right:
                legend = ax.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left', borderaxespad = 0., fontsize = font_size_legend, ncol = 1 + (len(line_labels) // 17))

        t_text_kwarg_defaults = dict(
            fontsize = 12,
        )
        if t_text_kwargs is None:
            t_text_kwargs = {}

        t_text_kwarg_defaults.update(t_text_kwargs)  # TODO: Messy, messy...

        print(t_fmt_string)
        t_fmt_string.format(t_data[0])

        t_str = t_fmt_string.format(uround(t_data[0], t_unit, digits = 3))
        if t_unit_name != '':
            t_str += fr' ${t_unit_name}$'

        t_text = plt.figtext(.7, .05, t_str, **t_text_kwarg_defaults, animated = True)

        # do animation

        frames = len(t_data)
        fps = frames / length

        path = f"{os.path.join(kwargs['target_dir'], name)}.mp4"
        utils.ensure_dir_exists(path)

        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(fig.bbox)
        canvas_width, canvas_height = fig.canvas.get_width_height()

        cmdstring = ("ffmpeg",
                     '-y',
                     '-r', '{}'.format(fps),  # choose fps
                     '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
                     '-pix_fmt', 'argb',  # pixel format
                     '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
                     '-vcodec', 'mpeg4',  # output encoding
                     '-q:v', '1',  # maximum quality
                     path)

        ffmpeg = subprocess.Popen(cmdstring, stdin = subprocess.PIPE, bufsize = -1)

        for t in tqdm(t_data):
            fig.canvas.restore_region(background)

            for line, y_func, y_kwargs in zip(lines, y_funcs, y_func_kwargs):
                line.set_ydata(y_func(x_data, t, **y_kwargs) / y_unit_value)
                fig.draw_artist(line)

            t_str = t_fmt_string.format(uround(t, t_unit, digits = 3))
            if t_unit_name != '':
                t_str += fr' ${t_unit_name}$'
            t_text.set_text(t_str)
            fig.draw_artist(t_text)

            fig.canvas.blit(fig.bbox)

            ffmpeg.stdin.write(fig.canvas.tostring_argb())

    if save_csv:
        raise NotImplementedError
        # csv_path = os.path.splitext(path)[0] + '.csv'
        # np.savetxt(csv_path, (x_data, *y_data), delimiter = ',')
        #
        # logger.debug('Saved figure data from {} to {}'.format(name, csv_path))

    return path
