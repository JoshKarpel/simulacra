import itertools as it
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from . import core, utils

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


def get_figure(fig_scale = 0.95, fig_width_pts = 498.66258, aspect_ratio = (np.sqrt(5.0) - 1.0) / 2.0):
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

    fig = plt.figure(figsize = _get_fig_dims(fig_scale, fig_width_pts = fig_width_pts, aspect_ratio = aspect_ratio))

    return fig


def save_current_figure(name,
                        name_postfix = '',
                        target_dir = None,
                        img_format = 'pdf',
                        img_scale = 1,
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
    :param img_scale: the scale to save the image at
    :type img_scale: float
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

    plt.savefig(path, dpi = img_scale * plt.gcf().dpi, bbox_inches = 'tight', transparent = transparent)

    logger.debug('Saved matplotlib figure {} to {}'.format(name, path))

    return path


class FigureManager:
    """
    A class that manages a matplotlib figure: creating it, showing it, saving it, and cleaning it up.
    """

    def __init__(self, name, name_postfix = '',
                 fig_scale = 0.95, fig_width_pts = 498.66258, aspect_ratio = (np.sqrt(5.0) - 1.0) / 2.0,
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
        return save_current_figure(name = self.name, name_postfix = self.name_postfix, target_dir = self.target_dir, img_format = self.img_format, img_scale = self.img_scale)

    def __enter__(self):
        if self.close_before:
            plt.close()

        self.fig = get_figure(fig_scale = self.fig_scale, fig_width_pts = self.fig_width_pts, aspect_ratio = self.aspect_ratio)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.save_on_exit:
            self.save()

        if self.show:
            plt.show()

        if self.close_after:
            self.fig.clear()
            plt.close(self.fig)


def xy_plot(name,
            x_data, *y_data,
            line_labels = (), line_kwargs = (),
            figure_manager = None,
            x_scale = 1, y_scale = 1,
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
    :param x_scale: a number to scale the x_data by. Can be a string corresponding to a key in the unit name/value dict.
    :param y_scale: a number to scale the y_data by. Can be a string corresponding to a key in the unit name/value dict.
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

        # determine if scale_x/y is a unit specifier or a number and set scale and labels accordingly
        if type(x_scale) == str:
            scale_x_label = r' (${}$)'.format(utils.unit_names_to_tex_strings[x_scale])
            x_scale = utils.unit_names_to_values[x_scale]
        else:
            scale_x_label = r''
        if type(y_scale) == str:
            scale_y_label = r' (${}$)'.format(utils.unit_names_to_tex_strings[y_scale])
            y_scale = utils.unit_names_to_values[y_scale]
        else:
            scale_y_label = r''

        # zip together each set of y data with its plotting options
        lines = []
        for y, lab, kw in it.zip_longest(y_data, line_labels, line_kwargs):
            if kw is None:  # means there are no kwargs for this y data
                kw = {}
            lines.append(plt.plot(x_data / x_scale, y / y_scale, label = lab, **kw)[0])

        # make any horizontal and vertical lines
        for vl, vkw in it.zip_longest(vlines, vline_kwargs):
            if vkw is None:
                vkw = {}
            kw = {'color': 'black', 'linestyle': '-'}
            kw.update(vkw)
            ax.axvline(x = vl / x_scale, **kw)
        for hl, hkw in it.zip_longest(hlines, hline_kwargs):
            if hkw is None:
                hkw = {}
            kw = {'color': 'black', 'linestyle': '-'}
            kw.update(hkw)
            ax.axhline(y = hl / y_scale, **kw)

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

        ax.set_xlim(left = x_lower_limit / x_scale, right = x_upper_limit / x_scale)
        ax.set_ylim(bottom = y_lower_limit / y_scale, top = y_upper_limit / y_scale)

        ax.grid(True, which = 'major', **grid_kwargs)

        ax.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

        # make title, axis labels, and legend
        if title is not None:
            title = ax.set_title(r'{}'.format(title), fontsize = font_size_title)
            title.set_y(1.06)  # move title up a little
        if x_label is not None:
            x_label = ax.set_xlabel(r'{}'.format(x_label) + scale_x_label, fontsize = font_size_axis_labels)
        if y_label is not None:
            y_label = ax.set_ylabel(r'{}'.format(y_label) + scale_y_label, fontsize = font_size_axis_labels)
        if len(line_labels) > 0:
            if not legend_on_right:
                legend = ax.legend(loc = 'best', fontsize = font_size_legend)
            if legend_on_right:
                legend = ax.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left', borderaxespad = 0., fontsize = font_size_legend, ncol = 1 + (len(line_labels) // 17))

        fig.canvas.draw()  # draw that figure so that the ticks exist, so that we can add more ticks

        if x_extra_ticks is not None and x_extra_tick_labels is not None:
            ax.set_xticks(list(ax.get_xticks()) + list(np.array(x_extra_ticks) / x_scale))  # append the extra tick labels, scaled appropriately
            x_tick_labels = list(ax.get_xticklabels())
            x_tick_labels[-len(x_extra_ticks):] = x_extra_tick_labels  # replace the last set of tick labels (the ones we just added) with the custom tick labels
            ax.set_xticklabels(x_tick_labels)

        if y_extra_ticks is not None and y_extra_tick_labels is not None:
            ax.set_yticks(list(ax.get_yticks()) + list(np.array(y_extra_ticks) / y_scale))  # append the extra tick labels, scaled appropriately
            y_tick_labels = list(ax.get_yticklabels())
            y_tick_labels[-len(y_extra_ticks):] = y_extra_tick_labels  # replace the last set of tick labels (the ones we just added) with the custom tick labels
            ax.set_yticklabels(y_tick_labels)

        # set these AFTER adding extra tick labels so that we don't have to slice into the middle of the label lists above
        ax.tick_params(labeltop = ticks_on_top, labelright = ticks_on_right)

    path = fm.path

    if save_csv:
        csv_path = os.path.splitext(path)[0] + '.csv'
        np.savetxt(csv_path, (x_data, *y_data), delimiter = ',')

        logger.debug('Saved figure data from {} to {}'.format(name, csv_path))

    return path
