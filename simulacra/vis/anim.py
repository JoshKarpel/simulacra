from pathlib import Path
from typing import Callable

from tqdm import tqdm

from ..info import Info

from .plots import *


def xyt_plot(
    name,
    x_data,
    t_data,
    *y_funcs,
    y_func_kwargs = (),
    line_labels = (),
    line_kwargs = (),
    figure_manager = None,
    x_unit = None,
    y_unit = None,
    t_unit = None,
    t_fmt_string = r'$t = {} \; {}$',
    t_text_kwargs = None,
    x_log_axis = False,
    y_log_axis = False,
    x_lower_limit = None,
    x_upper_limit = None,
    y_lower_limit = None,
    y_upper_limit = None,
    vlines = (),
    vline_kwargs = (),
    hlines = (),
    hline_kwargs = (),
    x_extra_ticks = None,
    y_extra_ticks = None,
    x_extra_tick_labels = None,
    y_extra_tick_labels = None,
    title = None,
    x_label = None,
    y_label = None,
    font_size_title = 15,
    font_size_axis_labels = 15,
    font_size_tick_labels = 10,
    font_size_legend = 12,
    title_offset = TITLE_OFFSET,
    ticks_on_top = True,
    ticks_on_right = True,
    legend_on_right = False,
    grid_kwargs = None,
    minor_grid_kwargs = None,
    legend_kwargs = None,
    length = 30,
    fig_dpi_scale = 3,
    save_csv = False,
    progress_bar = True,
    **kwargs,
):
    # set up figure and axis
    if figure_manager is None:
        figure_manager = FigureManager(name, save = False, fig_dpi_scale = fig_dpi_scale, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = fig.add_axes([.15, .15, .75, .7])

        grid_kwargs = collections.ChainMap(grid_kwargs or {}, GRID_KWARGS)
        minor_grid_kwargs = collections.ChainMap(minor_grid_kwargs or {}, MINOR_GRID_KWARGS)
        legend_kwargs = collections.ChainMap(legend_kwargs or {}, LEGEND_KWARGS)

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

        x_unit_value, x_unit_tex = u.get_unit_value_and_latex_from_unit(x_unit)
        x_unit_label = get_unit_str_for_axis_label(x_unit)

        y_unit_value, y_unit_tex = u.get_unit_value_and_latex_from_unit(y_unit)
        y_unit_label = get_unit_str_for_axis_label(y_unit)

        t_unit_value, t_unit_tex = u.get_unit_value_and_latex_from_unit(t_unit)
        t_unit_label = t_unit_tex

        attach_h_or_v_lines(ax, vlines, vline_kwargs, unit = x_unit, direction = 'v')
        attach_h_or_v_lines(ax, hlines, hline_kwargs, unit = y_unit, direction = 'h')

        x_lower_limit, x_upper_limit = set_axis_limits_and_scale(ax, x_data,
                                                                 lower_limit = x_lower_limit, upper_limit = x_upper_limit,
                                                                 log = x_log_axis,
                                                                 pad = 0, log_pad = 1,
                                                                 unit = x_unit, direction = 'x')
        y_lower_limit, y_upper_limit = set_axis_limits_and_scale(ax, *(y_func(x_data, t, **y_kwargs) for y_func, y_kwargs in zip(y_funcs, y_func_kwargs) for t in t_data),
                                                                 lower_limit = y_lower_limit, upper_limit = y_upper_limit,
                                                                 log = y_log_axis,
                                                                 pad = 0.05, log_pad = 10,
                                                                 unit = y_unit, direction = 'y')

        ax.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

        # make title, axis labels, and legend
        if title is not None:
            title = ax.set_title(r'{}'.format(title), fontsize = font_size_title)
            title.set_y(title_offset)  # move title up a little
        if x_label is not None:
            x_label = ax.set_xlabel(r'{}'.format(x_label) + x_unit_label, fontsize = font_size_axis_labels)
        if y_label is not None:
            y_label = ax.set_ylabel(r'{}'.format(y_label) + y_unit_label, fontsize = font_size_axis_labels)

        locator = plt.MaxNLocator(prune = 'both', nbins = 5)
        ax.yaxis.set_major_locator(locator)

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
                legend = ax.legend(loc = 'upper right', fontsize = font_size_legend, **legend_kwargs)
            if legend_on_right:
                legend = ax.legend(bbox_to_anchor = (1.15, 1), loc = 'upper left', borderaxespad = 0., fontsize = font_size_legend, ncol = 1 + (len(line_labels) // 17), **legend_kwargs)

        if t_text_kwargs is None:
            t_text_kwargs = {}

        t_text_kwargs = {**T_TEXT_KWARGS, **t_text_kwargs}

        t_str = t_fmt_string.format(u.uround(t_data[0], t_unit, digits = 3), t_unit_label)
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
        utils.ensure_parents_exist(path)

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
                t_text.set_text(t_fmt_string.format(u.uround(t, t_unit, digits = 3), t_unit_label))
                fig.draw_artist(t_text)

                for artist in itertools.chain(ax.xaxis.get_gridlines(), ax.yaxis.get_gridlines()):
                    fig.draw_artist(artist)

                fig.canvas.blit(fig.bbox)

                ffmpeg.stdin.write(fig.canvas.tostring_argb())

                if not progress_bar:
                    logger.debug(f'Wrote frame for t = {u.uround(t, t_unit, 3)} {t_unit} to ffmpeg')

    if save_csv:
        raise NotImplementedError
        # csv_path = os.path.splitext(path)[0] + '.csv'
        # np.savetxt(csv_path, (x_data, *(y_func(x_data, t, **y_kwargs) for y_func, y_kwargs in zip(y_funcs, y_func_kwargs) for t in t_data)), delimiter = ',')
        #
        # logger.debug('Saved figure data from {} to {}'.format(name, csv_path))

    return fm


def xyzt_plot(name,
              x_mesh,
              y_mesh,
              t_data,
              z_func,
              z_func_kwargs = None,
              figure_manager = None,
              x_unit = None,
              y_unit = None,
              t_unit = None,
              z_unit = None,
              t_fmt_string = r'$t = {} \; {}$',
              t_text_kwargs = None,
              x_log_axis = False,
              y_log_axis = False,
              z_log_axis = False,
              x_lower_limit = None,
              x_upper_limit = None,
              y_lower_limit = None,
              y_upper_limit = None,
              z_lower_limit = None,
              z_upper_limit = None,
              vlines = (),
              vline_kwargs = (),
              hlines = (),
              hline_kwargs = (),
              x_extra_ticks = None,
              y_extra_ticks = None,
              x_extra_tick_labels = None,
              y_extra_tick_labels = None,
              title = None,
              x_label = None,
              y_label = None,
              font_size_title = 15,
              font_size_axis_labels = 15,
              font_size_tick_labels = 10,
              ticks_on_top = True,
              ticks_on_right = True,
              grid_kwargs = None,
              minor_grid_kwargs = None,
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
        figure_manager = FigureManager(name, save = False, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = fig.add_axes([.15, .15, .75, .7])

        grid_kwargs = collections.ChainMap(grid_kwargs or {}, GRID_KWARGS)
        minor_grid_kwargs = collections.ChainMap(minor_grid_kwargs or {}, MINOR_GRID_KWARGS)

        # TODO: implement contours for xyzt plot
        # contour_kwargs = utils.handle_dict_default_merge(CONTOUR_KWARGS, contour_kwargs)
        # contour_label_kwargs = utils.handle_dict_default_merge(CONTOUR_LABEL_KWARGS, contour_label_kwargs)

        grid_color = colors.CMAP_TO_OPPOSITE.get(colormap, 'black')
        grid_kwargs['color'] = grid_color
        minor_grid_kwargs['color'] = grid_color

        if z_func_kwargs is None:
            z_func_kwargs = {}

        plt.set_cmap(colormap)

        x_unit_value, x_unit_tex = u.get_unit_value_and_latex_from_unit(x_unit)
        x_unit_label = get_unit_str_for_axis_label(x_unit)

        y_unit_value, y_unit_tex = u.get_unit_value_and_latex_from_unit(y_unit)
        y_unit_label = get_unit_str_for_axis_label(y_unit)

        z_unit_value, z_unit_name = u.get_unit_value_and_latex_from_unit(z_unit)
        z_unit_label = get_unit_str_for_axis_label(z_unit)

        t_unit_value, t_unit_tex = u.get_unit_value_and_latex_from_unit(t_unit)
        t_unit_label = t_unit_tex

        attach_h_or_v_lines(ax, vlines, vline_kwargs, unit = x_unit, direction = 'v')
        attach_h_or_v_lines(ax, hlines, hline_kwargs, unit = y_unit, direction = 'h')

        x_lower_limit, x_upper_limit = set_axis_limits_and_scale(
            ax,
            x_mesh,
            lower_limit = x_lower_limit,
            upper_limit = x_upper_limit,
            log = x_log_axis,
            unit = x_unit,
            direction = 'x', )
        y_lower_limit, y_upper_limit = set_axis_limits_and_scale(
            ax, y_mesh,
            lower_limit = y_lower_limit,
            upper_limit = y_upper_limit,
            log = y_log_axis,
            unit = y_unit,
            direction = 'y',
        )

        if not isinstance(colormap, colors.RichardsonColormap):
            z_lower_limit, z_upper_limit = calculate_axis_limits(
                *(z_func(x_mesh, y_mesh, t, **z_func_kwargs) for t in t_data),
                lower_limit = z_lower_limit, upper_limit = z_upper_limit,
                log = z_log_axis,
            )
            if z_log_axis:
                if z_lower_limit > 0:
                    norm = matplotlib.colors.LogNorm(vmin = z_lower_limit / z_unit_value, vmax = z_upper_limit / z_unit_value)
                else:
                    norm = matplotlib.colors.SymLogNorm(((np.abs(z_lower_limit) + np.abs(z_upper_limit)) / 2) * sym_log_norm_epsilon)
            else:
                norm = matplotlib.colors.Normalize(vmin = z_lower_limit / z_unit_value, vmax = z_upper_limit / z_unit_value)
        else:
            norm = colors.RichardsonNormalization(equator_magnitude = richardson_equator_magnitude)

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

        t_str = t_fmt_string.format(u.uround(t_data[0], t_unit, digits = 3), t_unit_label)
        t_text = plt.figtext(.7, .05, t_str, **t_text_kwargs, animated = True)

        # do animation

        frames = len(t_data)
        fps = int(frames / length)

        path = f"{os.path.join(kwargs['target_dir'], name)}.mp4"
        utils.ensure_parents_exist(path)

        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(fig.bbox)
        canvas_width, canvas_height = fig.canvas.get_width_height()

        cmd = (
            "ffmpeg",
            '-y',
            '-r', f'{fps}',  # choose fps
            '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
            '-pix_fmt', 'argb',  # pixel format
            '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
            '-vcodec', 'mpeg4',  # output encoding
            '-q:v', '1',  # maximum quality
            path,
        )

        if progress_bar:
            t_iter = tqdm(t_data)
        else:
            t_iter = t_data

        with utils.SubprocessManager(cmd, **FFMPEG_PROCESS_KWARGS) as ffmpeg:
            for t in t_iter:
                fig.canvas.restore_region(background)

                z = z_func(x_mesh, y_mesh, t, **z_func_kwargs)

                if shading == ColormapShader.FLAT:
                    z = z[:-1, :-1]

                colormesh.set_array(z.ravel())
                fig.draw_artist(colormesh)

                # update and redraw t strings
                t_text.set_text(t_fmt_string.format(u.uround(t, t_unit, digits = 3), t_unit_label))
                fig.draw_artist(t_text)

                for artist in itertools.chain(ax.xaxis.get_gridlines(), ax.yaxis.get_gridlines()):
                    fig.draw_artist(artist)

                fig.canvas.blit(fig.bbox)

                ffmpeg.stdin.write(fig.canvas.tostring_argb())

                if not progress_bar:
                    logger.debug(f'Wrote frame for t = {u.uround(t, t_unit, 3)} {t_unit} to ffmpeg')

    if save_csv:
        raise NotImplementedError
        # csv_path = os.path.splitext(path)[0] + '.csv'
        # np.savetxt(csv_path, (x_data, *y_data), delimiter = ',')
        #
        # logger.debug('Saved figure data from {} to {}'.format(name, csv_path))

    return fm


def animate(
    figure_manager: FigureManager,
    update_function: Callable,
    update_function_arguments,
    artists = None,
    length: float = 30,
    progress_bar: bool = True,
):
    artists = artists or []

    fig = figure_manager.fig

    path = os.path.join(figure_manager.target_dir, figure_manager.name + '.mp4')

    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    canvas_width, canvas_height = fig.canvas.get_width_height()

    fps = int(len(update_function_arguments) / length)

    cmd = (
        "ffmpeg",
        '-y',
        '-r', f'{fps}',  # choose fps
        '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
        '-pix_fmt', 'argb',  # pixel format
        '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
        '-vcodec', 'mpeg4',  # output encoding
        '-q:v', '1',  # maximum quality
        path,
    )

    with utils.SubprocessManager(cmd, **FFMPEG_PROCESS_KWARGS) as ffmpeg:
        if progress_bar:
            update_function_arguments = tqdm(update_function_arguments)

        for arg in update_function_arguments:
            fig.canvas.restore_region(background)

            update_function(arg)

            for artist in artists:
                fig.draw_artist(artist)

            fig.canvas.blit(fig.bbox)

            ffmpeg.stdin.write(fig.canvas.tostring_argb())


class AxisManager:
    """
    A superclass that manages a matplotlib axis for an Animator.
    """

    def __init__(self):
        self.redraw = []

    def initialize(self, simulation):
        """Hook method for initializing the AxisManager."""
        self.sim = simulation
        self.spec = simulation.spec

        self.initialize_axis()

        logger.debug(f'initialized {self}')

    def assign_axis(self, axis):
        self.axis = axis

        logger.debug(f'assigned {self} to {axis}')

    def initialize_axis(self):
        pass

    def update_axis(self):
        """Hook method for updating the AxisManager's internal state."""
        logger.debug(f'updated axis for {self}')

    def __repr__(self):
        return self.__class__.__name__

    def info(self) -> Info:
        info = Info(header = self.__class__.__name__)

        return info


class Animator:
    """
    A superclass that handles sending frames to ffmpeg to create animations.

    To actually make an animation there are two hook methods that need to be overwritten: _initialize_figure and _update_data.

    An Animator will generally contain a single matplotlib figure with some animation code of its own in addition to a list of :class:`AxisManagers ~<AxisManager>` that handle axes on the figure.

    For this class to function correctly :code:`ffmpeg` must be visible on the system path.
    """

    def __init__(
        self,
        postfix = '',
        target_dir: Optional[Union[Path, str]] = None,
        length: int = 60,
        fps: int = 30,
        colormap = plt.cm.get_cmap('viridis'),
    ):
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

    def initialize(self, sim):
        """
        Initialize the Animation by setting the Simulation and Specification, determining the target path for output, determining fps and decimation, and setting up the ffmpeg subprocess.

        _initialize_figure() is called during the execution of this method.
        It should assign a matplotlib figure object to ``self.fig``.

        The simulation should have an attribute available_animation_frames that returns an int describing how many raw frames might be available for use by the animation.

        :param sim: a Simulation for the AxisManager to collect data from
        """
        self.sim = sim
        self.spec = sim.spec

        self.file_name = f'{self.sim.file_name}{self.postfix}.mp4'
        self.file_path = os.path.join(self.target_dir, self.file_name)
        utils.ensure_parents_exist(self.file_path)
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
        for axman in self.axis_managers:
            logger.debug(f'initializing axis {axman} for {self}')
            axman.initialize(sim)

        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        canvas_width, canvas_height = self.fig.canvas.get_width_height()

        self.cmd = (
            'ffmpeg',
            '-y',
            '-r', '{}'.format(self.fps),  # choose fps
            '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
            '-pix_fmt', 'argb',  # pixel format
            '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
            '-vcodec', 'mpeg4',  # output encoding
            '-q:v', '1',  # maximum quality
            self.file_path,
        )

        self.ffmpeg = subprocess.Popen(self.cmd, **FFMPEG_PROCESS_KWARGS)

        logger.info(f'initialized {self} for {self.sim}')

    def cleanup(self):
        """
        Cleanup method for the Animator's ffmpeg subprocess.

        Should always be called via a try...finally clause (namely, in the finally) in Simulation.run_simulation.
        """
        self.ffmpeg.communicate()
        plt.close(self.fig)
        logger.info('Cleaned up {}'.format(self))

    def _initialize_figure(self):
        """
        Hook for a method to initialize the Animator's figure.

        Make sure that any plot element that will be mutated during the animation is created using the animation = True keyword argument and has a reference in self.redraw.
        """
        logger.debug(f'initialized figure for {self}')

    def _update_data(self):
        """Hook for a method to update the data for each animated figure element."""
        for axman in self.axis_managers:
            axman.update_axis()

        logger.debug(f'{self} updated data from {self.sim}')

    def _redraw_frame(self):
        """Redraw the figure frame."""
        logger.debug('redrawing frame for {}'.format(self))

        plt.set_cmap(self.colormap)  # make sure the colormap is correct, in case other figures have been created somewhere

        self.fig.canvas.restore_region(self.background)  # copy the static background back onto the figure

        self._update_data()  # get data from the Simulation and update any plot elements that need to be redrawn

        # draw everything that needs to be redrawn (any plot elements that will be mutated during the animation should be added to self.redraw)
        for rd in itertools.chain(self.redraw, *(ax.redraw for ax in self.axis_managers)):
            self.fig.draw_artist(rd)

        self.fig.canvas.blit(self.fig.bbox)  # blit the canvas, finalizing all of the draw_artists

        logger.debug(f'redrew frame for {self}')

    def send_frame_to_ffmpeg(self):
        """Redraw anything that needs to be redrawn, then write the figure to an RGB string and send it to ffmpeg."""
        logger.debug('{} sending frame to ffpmeg from {} {}'.format(self, self.sim.__class__.__name__, self.sim.name))

        self._redraw_frame()

        self.ffmpeg.stdin.write(self.fig.canvas.tostring_argb())

        logger.debug('{} sent frame to ffpmeg from {} {}'.format(self, self.sim.__class__.__name__, self.sim.name))

    def __repr__(self):
        return "{}(postfix = '{}')".format(self.__class__.__name__, self.postfix)

    def info(self) -> Info:
        info = Info(header = f'{self.__class__.__name__}: {self.postfix}')

        info.add_field('Length', f'{self.length} s')
        info.add_field('FPS', f'{self.fps}')

        for axis_manager in self.axis_managers:
            info.add_info(axis_manager.info())

        return info
