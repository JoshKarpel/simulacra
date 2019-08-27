import logging
from typing import Optional, Iterable, Collection, Mapping, Any

import itertools
import collections

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors

from .. import units as u

from . import colors, vutils, figman

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def xy_plot(
    name: str,
    x_data: np.ndarray,
    *y_data: np.ndarray,
    line_labels: Optional[Collection[str]] = None,
    line_kwargs: Optional[Collection[Mapping[str, Any]]] = None,
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
    title_offset: float = vutils.DEFAULT_TITLE_OFFSET,
    title_kwargs: Optional[dict] = None,
    x_label: Optional[str] = None,
    x_label_kwargs: Optional[dict] = None,
    y_label: Optional[str] = None,
    y_label_kwargs: Optional[dict] = None,
    font_size_tick_labels: float = 10,
    ticks_on_top: bool = True,
    ticks_on_right: bool = True,
    legend_on_right: bool = False,
    legend_kwargs: Optional[dict] = None,
    grids: bool = True,
    grid_kwargs: Optional[dict] = None,
    minor_grids: bool = False,
    minor_grid_kwargs: Optional[dict] = None,
    equal_aspect: bool = False,
    figure_manager: Optional[figman.FigureManager] = None,
    **kwargs,
) -> figman.FigureManager:
    """
    Generate and save a generic x vs. y plot.

    This function is suitable for displaying any number of curves as long as
    they all share the same ``x_data``. If you need to display multiple curves
    with independent ``x_data``, see :func:`xxyy_plot``.

    Parameters
    ----------
    name
        The filename for the plot
        (not including path, which should be passed via they keyword argument
        ``target_dir``).
    x_data
        A single array that will be used as x values for all the ``y_data``.
    y_data
        Any number of arrays, each of the same length as ``x_data``, each of
        which will appear as a curve on the figure.
    line_labels
        Labels for each of the ``y_data`` lines.
    line_kwargs
        Keyword arguments for each of the ``y_data`` lines
        (a list of dictionaries).
    x_unit
        The unit for the x-axis.
        Can be a number or the name of a unit as string.
    y_unit
        The unit for the y-axis.
        Can be a number or the name of a unit as string.
    x_log_axis
        If ``True``, the x-axis will be log-scaled.
    y_log_axis
        If ``True``, the y-axis will be log-scaled.
    x_lower_limit
        The lower limit for the x-axis. If ``None``, set automatically from
        the ``x_data``.
    x_upper_limit
        The upper limit for the x-axis. If ``None``, set automatically from
        the ``x_data``.
    y_lower_limit
        The lower limit for the y-axis. If ``None``, set automatically from
        the ``y_data``.
    y_upper_limit
        The upper limit for the y-axis. If ``None``, set automatically from
        the ``y_data``.
    y_pad
        The linear padding factor for the y-axis.
        See :func:`calculate_axis_limits`.
    y_log_pad
        The logarithmic padding factor for the y-axis.
        See :func:`calculate_axis_limits`.
    sym_log_linear_threshold
        When the y-axis is in symmetric log-linear mode,
        this is the threshold for switching from log to linear scaling.
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
    title_kwargs
        Keyword arguments for the title.
    x_label
        The label to display below the x-axis.
    x_label_kwargs
        Keyword arguments for the x-axis label.
    y_label
        The label to display to the left of the y-axis.
    y_label_kwargs
        Keyword arguments for the y-axis label.
    font_size_tick_labels
        The font size for the tick labels.
    ticks_on_top
        If ``True``, axis ticks will also be shown along the top side of the
        plot (in addition to the bottom).
    ticks_on_right
        If ``True``, axis ticks will be also shown along the right side of the
        plot (in addition to the left).
    legend_on_right
        If ``True``, the legend will be displayed hanging on the right side of
        the plot.
    legend_kwargs
        Keyword arguments for the legend.
    grids
        If ``True``, draw major gridlines.
    grid_kwargs
        Keyword arguments for the major gridlines.
    minor_grids
        If ``True``, draw minor gridlines.
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
    figman
        The :class:`FigureManager` that the figure was constructed in.
    """
    line_labels = line_labels or ()
    line_kwargs = line_kwargs or ()

    legend_kwargs = collections.ChainMap(
        legend_kwargs or {}, vutils.DEFAULT_LEGEND_KWARGS
    )
    grid_kwargs = collections.ChainMap(grid_kwargs or {}, vutils.DEFAULT_GRID_KWARGS)
    minor_grid_kwargs = collections.ChainMap(
        minor_grid_kwargs or {}, vutils.DEFAULT_MINOR_GRID_KWARGS
    )

    if figure_manager is None:
        figure_manager = figman.FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = fig.add_subplot(111)
        if equal_aspect:
            ax.set_aspect("equal")

        fm.elements["axis"] = ax

        # ensure data is in numpy arrays
        x_data = np.array(x_data)
        y_data = tuple(np.array(y) for y in y_data)
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

        for which, line_positions, line_kwargs, unit in (
            ("v", vlines, vline_kwargs, x_unit),
            ("h", hlines, hline_kwargs, y_unit),
        ):
            fm.elements[f"{which}lines"] = vutils.attach_h_or_v_lines(
                ax,
                which=which,
                line_positions=line_positions,
                line_kwargs=line_kwargs,
                unit=unit,
            )

        x_lower_limit, x_upper_limit = vutils.set_axis_limits(
            ax,
            x_data,
            lower_limit=x_lower_limit,
            upper_limit=x_upper_limit,
            log=x_log_axis,
            unit=x_unit,
            which="x",
            sym_log_linear_threshold=sym_log_linear_threshold,
        )
        y_lower_limit, y_upper_limit = vutils.set_axis_limits(
            ax,
            *y_data,
            lower_limit=y_lower_limit,
            upper_limit=y_upper_limit,
            log=y_log_axis,
            pad=y_pad,
            log_pad=y_log_pad,
            unit=y_unit,
            which="y",
            sym_log_linear_threshold=sym_log_linear_threshold,
        )

        vutils.make_legend(
            ax,
            figure_manager=fm,
            line_labels=line_labels,
            legend_kwargs=legend_kwargs,
            legend_on_right=legend_on_right,
        )
        _handle_xy_ish_figure_options(
            ax,
            fm,
            title,
            title_kwargs,
            title_offset,
            x_unit,
            x_log_axis,
            x_lower_limit,
            x_upper_limit,
            y_unit,
            y_log_axis,
            y_lower_limit,
            y_upper_limit,
            x_label,
            x_label_kwargs,
            y_label,
            y_label_kwargs,
            y_extra_ticks,
            y_extra_tick_labels,
            x_extra_ticks,
            x_extra_tick_labels,
            ticks_on_right,
            ticks_on_top,
            font_size_tick_labels,
            grids,
            grid_kwargs,
            minor_grids,
            minor_grid_kwargs,
        )

    return fm


def xxyy_plot(
    name,
    x_data,
    y_data,
    line_labels: Optional[Collection[str]] = None,
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
    title_offset=vutils.DEFAULT_TITLE_OFFSET,
    title_kwargs: Optional[dict] = None,
    x_label=None,
    x_label_kwargs: Optional[dict] = None,
    y_label=None,
    y_label_kwargs: Optional[dict] = None,
    font_size_tick_labels=10,
    ticks_on_top=True,
    ticks_on_right=True,
    legend_on_right=False,
    grids: bool = True,
    grid_kwargs: Optional[dict] = None,
    minor_grids: bool = False,
    minor_grid_kwargs=None,
    legend_kwargs=None,
    figure_manager=None,
    **kwargs,
) -> figman.FigureManager:
    line_labels = line_labels or ()

    legend_kwargs = collections.ChainMap(
        legend_kwargs or {}, vutils.DEFAULT_LEGEND_KWARGS
    )
    grid_kwargs = collections.ChainMap(grid_kwargs or {}, vutils.DEFAULT_GRID_KWARGS)
    minor_grid_kwargs = collections.ChainMap(
        minor_grid_kwargs or {}, vutils.DEFAULT_MINOR_GRID_KWARGS
    )

    if figure_manager is None:
        figure_manager = figman.FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = fig.add_subplot(111)

        fm.elements["axis"] = ax

        # ensure data is in numpy arrays
        x_data = [np.array(x) for x in x_data]
        y_data = [np.array(y) for y in y_data]
        line_labels = tuple(line_labels)
        line_kwargs = tuple(line_kwargs)

        x_unit_value, x_unit_tex = u.get_unit_value_and_latex(x_unit)
        y_unit_value, y_unit_tex = u.get_unit_value_and_latex(y_unit)

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

        for which, line_positions, line_kwargs, unit in (
            ("v", vlines, vline_kwargs, x_unit),
            ("h", hlines, hline_kwargs, y_unit),
        ):
            fm.elements[f"{which}lines"] = vutils.attach_h_or_v_lines(
                ax,
                which=which,
                line_positions=line_positions,
                line_kwargs=line_kwargs,
                unit=unit,
            )

        x_lower_limit, x_upper_limit = vutils.set_axis_limits(
            ax,
            *x_data,
            lower_limit=x_lower_limit,
            upper_limit=x_upper_limit,
            log=x_log_axis,
            pad=0,
            log_pad=1,
            unit=x_unit,
            which="x",
        )
        y_lower_limit, y_upper_limit = vutils.set_axis_limits(
            ax,
            *y_data,
            lower_limit=y_lower_limit,
            upper_limit=y_upper_limit,
            log=y_log_axis,
            pad=y_pad,
            log_pad=y_log_pad,
            unit=y_unit,
            which="y",
        )

        vutils.make_legend(
            ax,
            figure_manager=fm,
            line_labels=line_labels,
            legend_kwargs=legend_kwargs,
            legend_on_right=legend_on_right,
        )
        _handle_xy_ish_figure_options(
            ax,
            fm,
            title,
            title_kwargs,
            title_offset,
            x_unit,
            x_log_axis,
            x_lower_limit,
            x_upper_limit,
            y_unit,
            y_log_axis,
            y_lower_limit,
            y_upper_limit,
            x_label,
            x_label_kwargs,
            y_label,
            y_label_kwargs,
            y_extra_ticks,
            y_extra_tick_labels,
            x_extra_ticks,
            x_extra_tick_labels,
            ticks_on_right,
            ticks_on_top,
            font_size_tick_labels,
            grids,
            grid_kwargs,
            minor_grids,
            minor_grid_kwargs,
        )

    return fm


def xy_stackplot(
    name: str,
    x_data: np.ndarray,
    *y_data: np.ndarray,
    line_labels: Optional[Collection[str]] = None,
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
    title_offset: float = vutils.DEFAULT_TITLE_OFFSET,
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
    grids: bool = True,
    grid_kwargs: Optional[dict] = None,
    minor_grids: bool = False,
    minor_grid_kwargs: Optional[dict] = None,
    equal_aspect: bool = False,
    figure_manager: Optional[figman.FigureManager] = None,
    **kwargs,
) -> figman.FigureManager:
    """
    Generate and save a generic x vs. y plot.

    Parameters
    ----------
    name
        The filename for the plot (not including path, which should be passed via they keyword argument ``target_dir``).
    x_data
        A single array that will be used as x-values for all the ``y_data``.
    y_data
        Any number of arrays of the same length as ``x_data``, each of which will appear as a line on the plot.
    line_labels
        Labels for each of the ``y_data`` lines.
    x_unit
        The unit for the x-axis. Can be a number or the name of a unit as string.
    y_unit
        The unit for the y-axis. Can be a number or the name of a unit as string.
    x_log_axis
        If ``True``, the x-axis will be log-scaled.
    y_log_axis
        If ``True``, the y-axis will be log-scaled.
    x_lower_limit
        The lower limit for the x-axis. If ``None``, set automatically from the ``x_data``.
    x_upper_limit
        The upper limit for the x-axis. If ``None``, set automatically from the ``x_data``.
    y_lower_limit
        The lower limit for the y-axis. If ``None``, set automatically from the ``y_data``.
    y_upper_limit
        The upper limit for the y-axis. If ``None``, set automatically from the ``y_data``.
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
    line_labels = line_labels or ()

    legend_kwargs = collections.ChainMap(
        legend_kwargs or {}, vutils.DEFAULT_LEGEND_KWARGS
    )
    grid_kwargs = collections.ChainMap(grid_kwargs or {}, vutils.DEFAULT_GRID_KWARGS)
    minor_grid_kwargs = collections.ChainMap(
        minor_grid_kwargs or {}, vutils.DEFAULT_MINOR_GRID_KWARGS
    )

    if figure_manager is None:
        figure_manager = figman.FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = fig.add_subplot(111)
        if equal_aspect:
            ax.set_aspect("equal")

        fm.elements["axis"] = ax

        # ensure data is in numpy arrays
        x_data = np.array(x_data)
        y_data = tuple(np.array(y) for y in y_data)
        line_labels = tuple(line_labels)

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

        for which, line_positions, line_kwargs, unit in (
            ("v", vlines, vline_kwargs, x_unit),
            ("h", hlines, hline_kwargs, y_unit),
        ):
            fm.elements[f"{which}lines"] = vutils.attach_h_or_v_lines(
                ax,
                which=which,
                line_positions=line_positions,
                line_kwargs=line_kwargs,
                unit=unit,
            )

        x_lower_limit, x_upper_limit = vutils.set_axis_limits(
            ax,
            x_data,
            which="x",
            lower_limit=x_lower_limit,
            upper_limit=x_upper_limit,
            log=x_log_axis,
            unit=x_unit,
        )
        y_lower_limit, y_upper_limit = vutils.set_axis_limits(
            ax,
            *y_data,
            which="y",
            lower_limit=y_lower_limit,
            upper_limit=y_upper_limit,
            log=y_log_axis,
            pad=y_pad,
            log_pad=y_log_pad,
            unit=y_unit,
        )

        vutils.make_legend(
            ax,
            figure_manager=fm,
            line_labels=line_labels,
            legend_kwargs=legend_kwargs,
            legend_on_right=legend_on_right,
        )
        _handle_xy_ish_figure_options(
            ax,
            fm,
            title,
            title_kwargs,
            title_offset,
            x_unit,
            x_log_axis,
            x_lower_limit,
            x_upper_limit,
            y_unit,
            y_log_axis,
            y_lower_limit,
            y_upper_limit,
            x_label,
            x_label_kwargs,
            y_label,
            y_label_kwargs,
            y_extra_ticks,
            y_extra_tick_labels,
            x_extra_ticks,
            x_extra_tick_labels,
            ticks_on_right,
            ticks_on_top,
            font_size_tick_labels,
            grids,
            grid_kwargs,
            minor_grids,
            minor_grid_kwargs,
        )

    return fm


def xyz_plot(
    name: str,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    z_mesh: np.ndarray,
    x_unit: u.Unit = None,
    y_unit: u.Unit = None,
    z_unit: u.Unit = None,
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
    title_offset=vutils.DEFAULT_TITLE_OFFSET,
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
    shading=vutils.ColormapShader.FLAT,
    show_colorbar: bool = True,
    richardson_equator_magnitude=1,
    sym_log_norm_epsilon=1e-3,
    figure_manager=None,
    **kwargs,
) -> figman.FigureManager:
    grid_kwargs = collections.ChainMap(
        grid_kwargs or {},
        {"color": colors.CMAP_TO_OPPOSITE.get(colormap, "black")},
        vutils.DEFAULT_GRID_KWARGS,
    )
    minor_grid_kwargs = collections.ChainMap(
        minor_grid_kwargs or {},
        {"color": colors.CMAP_TO_OPPOSITE.get(colormap, "black")},
        vutils.DEFAULT_MINOR_GRID_KWARGS,
    )

    contour_kwargs = collections.ChainMap(
        contour_kwargs or {}, vutils.DEFAULT_CONTOUR_KWARGS
    )
    contour_label_kwargs = collections.ChainMap(
        contour_label_kwargs or {}, vutils.DEFAULT_CONTOUR_LABEL_KWARGS
    )

    if figure_manager is None:
        figure_manager = figman.FigureManager(name, **kwargs)
    with figure_manager as fm:
        fig = fm.fig
        ax = fig.add_subplot(111)

        plt.set_cmap(colormap)

        x_unit_value, y_unit_value, z_unit_value = u.get_unit_values(
            x_unit, y_unit, z_unit
        )
        z_unit_label = vutils.get_unit_str_for_axis_label(z_unit)

        x_lower_limit, x_upper_limit = vutils.set_axis_limits(
            ax,
            x_mesh,
            which="x",
            lower_limit=x_lower_limit,
            upper_limit=x_upper_limit,
            log=x_log_axis,
            pad=0,
            log_pad=1,
            unit=x_unit,
        )
        y_lower_limit, y_upper_limit = vutils.set_axis_limits(
            ax,
            y_mesh,
            which="y",
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
            z_lower_limit, z_upper_limit = vutils.calculate_axis_limits(
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

        for which, line_positions, line_kwargs, unit in (
            ("v", vlines, vline_kwargs, x_unit),
            ("h", hlines, hline_kwargs, y_unit),
        ):
            fm.elements[f"{which}lines"] = vutils.attach_h_or_v_lines(
                ax,
                which=which,
                line_positions=line_positions,
                line_kwargs=line_kwargs,
                unit=unit,
            )

        if show_colorbar and colormap.name != "richardson":
            cbar = plt.colorbar(mappable=colormesh, ax=ax, pad=0.1)
            if z_label is not None:
                z_label = cbar.set_label(r"{}".format(z_label) + z_unit_label)

        _handle_xy_ish_figure_options(
            ax,
            fm,
            title,
            title_kwargs,
            title_offset,
            x_unit,
            x_log_axis,
            x_lower_limit,
            x_upper_limit,
            y_unit,
            y_log_axis,
            y_lower_limit,
            y_upper_limit,
            x_label,
            x_label_kwargs,
            y_label,
            y_label_kwargs,
            y_extra_ticks,
            y_extra_tick_labels,
            x_extra_ticks,
            x_extra_tick_labels,
            ticks_on_right,
            ticks_on_top,
            font_size_tick_labels,
            grids,
            grid_kwargs,
            minor_grids,
            minor_grid_kwargs,
        )

    return fm


def _handle_xy_ish_figure_options(
    ax,
    figure_manager,
    title,
    title_kwargs,
    title_offset,
    x_unit,
    x_log_axis,
    x_lower_limit,
    x_upper_limit,
    y_unit,
    y_log_axis,
    y_lower_limit,
    y_upper_limit,
    x_label,
    x_label_kwargs,
    y_label,
    y_label_kwargs,
    y_extra_ticks,
    y_extra_tick_labels,
    x_extra_ticks,
    x_extra_tick_labels,
    ticks_on_right,
    ticks_on_top,
    font_size_tick_labels,
    grids,
    grid_kwargs,
    minor_grids,
    minor_grid_kwargs,
):
    ax.tick_params(axis="both", which="major", labelsize=font_size_tick_labels)

    figure_manager.elements["title"] = vutils.set_title(
        ax, title_text=title, title_offset=title_offset, title_kwargs=title_kwargs
    )

    for which, unit, label, label_kwargs in (
        ("x", x_unit, x_label, x_label_kwargs),
        ("y", y_unit, y_label, y_label_kwargs),
    ):
        figure_manager.elements[f"{which}_axis_label"] = vutils.set_axis_label(
            ax, which=which, label_text=label, unit=unit, label_kwargs=label_kwargs
        )

    # draw the figure so that the ticks exist, so that we can add more ticks
    figure_manager.fig.canvas.draw()

    for which, unit, limits in (
        ("x", x_unit, (x_lower_limit, x_upper_limit)),
        ("y", y_unit, (y_lower_limit, y_upper_limit)),
    ):
        vutils.maybe_set_pi_ticks_and_labels(ax, which, unit, *limits)

    for which, unit, extra_ticks, extra_tick_labels in (
        ("x", x_unit, x_extra_ticks, x_extra_tick_labels),
        ("y", y_unit, y_extra_ticks, y_extra_tick_labels),
    ):
        vutils.add_extra_ticks(
            ax,
            which=which,
            extra_ticks=extra_ticks,
            extra_tick_labels=extra_tick_labels,
            unit=unit,
        )

    vutils.set_grids(
        ax,
        grids=grids,
        grid_kwargs=grid_kwargs,
        minor_grids=minor_grids,
        minor_grid_kwargs=minor_grid_kwargs,
        x_log_axis=x_log_axis,
        y_log_axis=y_log_axis,
    )

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
