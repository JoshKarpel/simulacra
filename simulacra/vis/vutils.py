import logging
from typing import (
    Union,
    Iterable,
    Optional,
    Dict,
    Any,
    Tuple,
    Collection,
    List,
    Mapping,
)

import collections
import fractions
import itertools

import numpy as np
from matplotlib import pyplot as plt

from . import figman
from .. import units as u, utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_TITLE_KWARGS = dict(fontsize=16)
DEFAULT_LABEL_KWARGS = dict(fontsize=14)
DEFAULT_GRID_KWARGS = dict(linestyle="-", color="black", linewidth=0.5, alpha=0.4)
DEFAULT_MINOR_GRID_KWARGS = dict(linestyle="-", color="black", linewidth=0.5, alpha=0.2)
DEFAULT_COLORMESH_GRID_KWARGS = dict(linestyle="-", linewidth=0.5, alpha=0.4)
DEFAULT_LEGEND_KWARGS = dict(loc="best")
DEFAULT_HVLINE_KWARGS = dict(linestyle="-", color="black")
DEFAULT_CONTOUR_KWARGS = dict(linewidth=1, color="white")
DEFAULT_CONTOUR_LABEL_KWARGS = dict(inline=1, fontsize=8, color="white")

DEFAULT_TITLE_OFFSET = 1.15


class ColormapShader(utils.StrEnum):
    FLAT = "flat"
    GOURAUD = "gouraud"


def get_pi_ticks_and_labels(
    lower_limit: Union[float, int] = 0,
    upper_limit: Union[float, int] = u.twopi,
    common_denominator: int = 4,
) -> Tuple[List[float], List[str]]:
    """
    Produce fractions-of-pi ticks and labels.

    .. warning ::

        This function doesn't work very well for large ranges.

    Parameters
    ----------
    lower_limit
        The lower limit for the axis.
    upper_limit
        The upper limit for the axis.
    common_denominator
        The common denominator to use for all of the fractions.

    Returns
    -------
    ticks, labels
        The calculated ticks and labels.
    """
    low = int(np.floor(lower_limit / u.pi))
    high = int(np.ceil(upper_limit / u.pi))

    ticks = list(
        fractions.Fraction(n, common_denominator)
        for n in range(low * common_denominator, (high * common_denominator) + 1)
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


def maybe_set_pi_ticks_and_labels(
    ax, which: str, unit: u.Unit, lower_limit, upper_limit
):
    """
    This function handles the special case where an axis's units are in radians,
    in which case we should use a special set of axis ticks with fractions-of-pi
    labels.

    Parameters
    ----------
    ax
        The :class:`matplotlib.pyplot.Axes` to act on.
    which : {``'x'``, ``'y'``, ``'z'``}
        Which axis to act on.
    unit
        The unit for the axis. If it is ``'rad'``, this function will overwrite
        the axis ticks and labels into a more appropriate display.
    lower_limit
        The lower limit for the axis.
    upper_limit
        The upper limit for the axis.

    Returns
    -------

    """
    if unit == "rad":
        ticks, labels = get_pi_ticks_and_labels(lower_limit, upper_limit)
        set_axis_ticks_and_ticklabels(ax, which, ticks, labels)


def set_axis_ticks_and_ticklabels(
    ax: plt.Axes, which: str, ticks: Iterable[Union[float, int]], labels: Iterable[str]
):
    """
    Set the ticks and labels for ``axis` along `direction`.

    Parameters
    ----------
    ax
        The :class:`matplotlib.pyplot.Axes` to act on.
    which : {``'x'``, ``'y'``, ``'z'``}
        Which axis to act on.
    ticks
        The tick positions.
    labels
        The tick labels.
    """
    getattr(ax, f"set_{which}ticks")(ticks)
    getattr(ax, f"set_{which}ticklabels")(labels)


def set_axis_limits(
    axis: plt.Axes,
    *data: np.ndarray,
    which: str,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
    unit: u.Unit = None,
    pad: float = 0,
    log: bool = False,
    log_pad: float = 1,
    sym_log_linear_threshold: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculates and sets appropriate lower and upp axis limits for showing any
    number of ``data``.

    .. attention::

        This function **both sets and returns** the calculated axis limits. The
        limits are returned as a convenience for later use. If you just want
        to calculate the limits, see :func:`calculate_axis_limits`.

    Parameters
    ----------
    axis
        The :class:`matplotlib.pyplot.Axes` to act on.
    data
        The data sets to calculate axis limits for.
    which : {``'x'``, ``'y'``, ``'z'``}
        Which axis to add the label to.
    lower_limit
        If not ``None``, use this instead of the calculated lower limit.
    upper_limit
        If not ``None``, use this instead of the calculated upper limit.
    unit
        The unit for the axis.
    log
        If ``True``, the limits will be given in log space instead of linear
        space.
    pad
        This many extra fractions of the total data range will be added in
        each direction (upper and lower), if not in ``log`` mode.
    log_pad
        This many extra orders of magnitude will be added to the limits in
        each direction (upper and lower), if in ``log`` mode.
    sym_log_linear_threshold
        The linear-to-log threshold, for symmetric log-linear mode.

    Returns
    -------
    (lower_limit, upper_limit)
        The lower and upper limits, in the specified units.
    """
    unit_value = u.get_unit_value(unit)

    lower_limit, upper_limit = calculate_axis_limits(
        *data,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        log=log,
        pad=pad,
        log_pad=log_pad,
    )

    if log:
        if lower_limit > 0:
            getattr(axis, f"set_{which}scale")("log")
        else:
            getattr(axis, f"set_{which}scale")(
                "symlog", **{f"linthresh{which}": sym_log_linear_threshold}
            )

    return getattr(axis, f"set_{which}lim")(
        lower_limit / unit_value, upper_limit / unit_value
    )


def calculate_axis_limits(
    *data: np.ndarray,
    lower_limit: Optional[Union[float, int]] = None,
    upper_limit: Optional[Union[float, int]] = None,
    log: bool = False,
    pad: float = 0,
    log_pad: float = 1,
) -> Tuple[float, float]:
    """
    Calculates appropriate lower and upp axis limits for showing any number
    of ``data``.

    Parameters
    ----------
    data
        The data that axis limits need to be constructed for.
    lower_limit
        If not ``None``, use this instead of the calculated lower limit.
    upper_limit
        If not ``None``, use this instead of the calculated upper limit.
    log
        If ``True``, the limits will be given in log space instead of linear
        space.
    pad
        This many extra fractions of the total data range will be added in
        each direction (upper and lower), if not in ``log`` mode.
    log_pad
        This many extra orders of magnitude will be added to the limits in
        each direction (upper and lower), if in ``log`` mode.

    Returns
    -------
    (lower_limit, upper_limit)
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


def set_title(
    axis: plt.Axes,
    *,
    title_text: Optional[str],
    title_offset: float = DEFAULT_TITLE_OFFSET,
    title_kwargs: Optional[dict] = None,
) -> Optional[plt.Text]:
    """
    Create a figure title.
    The title will only be created if ``title_text`` is given.

    Parameters
    ----------
    axis
        The :class:`matplotlib.pyplot.Axes` to act on.
    title_text
        The desired title text.
    title_offset
        The vertical offset of the title from its normal position.
    title_kwargs
        Additional keyword arguments to pass to :meth:`matplotlib.pyplot.Axes.set_title`.

    Returns
    -------
    title
        The title that was created, if one was created.
    """
    kwargs = collections.ChainMap(title_kwargs or {}, DEFAULT_TITLE_KWARGS)
    if title_text is not None:
        title = axis.set_title(title_text, **kwargs)
        title.set_y(title_offset)
        return title
    return None


def set_axis_label(
    axis: plt.Axes,
    *,
    which: str,
    label_text: Optional[str] = None,
    label_kwargs: Optional[dict] = None,
    unit: u.Unit = None,
) -> Optional[plt.Text]:
    """
    Create an axis label.
    The label will only be created if ``label`` is given.

    Parameters
    ----------
    axis
        The :class:`matplotlib.pyplot.Axes` to act on.
    which : {``'x'``, ``'y'``, ``'z'``}
        Which axis to add the label to.
    label_text
        The desired label text.
    label_kwargs
        Keyword arguments for the label.
    unit
        If given, the label will have a unit specifier appended to it.

    Returns
    -------
    label
        The axis label that was created, if one was created.
    """
    kwargs = collections.ChainMap(label_kwargs or {}, DEFAULT_LABEL_KWARGS)
    unit_label = get_unit_str_for_axis_label(unit)
    if label_text is not None:
        ax_label = getattr(axis, f"set_{which}label")(label_text + unit_label, **kwargs)
        return ax_label
    return None


def get_unit_str_for_axis_label(unit: u.Unit) -> str:
    """
    Get a LaTeX-formatted unit label for ``unit``.
    Meant to be appended to an existing axis label.

    Parameters
    ----------
    unit
        The unit to get the formatted string for.

    Returns
    -------
    label
        The unit label.
    """
    unit_tex = u.get_unit_latex(unit)

    if unit_tex != "":
        return fr" (${unit_tex}$)"
    else:
        return ""


def attach_h_or_v_lines(
    ax: plt.Axes,
    *,
    which: str,
    line_positions: Optional[Iterable[float]] = None,
    line_kwargs: Optional[Iterable[Dict[str, Any]]] = None,
    unit: u.Unit = None,
):
    """
    Add hlines or vlines to an axis.

    Parameters
    ----------
    ax
        The :class:`matplotlib.pyplot.Axes` to act on.
    which : {``'h'``, ``'v'``}
        Which direction the lines are for (horizontal ``'h'`` or vertical ``'v'``).
    line_positions
        The positions of the lines to draw.
    line_kwargs
        The keyword arguments for each draw command.
    unit
        The unit for the lines.

    Returns
    -------
    lines
        A list containing the lines that were drawn.
    """
    unit_value = u.get_unit_value(unit)

    lines = []

    for position, kwargs in itertools.zip_longest(
        line_positions or (), line_kwargs or (), fillvalue=None
    ):
        if line_positions is None:
            logger.warning(
                f"Got more {which}line keyword arguments than lines, ignoring extras"
            )
            continue
        kwargs = collections.ChainMap(kwargs or {}, DEFAULT_HVLINE_KWARGS)
        line = getattr(ax, f"ax{which}line")(position / unit_value, **kwargs)
        lines.append(line)

    return lines


def add_extra_ticks(
    ax: plt.Axes,
    *,
    which: str,
    extra_ticks: np.array,
    extra_tick_labels: Optional[Collection[str]] = None,
    unit: u.Unit = None,
):
    """
    Adds extra tick marks to a :class:`matplotlib.pyplot.Axes`.

    Parameters
    ----------
    ax
        The :class:`matplotlib.pyplot.Axes` to add ticks to.
    which : {``'x'``, ``'y'``, ``'z'``}
        Which axis to add ticks to.
    extra_ticks
        The locations of the extra ticks.
    extra_tick_labels
        The labels for the extra ticks.
        If ``None``, they will not be labelled.
        If given, must be the same length as ``extra_ticks``.
    unit
        The unit for the axis.

    Returns
    -------

    """
    unit_value = u.get_unit_value(unit)

    if extra_ticks is not None:
        # append the extra tick labels, scaled appropriately
        existing_ticks = list(getattr(ax, f"get_{which}ticks"))
        new_ticks = list(np.array(extra_ticks) / unit_value)
        getattr(ax, f"set_{which}ticks")(existing_ticks + new_ticks)

        # replace the last set of tick labels (the ones we just added) with the custom tick labels
        if extra_tick_labels is not None:
            tick_labels = list(ax.get_xticklabels())
            tick_labels[-len(extra_ticks) :] = extra_tick_labels
            getattr(ax, f"set_{which}ticklabels")(tick_labels)


def set_grids(
    ax: plt.Axes,
    *,
    grids: bool = True,
    grid_kwargs: Mapping[str, Any],
    minor_grids: bool = False,
    minor_grid_kwargs: Mapping[str, Any],
    x_log_axis: bool = False,
    y_log_axis: bool = False,
):
    ax.grid(grids, which="major", **grid_kwargs)
    if x_log_axis:
        ax.grid(minor_grids, which="minor", axis="x", **minor_grid_kwargs)
    if y_log_axis:
        ax.grid(minor_grids, which="minor", axis="y", **minor_grid_kwargs)


def make_legend(
    ax: plt.Axes,
    *,
    figure_manager: figman.FigureManager,
    line_labels: Collection[str],
    legend_kwargs: Mapping[str, Any],
    legend_on_right: bool = False,
):
    if len(line_labels) > 0 or "handles" in legend_kwargs:
        if not legend_on_right:
            figure_manager.elements["legend"] = ax.legend(**legend_kwargs)
        if legend_on_right:
            legend_kwargs = collections.ChainMap(
                dict(
                    loc="upper left",
                    bbox_to_anchor=(1.2, 1),
                    borderaxespad=0,
                    ncol=1 + (len(line_labels) // 17),
                ),
                legend_kwargs,
            )
            figure_manager.elements["legend"] = ax.legend(**legend_kwargs)
