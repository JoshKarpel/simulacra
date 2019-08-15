import logging
from typing import Union, Iterable, Optional, Dict, Any

import collections
import fractions
import itertools

import numpy as np
from matplotlib import pyplot as plt

from simulacra import units as u, utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_TITLE_KWARGS = dict(fontsize=16)
DEFAULT_LABEL_KWARGS = dict(fontsize=14)
DEFAULT_GRID_KWARGS = dict(linestyle="-", color="black", linewidth=0.5, alpha=0.4)
DEFAULT_MINOR_GRID_KWARGS = dict(linestyle="-", color="black", linewidth=0.5, alpha=0.2)
DEFAULT_COLORMESH_GRID_KWARGS = dict(linestyle="-", linewidth=0.5, alpha=0.4)
DEFAULT_LEGEND_KWARGS = dict(loc="best")
DEFAULT_HVLINE_KWARGS = dict(linestyle="-", color="black")
DEFAULT_CONTOUR_KWARGS = dict()
DEFAULT_CONTOUR_LABEL_KWARGS = dict(inline=1, fontsize=8)

DEFAULT_TITLE_OFFSET = 1.15


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
    which
    unit
    lower_limit
    upper_limit

    Returns
    -------

    """
    if unit == "rad":
        ticks, labels = get_pi_ticks_and_labels(lower_limit, upper_limit)
        set_axis_ticks_and_ticklabels(ax, which, ticks, labels)


def set_axis_ticks_and_ticklabels(
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
            getattr(axis, f"set_{direction}scale")("log")
        else:
            getattr(axis, f"set_{direction}scale")(
                "symlog", **{f"linthresh{direction}": sym_log_linear_threshold}
            )

    return getattr(axis, f"set_{direction}lim")(
        lower_limit / unit_value, upper_limit / unit_value
    )


def set_title(
    axis: plt.Axes,
    title_text: Optional[str],
    title_offset: float = DEFAULT_TITLE_OFFSET,
    title_kwargs: Optional[dict] = None,
) -> Optional[plt.Text]:
    title_kwargs = collections.ChainMap(title_kwargs or {}, DEFAULT_TITLE_KWARGS)
    if title_text is not None:
        title = axis.set_title(title_text, **title_kwargs)
        title.set_y(title_offset)
        return title


def set_axis_label(
    axis: plt.Axes,
    which: str,
    label: Optional[str] = None,
    unit: Optional[u.Unit] = None,
    label_kwargs: Optional[dict] = None,
) -> Optional[plt.Text]:
    label_kwargs = collections.ChainMap(label_kwargs or {}, DEFAULT_LABEL_KWARGS)
    unit_label = get_unit_str_for_axis_label(unit)
    if label is not None:
        ax_label = getattr(axis, f"set_{which}label")(
            label + unit_label, **label_kwargs
        )
        return ax_label


def get_unit_str_for_axis_label(unit: u.Unit):
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


def attach_h_or_v_lines(
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
        kw = {**DEFAULT_HVLINE_KWARGS, **kw}
        lines.append(getattr(axis, f"ax{direction}line")(position / unit_value, **kw))

    return lines


def add_extra_ticks(ax, which: str, extra_ticks, labels, unit: u.Unit):
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


class ColormapShader(utils.StrEnum):
    FLAT = "flat"
    GOURAUD = "gouraud"
