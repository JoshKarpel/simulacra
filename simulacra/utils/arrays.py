import logging
from typing import Any

import collections

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

NearestEntry = collections.namedtuple("NearestEntry", ("index", "value", "target"))


def find_nearest_entry(array: np.ndarray, target: Any) -> NearestEntry:
    """
    Returns the ``(index, value, target)`` of the ``array`` entry closest to the
    given ``target``.

    Parameters
    ----------
    array
        The array to search for ``target`` in.
    target
        The value to search for in ``array``.

    Returns
    -------
    nearest_entry :
        A tuple containing the index of the nearest value to the target,
        that value, and the original target value.
    """
    array = np.array(array)

    index = np.argmin(np.abs(array - target))
    value = array[index]

    return NearestEntry(index, value, target)
