import collections
import logging
from typing import Optional, Union, NamedTuple, Callable, Iterable, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

KeyValueArrays = collections.namedtuple("KeyValueArrays", ("key_array", "value_array"))


def dict_to_arrays(dct: dict, key: Optional[Callable] = None) -> KeyValueArrays:
    """
    Return the keys and values of a dictionary as two numpy arrays, in key-sorted order.

    Parameters
    ----------
    dct
        The dictionary to array-ify.
    key
        An optional function to use as the ``key`` for ``sorted``.
        It is passed each element of ``dct.items()``.

    Returns
    -------

    """
    key_list = []
    val_list = []

    for key, val in sorted(dct.items(), key=key):
        key_list.append(key)
        val_list.append(val)

    return KeyValueArrays(np.array(key_list), np.array(val_list))


NearestEntry = collections.namedtuple("NearestEntry", ("index", "value", "target"))


def find_nearest_entry(array: np.ndarray, target: Union[float, int]) -> NearestEntry:
    """
    Returns the ``(index, value, target)`` of the `array` entry closest to the given `target`.
    Parameters
    ----------
    array : :class:`numpy.ndarray`
        The array to for `target` in.
    target
        The value to search for in `array`.
    Returns
    -------
    :class:`tuple`
        A tuple containing the index of the nearest value to the target, that value, and the original target value.
    """
    array = np.array(array)  # turn the array into a numpy array

    index = np.argmin(np.abs(array - target))
    value = array[index]

    return NearestEntry(index, value, target)
