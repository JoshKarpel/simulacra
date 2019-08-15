import datetime
import itertools
import time
import logging
import enum
from typing import Callable, Iterable, Collection, Any

from . import timing

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def grouper(iterable: Iterable, n: int, fill_value=None) -> Iterable:
    """

    Parameters
    ----------
    iterable
        An iterable to chunk
    n
        The size of the chunks
    fill_value
        A value to fill with when iterable has run out of values, but the last chunk isn't full

    Returns
    -------
    Iterable
        An iterator over length ``n`` groups of the input iterable
    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fill_value)


class StrEnum(str, enum.Enum):
    """An :class:`enum.Enum` whose members are also strings."""

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.value.upper()}"

    def __str__(self):
        return self.value
