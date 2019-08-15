from typing import Iterable

import itertools
import logging
import enum
import subprocess


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


class SubprocessManager:
    def __init__(self, cmd_string, **subprocess_kwargs):
        self.cmd_string = cmd_string
        self.subprocess_kwargs = subprocess_kwargs

        self.name = self.cmd_string[0]

        self.subprocess = None

    def __enter__(self):
        self.subprocess = subprocess.Popen(self.cmd_string, **self.subprocess_kwargs)

        logger.debug(f"Opened subprocess {self.name}")

        return self.subprocess

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.subprocess.communicate()
            logger.debug(f"Closed subprocess {self.name}")
        except AttributeError:
            logger.warning(
                f"Exception while trying to close subprocess {self.name}, possibly not closed"
            )
