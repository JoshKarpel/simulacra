import datetime
import itertools
import time
import logging
import enum
from typing import Callable, Iterable, Collection, Any

from . import timing

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def try_loop(
    *functions_to_run: Collection[Callable[[Any], None]],
    wait_after_success: datetime.timedelta = datetime.timedelta(hours = 1),
    wait_after_failure: datetime.timedelta = datetime.timedelta(minutes = 1),
    begin_text: str = 'Beginning loop',
    complete_text: str = 'Completed loop',
):
    """
    Run the given functions in a loop.

    Parameters
    ----------
    functions_to_run
        Call these functions in order during each loop
    wait_after_success
        How long to wait after a loop completes before trying again if the loop succeeds
    wait_after_failure
        How long to wait after a loop fails (i.e., raises an exception) before trying again
    begin_text
        A string to print at the beginning of the loop
    complete_text
        A string to print at the end of the loop
    """
    while True:
        logger.info(begin_text)

        with timing.BlockTimer() as timer:
            failed = False
            for f in functions_to_run:
                try:
                    f()
                except Exception as e:
                    logger.exception(f'Exception encountered while executing loop function: {f}')
                    failed = True

        logger.info(f'{complete_text}. Elapsed time: {timer.wall_time_elapsed}')

        if failed:
            s, wait = 'failed', wait_after_failure
        else:
            s, wait = 'succeeded', wait_after_success

        logger.info(f'Loop cycle {s}, next cycle in {wait.total_seconds()} seconds')
        time.sleep(wait.total_seconds())


def grouper(iterable: Iterable, n: int, fill_value = None) -> Iterable:
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
    return itertools.zip_longest(*args, fillvalue = fill_value)


class StrEnum(str, enum.Enum):
    """An :class:`enum.Enum` whose members are also strings."""

    def __repr__(self):
        return f'{self.__class__.__name__}.{self.value.upper()}'

    def __str__(self):
        return self.value
