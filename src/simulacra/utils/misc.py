import datetime
import itertools
import time
import logging
import enum
from typing import Optional, Callable, Iterable

from . import timing

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def try_loop(
    *functions_to_run,
    wait_after_success: datetime.timedelta = datetime.timedelta(hours = 1),
    wait_after_failure: datetime.timedelta = datetime.timedelta(minutes = 1),
    begin_text: str = 'Beginning loop',
    complete_text: str = 'Completed loop',
):
    """
    Run the given functions in a constant loop.

    :param functions_to_run: call these functions in order during each loop
    :param wait_after_success: a datetime.timedelta object specifying how long to wait after a loop completes
    :param wait_after_failure: a datetime.timedelta object specifying how long to wait after a loop fails (i.e., raises an exception)
    :param begin_text: a string to print at the beginning of each loop
    :type begin_text: str
    :param complete_text: a string to print at the end of each loop
    :type complete_text: str
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
    Collect data from iterable into fixed-length chunks or blocks of length n

    See https://docs.python.org/3/library/itertools.html#itertools-recipes

    :param iterable: an iterable to chunk
    :param n: the size of the chunks
    :param fill_value: a value to fill with when iterable has run out of values, but the last chunk isn't full
    :return:
    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue = fill_value)


class StrEnum(str, enum.Enum):
    def __repr__(self):
        return f'{self.__class__.__name__}.{self.value.upper()}'

    def __str__(self):
        return self.value
