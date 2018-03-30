import datetime
import functools
import time
import logging
from typing import Optional, Callable, Iterable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def timed(func: Callable):
    """A decorator that times the execution of the decorated function. A log message is emitted at level ``DEBUG`` with the timing information."""

    @functools.wraps(func)
    def timed_wrapper(*args, **kwargs):
        time_start = datetime.datetime.now()
        val = func(*args, **kwargs)
        time_end = datetime.datetime.now()

        time_elapsed = time_end - time_start

        msg = f'Execution of {func} took {time_elapsed}'
        logger.debug(msg)
        print(msg)

        return val

    return timed_wrapper


class BlockTimer:
    """A context manager that times the code in the ``with`` block. Print the :class:`BlockTimer` after exiting the block to see the results."""

    __slots__ = (
        'wall_time_start',
        'wall_time_end',
        'wall_time_elapsed',
        'proc_time_start',
        'proc_time_end',
        'proc_time_elapsed',
    )

    def __init__(self):
        self.wall_time_start = None
        self.wall_time_end = None
        self.wall_time_elapsed = None

        self.proc_time_start = None
        self.proc_time_end = None
        self.proc_time_elapsed = None

    def __enter__(self):
        self.wall_time_start = datetime.datetime.now()
        self.proc_time_start = time.process_time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.proc_time_end = time.process_time()
        self.wall_time_end = datetime.datetime.now()

        self.wall_time_elapsed = self.wall_time_end - self.wall_time_start
        self.proc_time_elapsed = datetime.timedelta(seconds = self.proc_time_end - self.proc_time_start)

    def __str__(self):
        if self.wall_time_end is None:
            return f'{self.__class__.__name__} started at {self.wall_time_start} and is still running'

        return f'{self.__class__.__name__} started at {self.wall_time_start}, ended at {self.wall_time_end}. Elapsed time: {self.wall_time_elapsed}. Process time: {self.proc_time_elapsed}.'
