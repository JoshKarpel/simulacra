import datetime
import functools
import itertools
import multiprocessing
import os
import time
import logging
import enum
from typing import Optional, Union, NamedTuple, Callable, Iterable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def find_or_init_sim(spec, search_dir: Optional[str] = None, file_extension = '.sim'):
    """
    Try to load a :class:`simulacra.Simulation` by looking for a pickled :class:`simulacra.core.Simulation` named ``{search_dir}/{spec.file_name}.{file_extension}``.
    If that fails, create a new Simulation from `spec`.

    Parameters
    ----------
    spec : :class:`simulacra.sims.Specification`
    search_dir : str
    file_extension : str

    Returns
    -------
    :class:`simulacra.sims.Simulation`
    """
    try:
        if search_dir is None:
            search_dir = os.getcwd()
        path = os.path.join(search_dir, spec.file_name + file_extension)
        sim = sims.Simulation.load(file_path = path)
    except FileNotFoundError:
        sim = spec.to_sim()

    return sim


def run_in_process(func, args = (), kwargs = None):
    """
    Run a function in a separate process.

    :param func: the function to run
    :param args: positional arguments for function
    :param kwargs: keyword arguments for function
    """
    if kwargs is None:
        kwargs = {}

    with multiprocessing.Pool(processes = 1) as pool:
        output = pool.apply(func, args, kwargs)

    return output


def run_from_simlib(spec, simlib = None, **kwargs):
    sim = find_or_init_sim(spec, search_dir = simlib)

    if sim.status != sims.Status.FINISHED:
        sim.run(**kwargs)
        sim.save(target_dir = simlib)

    return sim


def multi_map(func: Callable, targets: Iterable, processes: Optional[int] = None, **kwargs):
    """
    Map a function over a list of inputs using multiprocessing.

    Function should take a single positional argument (an element of targets) and any number of keyword arguments, which must be the same for each target.

    Parameters
    ----------
    func : a callable
        The function to call on each of the `targets`.
    targets : an iterable
        An iterable of arguments to call the function on.
    processes : :class:`int`
        The number of processes to use. Defaults to the half of the number of cores on the computer.
    kwargs
        Keyword arguments are passed to :func:`multiprocess.pool.map`.

    Returns
    -------
    :class:`tuple`
        The outputs of the function being applied to the targets.
    """
    if processes is None:
        processes = max(int(multiprocessing.cpu_count() / 2) - 1, 1)

    with multiprocessing.Pool(processes = processes) as pool:
        output = pool.map(func, targets, **kwargs)

    return tuple(output)


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
        'wall_time_start', 'wall_time_end', 'wall_time_elapsed',
        'proc_time_start', 'proc_time_end', 'proc_time_elapsed'
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
        self.wall_time_end = datetime.datetime.now()
        self.proc_time_end = time.process_time()

        self.wall_time_elapsed = self.wall_time_end - self.wall_time_start
        self.proc_time_elapsed = self.proc_time_end - self.proc_time_start

    def __str__(self):
        if self.wall_time_end is None:
            return 'Timer started at {}, still running'.format(self.wall_time_start)
        else:
            return 'Timer started at {}, ended at {}, elapsed time {}. Process time: {}.'.format(self.wall_time_start, self.wall_time_end, self.wall_time_elapsed, datetime.timedelta(seconds = self.proc_time_elapsed))


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

        with BlockTimer() as timer:
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
