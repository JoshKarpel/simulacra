"""
Simulacra utility sub-package.


Copyright 2017 Josh Karpel

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import collections
import datetime
import functools
import itertools
import multiprocessing
import subprocess
import os
import sys
import time
import logging
from typing import Optional, Union, NamedTuple, Callable, Iterable

import numpy as np
import psutil

from . import core
from .units import uround

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

LOG_FORMATTER = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt = '%y/%m/%d %H:%M:%S')  # global log format specification

key_value_arrays = collections.namedtuple('key_value_arrays', ('key_array', 'value_array'))


def field_str(obj, *fields, digits: int = 3):
    """
    Generate a repr-like string from the object's attributes.

    Each field should be a string containing the name of an attribute or a ('attribute_name', 'unit_name') pair. uround will be used to format in the second case.

    :param obj: the object to get attributes from
    :param fields: the attributes or (attribute, unit) pairs to get from obj
    :param digits: the number of digits to round to for uround
    :return: the formatted string
    """
    field_strings = []
    for field in fields:
        try:
            field_name, unit = field
            try:
                field_strings.append('{} = {} {}'.format(field_name, uround(getattr(obj, field_name), unit, digits = digits), unit))
            except TypeError:
                field_strings.append('{} = {}'.format(field_name, getattr(obj, field_name)))
        except (ValueError, TypeError):
            field_strings.append('{} = {}'.format(field, getattr(obj, field)))
    return '{}({})'.format(obj.__class__.__name__, ', '.join(field_strings))


def dict_to_arrays(dct: dict):
    """
    Return the keys and values of a dictionary as two numpy arrays, in key-sorted order.

    :param dct: the dictionary to array-ify
    :type dct: dict
    :return: (key_array, value_array)
    """
    key_list = []
    val_list = []

    for key, val in sorted(dct.items()):
        key_list.append(key)
        val_list.append(val)

    return key_value_arrays(np.array(key_list), np.array(val_list))


def get_now_str() -> str:
    """Return a formatted string with the current year-month-day_hour-minute-second."""
    return datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')


class LogManager:
    """
    A context manager to easily set up logging.

    Within a managed block, logging messages are intercepted if their highest-level logger is named in `logger_names`.
    The object returned by the LogManager ``with`` statement can be used as a logger, with name given by `manual_logger_name`.
    """

    def __init__(self,
                 *logger_names,
                 manual_logger_name: str = 'simulacra',
                 stdout_logs: bool = True,
                 stdout_level = logging.DEBUG,
                 file_logs: bool = False,
                 file_level = logging.DEBUG,
                 file_name: Optional[str] = None,
                 file_dir: Optional[str] = None,
                 file_mode: str = 'a',
                 disable_level = logging.NOTSET):
        """
        Parameters
        ----------
        logger_names
            The names of loggers to intercept.
        manual_logger_name
            The name used by the logger returned by the LogManager ``with`` statement.
        stdout_logs : :class:`bool`
            If ``True``, log messages will be displayed on stdout.
        stdout_level : :class:`bool`
        file_logs
        file_level
        file_name
        file_dir
        file_mode : :class:`str`
            the file mode to open the log file with, defaults to 'a' (append)
        disable_level
        """
        """
        Initialize a Logger context manager.

        :param logger_names: the names of loggers to catch/modify and/or create
        :param manual_logger_name: the name of the logger that will be returned by the context manager's __enter__ method
        :param stdout_logs: whether to print log messages to stdout
        :param stdout_level: the lowest level for stdout log messages
        :param file_logs: whether to print log messages to a file
        :param file_level: the lowest level for file log messages
        :param file_name: the filename for the log file, defaults to 'log__{timestamp}'. If file_name does not end with '.log', it will be appended.
        :param file_dir: the director for the log file, defaults to the current working directory
        :param file_mode: the file mode to open the log file with, defaults to 'a' (append)
        :param disable_level: log level to disable, short-circuits propagation of logs <= this level
        :return None
        """
        self.logger_names = list(logger_names)
        if manual_logger_name is not None and manual_logger_name not in self.logger_names:
            self.logger_names = [manual_logger_name] + self.logger_names

        self.stdout_logs = stdout_logs
        self.stdout_level = stdout_level

        self.file_logs = file_logs
        self.file_level = file_level

        if file_name is None:
            file_name = f'log__{get_now_str()}'
        self.file_name = file_name
        if not self.file_name.endswith('.log'):
            self.file_name += '.log'

        if file_dir is None:
            file_dir = os.getcwd()
        self.file_dir = os.path.abspath(file_dir)

        self.file_mode = file_mode

        self.disable_level = disable_level

        self.logger = None

    def __enter__(self):
        """Gets a logger with the specified name, replace it's handlers with, and returns itself."""
        logging.disable(self.disable_level)

        self.loggers = {name: logging.getLogger(name) for name in self.logger_names}

        new_handlers = [logging.NullHandler()]

        if self.stdout_logs:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(self.stdout_level)
            stdout_handler.setFormatter(LOG_FORMATTER)

            new_handlers.append(stdout_handler)

        if self.file_logs:
            log_file_path = os.path.join(self.file_dir, self.file_name)

            ensure_dir_exists(log_file_path)  # the log message emitted here will not be included in the logger being created by this context manager

            file_handler = logging.FileHandler(log_file_path, mode = self.file_mode)
            file_handler.setLevel(self.file_level)
            file_handler.setFormatter(LOG_FORMATTER)

            new_handlers.append(file_handler)

        self.old_levels = {name: logger.level for name, logger in self.loggers.items()}
        self.old_handlers = {name: logger.handlers for name, logger in self.loggers.items()}

        for logger in self.loggers.values():
            logger.setLevel(logging.DEBUG)
            logger.handlers = new_handlers

        return self.loggers[self.logger_names[0]]

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restores the logger to it's pre-context state."""
        logging.disable(logging.NOTSET)

        for name, logger in self.loggers.items():
            logger.level = self.old_levels[name]
            logger.handlers = self.old_handlers[name]


ILLEGAL_FILENAME_CHARACTERS = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']  # these characters should be stripped from file names before use


def strip_illegal_characters(string: str) -> str:
    """Strip characters that cannot be included in file names from a string."""
    return ''.join([char for char in string if char not in ILLEGAL_FILENAME_CHARACTERS])


NearestEntry = collections.namedtuple('NearestEntry', ('index', 'value', 'target'))


def find_nearest_entry(array: np.ndarray, target: Union[float, int]) -> NamedTuple:
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


def ensure_dir_exists(path):
    """
    Ensure that the directory tree to the path exists.

    Parameters
    ----------
    path
        A path to a file or directory.

    Returns
    -------
    :class:`str`
        The path that was created.
    """
    """
    

    :param path: the path to a file or directory
    :type path: str
    """
    split_path = os.path.splitext(path)
    if split_path[0] != path:  # path is file
        path_to_make = os.path.dirname(split_path[0])
    else:  # path is dir
        path_to_make = split_path[0]
    os.makedirs(path_to_make, exist_ok = True)

    logger.debug('Ensured dir {} exists'.format(path_to_make))

    return path_to_make


def downsample(dense_x_array: np.ndarray,
               sparse_x_array: np.ndarray,
               dense_y_array: np.ndarray):
    """
    Downsample (dense_x_array, dense_y_array) to (sparse_x_array, sparse_y_array).

    The downsampling is performed by matching points from sparse_x_array to dense_x_array using find_nearest_entry. Use with caution!

    Parameters
    ----------
    dense_x_array : :class:`numpy.ndarray`
        A dense array of x values.
    sparse_x_array : :class:`numpy.ndarray`
        A sparse array of x values.
    dense_y_array : :class:`numpy.ndarray`
        A dense array of y values corresponding to `dense_x_array`.

    Returns
    -------
    :class:`numpy.ndarray`
        The sparsified y array.
    """
    sparse_y_array = np.zeros(len(sparse_x_array), dtype = dense_y_array.dtype) * np.NaN

    for sparse_index, x in enumerate(sparse_x_array):
        dense_index, _, _ = find_nearest_entry(dense_x_array, x)
        sparse_y_array[sparse_index] = dense_y_array[dense_index]

    return sparse_y_array


def run_in_process(func: Callable, *args, **kwargs):
    """
    Run a function in a separate thread.

    :param func: the function to run
    :param args: positional arguments for function
    :param kwargs: keyword arguments for function
    """
    with multiprocessing.Pool(processes = 1) as pool:
        output = pool.apply(func, args, kwargs)

    return output


def find_or_init_sim(spec, search_dir: Optional[str] = None, file_extension = '.sim'):
    """
    Try to load a :class:`simulacra.Simulation` by looking for a pickled :class:`simulacra.core.Simulation` named ``{search_dir}/{spec.file_name}.{file_extension}``.
    If that fails, create a new Simulation from `spec`.

    Parameters
    ----------
    spec : :class:`simulacra.core.Specification`
    search_dir : str
    file_extension : str

    Returns
    -------
    :class:`simulacra.core.Simulation`
    """
    try:
        if search_dir is None:
            search_dir = os.getcwd()
        path = os.path.join(search_dir, spec.file_name + file_extension)
        sim = core.Simulation.load(file_path = path)
    except FileNotFoundError:
        sim = spec.to_simulation()

    return sim


def multi_map(function, targets, processes = None, **kwargs):
    """
    Map a function over a list of inputs using multiprocessing.

    Function should take a single positional argument (an element of targets) and any number of keyword arguments, which must be the same for each target.

    Parameters
    ----------
    function : a callable
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
        output = pool.map(function, targets, **kwargs)

    return tuple(output)


class cached_property:
    """
    A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property.

    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def method_dispatch(func):
    """Works the same as :func:`functools.singledispatch`, but uses the second argument instead of the first so that it can be used for instance methods."""
    dispatcher = functools.singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    functools.update_wrapper(wrapper, func)

    return wrapper


def hash_args_kwargs(*args, **kwargs):
    """Return the hash of a tuple containing the args and kwargs."""
    return hash(args + tuple(kwargs.items()))


def memoize(func: Callable):
    """Memoize a function by storing a dictionary of {inputs: outputs}."""
    memo = {}

    @functools.wraps(func)
    def memoizer(*args, **kwargs):
        key = hash_args_kwargs(*args, **kwargs)
        if key not in memo:
            memo[key] = func(*args, **kwargs)
        return memo[key]

    return memoizer


def watcher(watch):
    """
    Returns a decorator that memoizes the result of a method call until the watcher function returns a different value.

    The watcher function is passed the instance that the original method is bound to.

    :param watch: a function which is called to check whether to recompute the wrapped function
    :return: a Watcher decorator
    """

    class Watcher:
        __slots__ = ('func', 'cached', 'watched', '__doc__')

        def __init__(self, func):
            self.func = func
            self.cached = {}
            self.watched = {}

            self.__doc__ = func.__doc__

        def __str__(self):
            return 'Watcher wrapper over {}'.format(self.func.__name__)

        def __repr__(self):
            return 'watcher({})'.format(repr(self.func))

        def __call__(self, instance, *args, **kwargs):
            check = watch(instance)

            if self.watched.get(instance) != check:
                self.cached[instance] = self.func(instance, *args, **kwargs)
                self.watched[instance] = check

            return self.cached[instance]

        def __get__(self, instance, cls):
            # support instance methods
            return functools.partial(self.__call__, instance)

    return Watcher


def timed(func: Callable):
    """A decorator that times the execution of the decorated function. A log message is emitted at level ``DEBUG`` with the timing information."""

    @functools.wraps(func)
    def timed_wrapper(*args, **kwargs):
        time_start = datetime.datetime.now()
        val = func(*args, **kwargs)
        time_end = datetime.datetime.now()

        time_elapsed = time_end - time_start

        logger.debug(f'Execution of {func} took {time_elapsed}')

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


class Descriptor:
    """
    A generic descriptor that implements default descriptor methods for easy overriding in subclasses.

    The data is stored in the instance dictionary.
    """

    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]


class RestrictedValues(Descriptor):
    """
    A descriptor that forces the attribute to have a certain set of possible values.

    If the value is not in the set of legal values a ValueError is raised.
    """

    __slots__ = ('name', 'legal_values')

    def __init__(self, name, legal_values = set()):
        self.legal_values = set(legal_values)

        super().__init__(name)

    def __set__(self, instance, value):
        if value not in self.legal_values:
            raise ValueError('Expected {} to be from {}'.format(value, self.legal_values))
        else:
            super().__set__(instance, value)


class Typed(Descriptor):
    """
    A descriptor that forces the attribute to have a certain type.

    If the value does not match the provided type a TypeError is raised.
    """

    __slots__ = ('name', 'legal_type')

    def __init__(self, name, legal_type = str):
        self.legal_type = legal_type

        super().__init__(name)

    def __set__(self, instance, value):
        if not isinstance(value, self.legal_type):
            raise TypeError('Expected {} to be a {}'.format(value, self.legal_type))
        else:
            super().__set__(instance, value)


class Checked(Descriptor):
    """
    A descriptor that only allows setting with values that return True from a provided checking function.

    If the value does not pass the check a ValueError is raised.
    """

    __slots__ = ('name', 'check')

    def __init__(self, name, check = None):
        if check is None:
            check = lambda value: True
        self.check = check

        super().__init__(name)

    def __set__(self, instance, value):
        if not self.check(value):
            raise ValueError(f'Value {value} did not pass the check function {self.check} for attribute {self.name} on {instance}')
        else:
            super().__set__(instance, value)


def bytes_to_str(num: Union[float, int]) -> str:
    """Return a number of bytes as a human-readable string."""
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def get_file_size(file_path: str):
    """Return the size of the file at file_path."""
    return os.stat(file_path).st_size


def get_file_size_as_string(file_path: str) -> str:
    """Return the size of the file at file_path as a human-readable string."""
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return bytes_to_str(file_info.st_size)


def try_loop(*functions_to_run,
             wait_after_success: datetime.timedelta = datetime.timedelta(hours = 1),
             wait_after_failure: datetime.timedelta = datetime.timedelta(minutes = 1),
             begin_text: str = 'Beginning loop',
             complete_text: str = 'Completed loop'):
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
                    logger.exception(f'Exception encountered while executing loop function {f}')
                    failed = True

        logger.info(complete_text + '. Elapsed time: {}'.format(timer.wall_time_elapsed))

        if failed:
            wait = wait_after_failure
            logger.info(f'Loop cycle failed, retrying in {wait_after_failure.total_seconds()} seconds')
        else:
            wait = wait_after_success
            logger.info(f'Loop cycle succeeded, next cycle in {wait_after_success.total_seconds()} seconds')

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


class SubprocessManager:
    def __init__(self, cmd_string, **subprocess_kwargs):
        self.cmd_string = cmd_string
        self.subprocess_kwargs = subprocess_kwargs

        self.name = self.cmd_string[0]

        self.subprocess = None

    def __enter__(self):
        self.subprocess = subprocess.Popen(self.cmd_string,
                                           **self.subprocess_kwargs)

        logger.debug(f'Opened subprocess {self.name}')

        return self.subprocess

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.subprocess.communicate()
            logger.debug(f'Closed subprocess {self.name}')
        except AttributeError:
            logger.warning(f'Exception while trying to close subprocess {self.name}, possibly not closed')


def get_processes_by_name(process_name: str) -> Iterable[psutil.Process]:
    """
    Return an iterable of processes that match the given name.

    :param process_name: the name to search for
    :type process_name: str
    :return: an iterable of psutil Process instances
    """
    return [p for p in psutil.process_iter() if p.name() == process_name]


def suspend_processes(processes: Iterable[psutil.Process]):
    """
    Suspend a list of processes.

    Parameters
    ----------
    processes : iterable of psutil.Process
    """
    for p in processes:
        p.suspend()
        logger.info('Suspended {}'.format(p))


def resume_processes(processes: Iterable[psutil.Process]):
    """
    Resume a list of processes.

    Parameters
    ----------
    processes : iterable of psutil.Process
    """
    for p in processes:
        p.resume()
        logger.info('Resumed {}'.format(p))


def suspend_processes_by_name(process_name: str):
    processes = get_processes_by_name(process_name)

    suspend_processes(processes)


def resume_processes_by_name(process_name: str):
    processes = get_processes_by_name(process_name)

    resume_processes(processes)


class SuspendProcesses:
    def __init__(self, *processes):
        """

        Parameters
        ----------
        processes
            :class:`psutil.Process` objects or strings to search for using :func:`get_process_by_name`
        """
        self.processes = []
        for process in processes:
            if type(process) == str:
                self.processes += get_processes_by_name(process)
            elif type(process) == psutil.Process:
                self.processes.append(process)

    def __enter__(self):
        suspend_processes(self.processes)

    def __exit__(self, exc_type, exc_val, exc_tb):
        resume_processes(self.processes)
