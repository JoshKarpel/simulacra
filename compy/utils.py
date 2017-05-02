import collections
import datetime as dt
import functools as ft
import itertools as it
import multiprocessing as mp
import subprocess
import os
import sys
import time

import numpy as np
import psutil

import logging
from .units import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

LOG_FORMATTER = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt = '%y/%m/%d %H:%M:%S')  # global log format specification

key_value_arrays = collections.namedtuple('key_value_arrays', ('key_array', 'value_array'))


def dict_to_arrays(d):
    """
    Return the keys and values of a dictionary as two numpy arrays, in key-sorted order.
    
    :param d: the dictionary to array-ify
    :type d: dict
    :return: (key_array, value_array)
    """
    key_list = []
    val_list = []

    for key, val in sorted(d.items()):
        key_list.append(key)
        val_list.append(val)

    return key_value_arrays(np.array(key_list), np.array(val_list))


def field_str(obj, *fields, digits = 3):
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
            field_name, unit_name = field
            try:
                field_strings.append('{} = {} {}'.format(field_name, uround(getattr(obj, field_name), unit_names_to_values[unit_name], digits = digits), unit_name))
            except TypeError:
                field_strings.append('{} = {}'.format(field_name, getattr(obj, field_name)))
        except (ValueError, TypeError):
            field_strings.append('{} = {}'.format(field, getattr(obj, field)))
    return '{}({})'.format(obj.__class__.__name__, ', '.join(field_strings))


def now_string():
    """Return a formatted string with the current year-month-day_hour-minute-second."""
    return dt.datetime.now().strftime('%y-%m-%d_%H-%M-%S')


class Logger:
    """A context manager to easily set up logging."""

    def __init__(self, *logger_names,
                 manual_logger_name = 'compy',
                 stdout_logs = True, stdout_level = logging.DEBUG,
                 file_logs = False, file_level = logging.DEBUG, file_name = None, file_dir = None, file_mode = 'a',
                 disable_level = logging.NOTSET):
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
            file_name = f'log__{now_string()}'
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


def strip_illegal_characters(string):
    """Strip characters that cannot be included in filenames from a string."""
    return ''.join([char for char in string if char not in ILLEGAL_FILENAME_CHARACTERS])


NearestEntry = collections.namedtuple('NearestEntry', ('index', 'value', 'target'))


def find_nearest_entry(array, target):
    """
    Returns the :code:`(index, value, target)` of the :code:`array` entry closest to the given :code:`target`.
    
    :param array: the array to look for :code:`target` in
    :param target: the target value
    :returns: a tuple containing the index of the closest value, the value of the closest value, and the original target value
    """
    array = np.array(array)  # turn the array into a numpy array

    index = np.argmin(np.abs(array - target))
    value = array[index]

    return NearestEntry(index, value, target)


def ensure_dir_exists(path):
    """
    Ensure that the directory tree to the path exists.
    
    :param path: the path to a file or directory
    :type path: str
    """
    split_path = os.path.splitext(path)
    if split_path[0] != path:  # path is file
        make_path = os.path.dirname(split_path[0])
    else:  # path is dir
        make_path = split_path[0]
    os.makedirs(make_path, exist_ok = True)

    logger.debug('Ensured dir {} exists'.format(make_path))


def downsample(dense_x_array, sparse_x_array, dense_y_array):
    """
    Downsample (dense_x_array, dense_y_array) to (sparse_x_array, sparse_y_array).

    The downsampling is performed by matching points from sparse_x_array to dense_x_array using find_nearest_entry. Use with caution!

    :param dense_x_array: 
    :param sparse_x_array:
    :param dense_y_array:
    :return: a sparsified version of dense_y_array
    """

    sparse_y_array = np.zeros(len(sparse_x_array), dtype = dense_y_array.dtype) * np.NaN

    for sparse_index, x in enumerate(sparse_x_array):
        dense_index, _, _ = find_nearest_entry(dense_x_array, x)
        sparse_y_array[sparse_index] = dense_y_array[dense_index]

    return sparse_y_array


def run_in_process(func, args = (), kwargs = None, name = None):
    """
    Run a function in a separate thread.
    
    :param func: the function to run
    :param args: positional arguments for function
    :param kwargs: keyword arguments for function
    :param name: a name for the process
    """
    if kwargs is None:
        kwargs = {}

    with mp.Pool(processes = 1) as pool:
        output = pool.apply(func, args, kwargs)

    return output


def multi_map(function, targets, processes = None, **kwargs):
    """
    Map a function over a list of inputs using multiprocessing.
    
    Function should take a single positional argument (an element of targets) and any number of keyword arguments, which must be the same for each target.
    
    :param function: the function to call on each of the targets
    :param targets: an list of arguments to call function on
    :param processes: the number of simultaneous processes to use
    :param kwargs: keyword arguments for the function
    :return: the list of outputs from the function being applied to the targets
    """
    if processes is None:
        processes = mp.cpu_count()

    with mp.Pool(processes = processes) as pool:
        output = pool.map(function, targets, **kwargs)

    return output


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
    """Works the same as functools.singledispatch, but uses the second argument instead of the first so that it can be used for instance methods."""
    dispatcher = ft.singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    ft.update_wrapper(wrapper, func)

    return wrapper


def hash_args_kwargs(*args, **kwargs):
    """Return the hash of a tuple containing the args and kwargs."""
    return hash(args + tuple(kwargs.items()))


def memoize(func):
    """Memoize a function by storing a dictionary of {inputs: outputs}."""
    memo = {}

    @ft.wraps(func)
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
            return ft.partial(self.__call__, instance)

    return Watcher


def timed(func):
    """A decorator that times the execution of the decorated function. A log message is emitted at level DEBUG, and the function output is replaced with a tuple (output, time_elapsed)."""

    @ft.wraps(func)
    def timed_wrapper(*args, **kwargs):
        time_start = dt.datetime.now()
        val = func(*args, **kwargs)
        time_end = dt.datetime.now()

        time_elapsed = time_end - time_start

        logger.debug(f'Execution of {func} took {time_elapsed}')

        return val, time_elapsed

    return timed_wrapper


class BlockTimer:
    """A context manager that times the code in the with block. Print the BlockTimer after exiting the block to see the results."""

    __slots__ = ('time_start', 'time_end', 'time_elapsed')

    def __init__(self):
        self.time_start = None
        self.time_end = None
        self.time_elapsed = None

    def __enter__(self):
        self.time_start = dt.datetime.now()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time_end = dt.datetime.now()
        self.time_elapsed = self.time_end - self.time_start

    def __str__(self):
        if self.time_end is None:
            return 'Timer started at {}, still running'.format(self.time_start)
        else:
            return 'Timer started at {}, ended at {}, elapsed time {}'.format(self.time_start, self.time_end, self.time_elapsed)


class Descriptor:
    """
    A generic descriptor that implements default descriptor methods for easy overriding in subclasses.

    The data is stored in the instance dictionary.
    """

    __slots__ = ['name']

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

    __slots__ = ['name', 'legal_values']

    def __init__(self, name, legal_values = set()):
        self.legal_values = set(legal_values)

        super(RestrictedValues, self).__init__(name)

    def __set__(self, instance, value):
        if value not in self.legal_values:
            raise ValueError('Expected {} to be from {}'.format(value, self.legal_values))
        else:
            super(RestrictedValues, self).__set__(instance, value)


class Typed(Descriptor):
    """
    A descriptor that forces the attribute to have a certain type.

    If the value does not match the provided type a TypeError is raised.
    """

    __slots__ = ['name', 'legal_type']

    def __init__(self, name, legal_type = str):
        self.legal_type = legal_type

        super(Typed, self).__init__(name)

    def __set__(self, instance, value):
        if not isinstance(value, self.legal_type):
            raise TypeError('Expected {} to be a {}'.format(value, self.legal_type))
        else:
            super(Typed, self).__set__(instance, value)


class Checked(Descriptor):
    """
    A descriptor that only allows setting with values that return True from a provided checking function.

    If the value does not pass the check a ValueError is raised.
    """

    __slots__ = ['name', 'check']

    def __init__(self, name, check = lambda: True):
        self.check = check

        super(Checked, self).__init__(name)

    def __set__(self, instance, value):
        if not self.check(value):
            raise ValueError('Value {} did not pass the check'.format(value))
        else:
            super(Checked, self).__set__(instance, value)


def bytes_to_str(num):
    """Return a number of bytes as a human-readable string."""
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def get_file_size(file_path):
    """Return the size of the file at file_path."""
    return os.stat(file_path).st_size


def get_file_size_as_string(file_path):
    """Return the size of the file at file_path as a human-readable string."""
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return bytes_to_str(file_info.st_size)


def try_loop(*functions_to_run,
             wait_after_success = dt.timedelta(hours = 1), wait_after_failure = dt.timedelta(minutes = 1),
             begin_text = 'Beginning loop', complete_text = 'Completed loop'):
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

        logger.info(complete_text + '. Elapsed time: {}'.format(timer.time_elapsed))

        if failed:
            wait = wait_after_failure
            logger.info(f'Loop cycle failed, retrying in {wait_after_failure.total_seconds()} seconds')
        else:
            wait = wait_after_success
            logger.info(f'Loop cycle succeeded, next cycle in {wait_after_success.total_seconds()} seconds')

        time.sleep(wait.total_seconds())


def grouper(iterable, n, fill_value = None):
    """
    Collect data from iterable into fixed-length chunks or blocks of length n

    See https://docs.python.org/3/library/itertools.html#itertools-recipes

    :param iterable: an iterable to chunk
    :param n: the size of the chunks
    :param fill_value: a value to fill with when iterable has run out of values, but the last chunk isn't full
    :return:
    """
    args = [iter(iterable)] * n
    return it.zip_longest(*args, fillvalue = fill_value)


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


def get_processes_by_name(process_name):
    """
    Return an iterable of processes that match the given name.

    :param process_name: the name to search for
    :type process_name: str
    :return: an iterable of psutil Process instances
    """
    return [p for p in psutil.process_iter() if p.name() == process_name]


def suspend_processes(processes):
    for p in processes:
        p.suspend()
        logger.info('Suspended {}'.format(p))


def resume_processes(processes):
    for p in processes:
        p.resume()
        logger.info('Resumed {}'.format(p))


def suspend_processes_by_name(process_name):
    processes = get_processes_by_name(process_name)

    suspend_processes(processes)


def resume_processes_by_name(process_name):
    processes = get_processes_by_name(process_name)

    resume_processes(processes)


class SuspendProcesses:
    def __init__(self, *processes):
        """
        
        :param processes: psutil.Process objects or strings to search for using get_process_by_name
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
