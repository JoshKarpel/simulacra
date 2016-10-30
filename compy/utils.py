import datetime as dt
import functools
import logging
import multiprocessing as mp
import os
import gzip
import pickle
import sys
import uuid
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from compy.units import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


LOG_FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt = '%y/%m/%d %H:%M:%S')  # global log format specification


class Logger:
    """A context manager to easily set up logging."""

    def __init__(self, *logger_names,
                 manual_logger_name = None,
                 stdout_logs = True, stdout_level = logging.DEBUG,
                 file_logs = False, file_level = logging.DEBUG, file_name = None, file_dir = None, file_mode = 'a'):
        """
        Initialize a Logger context manager.

        :param logger_names: the names of loggers to catch/modify and/or create
        :param manual_logger_name: the name of the logger that will be returned by the context manager's __enter__ method
        :param stdout_logs: whether to print log messages to stdout
        :param stdout_level: the lowest level for stdout log messages
        :param file_logs: whether to print log messages to a file
        :param file_level: the lowest level for file log messages
        :param file_name: the filename for the log file, defaults to 'log__{timestamp}'
        :param file_dir: the director for the log file, defaults to the current working directory
        :param file_mode: the file mode to open the log file with, defaults to 'a' (append)
        """
        self.logger_names = list(logger_names)
        if manual_logger_name is not None and manual_logger_name not in self.logger_names:
            self.logger_names = [manual_logger_name] + self.logger_names

        self.stdout_logs = stdout_logs
        self.stdout_level = stdout_level

        self.file_logs = file_logs
        self.file_level = file_level

        if file_name is None:
            file_name = '{}__{}'.format('log', dt.datetime.now().strftime('%y-%m-%d_%H-%M-%S'))
        self.file_name = file_name

        if file_dir is None:
            file_dir = os.getcwd()
        self.file_dir = file_dir

        self.file_mode = file_mode

        self.logger = None

    def __enter__(self):
        """Enter special method. Gets a logger with the specified name, replace it's handlers with, and returns itself."""
        self.loggers = {name: logging.getLogger(name) for name in self.logger_names}

        new_handlers = [logging.NullHandler()]

        if self.stdout_logs:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(self.stdout_level)
            stdout_handler.setFormatter(LOG_FORMATTER)

            new_handlers.append(stdout_handler)

        if self.file_logs:
            log_file_path = os.path.join(self.file_dir, '{}.log'.format(self.file_name))

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
        """Exit special method. Restores the logger to it's pre-context state."""
        for name, logger in self.loggers.items():
            logger.level = self.old_levels[name]
            logger.handlers = self.old_handlers[name]


ILLEGAL_FILENAME_CHARACTERS = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']  # these characters should be stripped from file names before use


def strip_illegal_characters(string):
    """Strip characters that cannot be included in filenames from a string."""
    return ''.join([char for char in string if char not in ILLEGAL_FILENAME_CHARACTERS])


class Beet:
    """
    A superclass that provides an easy interface for pickling and unpickling instances.

    Two Beets compare and hash equal if they have the same Beet.uid, a uuid4 generated at initialization.
    """

    def __init__(self, name, file_name = None):
        """
        Construct a Beet with the given name and file_name.

        The file_name is automatically derived from the name if None is given.

        :param name: the internal name of the Beet
        :param file_name: the desired external name, used for pickling. Illegal characters are stripped before use.
        """
        self.name = name
        if file_name is None:
            file_name = name

        file_name_stripped = strip_illegal_characters(file_name)
        if file_name_stripped != file_name:
            logger.warning('Using file name {} instead of {} for {}'.format(file_name_stripped, file_name, self.name))
        self.file_name = file_name_stripped

        self.initialized_at = dt.datetime.now()
        self.uid = uuid.uuid4()

        logger.info('Initialized {}'.format(repr(self)))

    def __str__(self):
        return '{}: {} ({}) [{}]'.format(self.__class__.__name__, self.name, self.file_name, self.uid)

    def __repr__(self):
        return '{}(name = {}, file_name = {}, uid = {})'.format(self.__class__.__name__, self.name, self.file_name, self.uid)

    def __eq__(self, other):
        return self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)

    def copy(self):
        """Return a deepcopy of the Beet."""
        return deepcopy(self)

    def save(self, target_dir = None, file_extension = '.beet'):
        """
        Atomically pickle the Beet to {target_dir}/{self.file_name}.{file_extension}, and gzip it for reduced disk usage.

        :param target_dir: directory to save the Beet to
        :param file_extension: file extension to name the Beet with
        :return: None
        """
        if target_dir is None:
            target_dir = os.getcwd()

        file_path = os.path.join(target_dir, self.file_name + file_extension)
        file_path_working = file_path + '.working'

        ensure_dir_exists(file_path_working)

        with gzip.open(file_path_working, mode = 'wb') as file:
            pickle.dump(self, file, protocol = -1)

        os.replace(file_path_working, file_path)

        logger.info('Saved {} {} to {}'.format(self.__class__.__name__, self.name, file_path))

        return file_path

    @classmethod
    def load(cls, file_path):
        """
        Load a Beet from file_path.

        :param file_path: the path to try to load a Beet from
        :return: the loaded Beet
        """
        with gzip.open(file_path, mode = 'rb') as file:
            beet = pickle.load(file)

        logger.info('Loaded {} {} from {}'.format(beet.__class__.__name__, beet.name, file_path))

        return beet

    def info(self):
        return str(self)


def index_of_closest(array, value):
    """Returns the index of the numpy array entry closest to the given value."""
    return np.argmin(np.abs(array - value))


def ensure_dir_exists(path):
    """Ensure that the directory tree to the path exists."""
    split_path = os.path.splitext(path)
    if split_path[0] != path:  # path is file
        make_path = os.path.dirname(split_path[0])
    else:  # path is dir
        make_path = split_path[0]
    os.makedirs(make_path, exist_ok = True)

    logger.debug('Ensured path exists: {}'.format(make_path))


def save_current_figure(name, name_postfix = '', target_dir = None, img_format = 'png', img_scale = 1, transparent = True, **kwargs):
    """Save the current matplotlib figure with the given name to the given folder."""
    if target_dir is None:
        target_dir = os.getcwd()
    path = os.path.join(target_dir, '{}{}.{}'.format(name, name_postfix, img_format))

    ensure_dir_exists(path)

    plt.savefig(path, dpi = img_scale * plt.gcf().dpi, bbox_inches = 'tight', transparent = transparent)

    logger.info('Saved matplotlib figure {} to {}'.format(name, path))


def xy_plot(x, *y,
            legends = None, plot_args = None,
            x_scale = None, y_scale = None,
            title = None, x_label = None, y_label = None,
            x_center = 0, x_range = None,
            y_center = None, y_range = None,
            log_x = False, log_y = False,
            vlines = None,
            aspect_ratio = 1.5, title_size = 15, label_size = 15, unit_size = 10, legend_size = 12,
            save_csv = False,
            **kwargs):
    fig = plt.figure(figsize = (7 * aspect_ratio, 7), dpi = 600)
    fig.set_tight_layout(True)
    axis = plt.subplot(111)

    # generate scaled x data
    if x_scale is not None:
        scaled_x = x / unit_names_to_values[x_scale]
    else:
        scaled_x = x

    # generate scaled y data
    scaled_y = []
    for yy in y:
        if y_scale is not None:
            scaled_y.append(yy / unit_names_to_values[y_scale])
        else:
            scaled_y.append(yy)

    # TODO: figure out plot_args

    # plot y vs. x data
    for ii, yy in enumerate(scaled_y):
        if legends is not None:
            plt.plot(scaled_x, yy, label = legends[ii])
        else:
            plt.plot(scaled_x, yy)

    # set title
    if title is not None:
        title = axis.set_title(r'{}'.format(title), fontsize = title_size)
        title.set_y(1.025)

    # set x label
    if x_label is not None:
        if x_scale is not None:
            x_label += r' ({})'.format(unit_names_to_tex_strings[x_scale])

        axis.set_xlabel(r'{}'.format(x_label), fontsize = label_size)

    # set y label
    if y_label is not None:
        if y_scale is not None:
            y_label += r' ({})'.format(unit_names_to_tex_strings[y_scale])

        axis.set_ylabel(r'{}'.format(y_label), fontsize = label_size)

    # set x axis limits
    if x_range is None:
        lower_limit_x = np.min(scaled_x)
        upper_limit_x = np.max(scaled_x)
    else:
        lower_limit_x = (x_center - x_range) / unit_names_to_values[x_scale]
        upper_limit_x = (x_center + x_range) / unit_names_to_values[x_scale]

    axis.set_xlim(lower_limit_x, upper_limit_x)

    if y_center is not None and y_range is not None:
        y_lower = (y_center - y_range)  # TODO: units
        y_upper = (y_center + y_range)
        axis.set_ylim(y_lower, y_upper)

    # set whether axes are log scale
    if log_x:
        axis.set_xscale('log')
    if log_y:
        axis.set_yscale('log')

    if vlines is not None:
        for x in vlines:
            plt.axvline(x / unit_names_to_values[x_scale], color = 'red', linestyle = ':')

    # grid and tick options
    axis.grid(True, color = 'gray', linestyle = ':', alpha = 0.9)
    axis.tick_params(axis = 'both', which = 'major', labelsize = unit_size)

    # draw legend
    if legends is not None:
        axis.legend(loc = 'best', fontsize = legend_size)

    save_current_figure(**kwargs)

    if save_csv:
        path = os.path.join(kwargs['target_dir'], '{}.csv'.format(kwargs['name']))
        np.savetxt(path, (x, *y), delimiter = ',', newline = '\n')

        logger.info('Saved figure data from {} to {}'.format(kwargs['name'], path))

    plt.close()


def multi_map(function, targets, processes = None):
    """Map a function over a list of inputs using multiprocessing."""
    if processes is None:
        processes = mp.cpu_count() - 1

    with mp.Pool(processes = processes) as pool:
        output = pool.map(function, targets)

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
    dispatcher = functools.singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    functools.update_wrapper(wrapper, func)

    return wrapper


def memoize(copy_output = False):
    """
    Returns a decorator that memoizes the result of a function call.

    :param copy_output: if True, the output of the memo will be deepcopied before returning. Defaults to False.
    :return: a Memoize decorator
    """

    # TODO: add @functools.wraps

    class Memoize:

        __slots__ = ('func', 'memo', '__doc__')

        def __init__(self, func):
            self.func = func
            self.memo = {}

            self.__doc__ = func.__doc__

        def __str__(self):
            return 'Memoized wrapper over {}'.format(self.func.__name__)

        def __repr__(self):
            return 'memoize(copy_output = {})({})'.format(copy_output, repr(self.func))

        def __call__(self, *args, **kwargs):
            key = args
            for k, v in kwargs.items():
                try:
                    key += (k, tuple(v))
                except TypeError:
                    key += (k, v)

            try:
                value = self.memo[key]
                # logger.debug('Hit on memo for {}, key = {}'.format(repr(self.func), key))
            except (KeyError, TypeError):
                value = self.func(*args, **kwargs)
                self.memo[key] = value
                # logger.debug('Miss on memo for {}, key = {}'.format(repr(self.func), key))

            if copy_output:
                try:
                    value = value.copy()
                except AttributeError:
                    value = deepcopy(value)

            return value

        def __get__(self, obj, objtype):
            # support instance methods
            return functools.partial(self.__call__, obj)

    return Memoize


class Timer:
    """A context manager that times the code in the with block. Print the Timer to see the results."""

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
