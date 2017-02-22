import datetime as dt
import functools as ft
import itertools as it
import logging
import multiprocessing as mp
import collections
import os
import gzip
import pickle
import sys
import uuid
from copy import deepcopy
import time
import psutil

import matplotlib.pyplot as plt
import numpy as np

from .units import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

LOG_FORMATTER = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt = '%y/%m/%d %H:%M:%S')  # global log format specification


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
            file_name = '{}__{}'.format('log', now_string())
        self.file_name = file_name
        if not self.file_name.endswith('.log'):
            self.file_name += '.log'

        if file_dir is None:
            file_dir = os.getcwd()
        self.file_dir = file_dir

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
        self.name = str(name)
        if file_name is None:
            file_name = self.name

        file_name_stripped = strip_illegal_characters(str(file_name))
        if file_name_stripped != file_name:
            logger.warning('Using file name {} instead of {} for {}'.format(file_name_stripped, file_name, self.name))
        self.file_name = file_name_stripped

        self.initialized_at = dt.datetime.now()
        self.uid = uuid.uuid4()

        logger.info('Initialized {}'.format(repr(self)))

    def __str__(self):
        return '{}: {} ({}) [{}]'.format(self.__class__.__name__, self.name, self.file_name, self.uid)

    def __repr__(self):
        return field_str(self, 'name', 'file_name', 'uid')

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.uid == other.uid

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
        :return: the path to the saved Beet
        """
        if target_dir is None:
            target_dir = os.getcwd()

        file_path = os.path.join(target_dir, self.file_name + file_extension)
        file_path_working = file_path + '.working'

        ensure_dir_exists(file_path_working)

        with gzip.open(file_path_working, mode = 'wb') as file:
            pickle.dump(self, file, protocol = -1)

        os.replace(file_path_working, file_path)

        logger.debug('Saved {} {} to {}'.format(self.__class__.__name__, self.name, file_path))

        return file_path

    @classmethod
    def load(cls, file_path):
        """
        Load a Beet from file_path.

        :param file_path: the path to load a Beet from
        :return: the loaded Beet
        """
        with gzip.open(file_path, mode = 'rb') as file:
            beet = pickle.load(file)

        logger.debug('Loaded {} {} from {}'.format(beet.__class__.__name__, beet.name, file_path))

        return beet

    def info(self):
        return str(self)


NearestEntry = collections.namedtuple('NearestEntry', ('index', 'value', 'target'))


def find_nearest_entry(ndarray, target):
    """Returns the (index, value, target) of the numpy array entry closest to the given target."""
    index = np.argmin(np.abs(ndarray - target))
    value = ndarray[index]

    return NearestEntry(index, value, target)


def ensure_dir_exists(path):
    """Ensure that the directory tree to the path exists."""
    split_path = os.path.splitext(path)
    if split_path[0] != path:  # path is file
        make_path = os.path.dirname(split_path[0])
    else:  # path is dir
        make_path = split_path[0]
    os.makedirs(make_path, exist_ok = True)

    logger.debug('Ensured dir {} exists'.format(make_path))


def save_current_figure(name, name_postfix = '', target_dir = None, img_format = 'pdf', img_scale = 1, transparent = True, colormap = plt.cm.inferno, **kwargs):
    """Save the current matplotlib figure with the given name to the given folder."""
    plt.set_cmap(colormap)

    if target_dir is None:
        target_dir = os.getcwd()
    path = os.path.join(target_dir, '{}{}.{}'.format(name, name_postfix, img_format))

    ensure_dir_exists(path)

    plt.savefig(path, dpi = img_scale * plt.gcf().dpi, bbox_inches = 'tight', transparent = transparent)

    logger.debug('Saved matplotlib figure {} to {}'.format(name, path))

    return path


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


XYAxis = collections.namedtuple('XYAxis', field_names = ('axis', 'lines', 'title', 'x_label', 'y_label', 'legend'))


def make_xy_axis(axis,
                 x_data, *y_data,
                 line_labels = (), line_kwargs = (),
                 x_scale = 1, y_scale = 1,
                 x_log_axis = False, y_log_axis = False,
                 x_lower_limit = None, x_upper_limit = None, y_lower_limit = None, y_upper_limit = None,
                 vlines = (), vline_kwargs = (), hlines = (), hline_kwargs = (),
                 title = None, x_label = None, y_label = None,
                 font_size_title = 15, font_size_axis_labels = 15, font_size_tick_labels = 10, font_size_legend = 12,
                 ticks_on_top = True, ticks_on_right = True, legend_on_right = False,
                 **kwargs):
    """
    Turn a matplotlib axis object into a basic x-y line plot.

    :param axis: the axis to perform operations on
    :param x_data: a single array to plot the y data against
    :param y_data: any number of arrays of the same length as x_data to plot
    :param line_labels: the labels for the lines
    :param line_kwargs: keyword arguments for each line's .plot() call (None for default)
    :param x_scale: a number to scale the x_data by. Can be a string corresponding to a key in the unit name/value dict.
    :param y_scale: a number to scale the y_data by. Can be a string corresponding to a key in the unit name/value dict.
    :param x_log_axis: if True, log the x axis
    :param y_log_axis: if True, log the y axis
    :param x_lower_limit: lower limit for the x axis, defaults to np.nanmin(x_data)
    :param x_upper_limit: upper limit for the x axis, defaults to np.nanmax(x_data)
    :param y_lower_limit: lower limit for the y axis, defaults to min(np.nanmin(y_data))
    :param y_upper_limit: upper limit for the y axis, defaults to min(np.nanmin(y_data))
    :param vlines: a list of x positions to place vertical lines
    :param vline_kwargs: a list of kwargs for each vline (None for default)
    :param hlines: a list of y positions to place horizontal lines
    :param hline_kwargs: a list of kwargs for each hline (None for default)
    :param title: a title for the plot
    :param x_label: a label for the x axis
    :param y_label: a label for the y axis
    :param font_size_title: font size for the title
    :param font_size_axis_labels: font size for the axis labels
    :param font_size_tick_labels: font size for the tick labels
    :param font_size_legend: font size for the legend
    :param kwargs: absorbs kwargs
    :return:
    """
    # ensure data is in numpy arrays
    x_data = np.array(x_data)
    y_data = [np.array(y) for y in y_data]
    line_labels = tuple(line_labels)
    line_kwargs = tuple(line_kwargs)

    # determine if scale_x/y is a unit specifier or a number and set scale and labels accordingly
    if type(x_scale) == str:
        scale_x_label = r' ({})'.format(unit_names_to_tex_strings[x_scale])
        x_scale = unit_names_to_values[x_scale]
    else:
        scale_x_label = r''
    if type(y_scale) == str:
        scale_y_label = r' ({})'.format(unit_names_to_tex_strings[y_scale])
        y_scale = unit_names_to_values[y_scale]
    else:
        scale_y_label = r''

    # zip together each set of y data with its plotting options
    lines = []
    for y, lab, kw in it.zip_longest(y_data, line_labels, line_kwargs):
        if kw is None:  # means there are no kwargs for this y data
            kw = {}
        lines.append(plt.plot(x_data / x_scale, y / y_scale, label = lab, **kw)[0])

    # make any horizontal and vertical lines
    for vl, vkw in it.zip_longest(vlines, vline_kwargs):
        if vkw is None:
            vkw = {}
        kw = {'color': 'black', 'linestyle': '-'}
        kw.update(vkw)
        axis.axvline(x = vl / x_scale, **kw)
    for hl, hkw in it.zip_longest(hlines, hline_kwargs):
        if hkw is None:
            hkw = {}
        kw = {'color': 'black', 'linestyle': '-'}
        kw.update(hkw)
        axis.axhline(y = hl / y_scale, **kw)

    if x_log_axis:
        axis.set_xscale('log')
    if y_log_axis:
        axis.set_yscale('log')

    # set axis limits independently
    if x_lower_limit is None:
        x_lower_limit = np.nanmin(x_data)
    if x_upper_limit is None:
        x_upper_limit = np.nanmax(x_data)
    if y_lower_limit is None:
        y_lower_limit = min([np.nanmin(y) for y in y_data])
    if y_upper_limit is None:
        y_upper_limit = max([np.nanmax(y) for y in y_data])
    axis.set_xlim(left = x_lower_limit / x_scale, right = x_upper_limit / x_scale)
    axis.set_ylim(bottom = y_lower_limit / y_scale, top = y_upper_limit / y_scale)

    # set up a grid of gray dotted lines
    axis.grid(True, color = 'gray', linestyle = ':')
    axis.tick_params(axis = 'both', which = 'major', labelsize = font_size_tick_labels)

    axis.tick_params(labeltop = ticks_on_top)
    axis.tick_params(labelright = ticks_on_right)

    # make title, axis labels, and legend
    _title, _x_label, _y_label, _legend = None, None, None, None
    if title is not None:
        _title = axis.set_title(r'{}'.format(title), fontsize = font_size_title)
        _title.set_y(1.05)  # move title up a little
    if x_label is not None:
        _x_label = axis.set_xlabel(r'{}'.format(x_label) + scale_x_label, fontsize = font_size_axis_labels)
    if y_label is not None:
        _y_label = axis.set_ylabel(r'{}'.format(y_label) + scale_y_label, fontsize = font_size_axis_labels)
    if len(line_labels) > 0:
        if not legend_on_right:
            _legend = axis.legend(loc = 'best', fontsize = font_size_legend)
        if legend_on_right:
            _legend = axis.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left', borderaxespad = 0., fontsize = font_size_legend, ncol = 1 + (len(line_labels) // 17))

    return XYAxis(axis = axis, lines = lines, title = _title, x_label = _x_label, y_label = _y_label, legend = _legend)


def xy_plot(name, x_data, *y_data,
            aspect_ratio = 1.5,
            save_csv = False,
            **kwargs):
    """
    A wrapper over make_xy_axis that also saves the resulting plot.

    :param name: filename for the plot
    :param x_data: a single array to plot the y data against
    :param y_data: any number of arrays of the same length as x_data to plot
    :param aspect_ratio: aspect ratio for the plot, >1 for a wider plot
    :param save_csv: if True, save x_data and y_data to a CSV file
    :param kwargs: kwargs to be passed to make_xy_axis() and save_current_figure()
    :return: the path the plot was saved to
    """
    # set up figure and axis
    fig = plt.figure(figsize = (7 * aspect_ratio, 7), dpi = 600)
    axis = plt.subplot(111)
    make_xy_axis(axis, x_data, *y_data, **kwargs)

    path = save_current_figure(name, **kwargs)

    if save_csv:
        csv_path = os.path.splitext(path)[0] + '.csv'
        np.savetxt(csv_path, (x_data, *y_data), delimiter = ',')

        logger.debug('Saved figure data from {} to {}'.format(name, csv_path))

    plt.close()

    return path


def multi_map(function, targets, processes = None, **kwargs):
    """Map a function over a list of inputs using multiprocessing."""
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
    """Works the same as ft.singledispatch, but uses the second argument instead of the first so that it can be used for instance methods."""
    dispatcher = ft.singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    ft.update_wrapper(wrapper, func)

    return wrapper


def hash_args_kwargs(*args, **kwargs):
    return hash(args + tuple(kwargs.items()))

def memoize(function):
    """Memoize a function by storing a dictionary of {inputs: outputs}."""
    memo = {}

    @ft.wraps(function)
    def memoizer(*args, **kwargs):
        key = hash_args_kwargs(*args, **kwargs)
        if key not in memo:
            memo[key] = function(*args, **kwargs)
        return memo[key]

    return memoizer


def watcher(watcher):
    """
    Returns a decorator that memoizes the result of a method call until the watcher function returns a different value.

    The watcher function is passed the instance that the original method is bound to.

    :param watcher: a function which is called to check whether to recompute the wrapped function
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
            check = watcher(instance)

            if self.watched.get(instance) != check:
                self.cached[instance] = self.func(instance, *args, **kwargs)
                self.watched[instance] = check

            return self.cached[instance]

        def __get__(self, instance, cls):
            # support instance methods
            return ft.partial(self.__call__, instance)

    return Watcher


class Timer:
    """A context manager that times the code in the with block. Print the Timer after exiting the block to see the results."""

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


def convert_bytes(num):
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
        return convert_bytes(file_info.st_size)


def try_loop(*functions_to_run,
             wait_after_success = dt.timedelta(hours = 1), wait_after_failure = dt.timedelta(minutes = 1),
             begin_text = 'Beginning loop', complete_text = 'Completed loop'):
    """
    Run the given functions in a constant loop.

    :param functions_to_run: call these functions in order during each loop
    :param wait_after_success: a datetime.timedelta object specifying how long to wait after a loop completes
    :param wait_after_failure: a datetime.timedelta object specifying how long to wait after a loop fails (i.e., raises an exception)
    :param begin_text: a string to print at the beginning of each loop
    :param complete_text: a string to print at the end of each loop
    :return:
    """
    while True:
        try:
            logger.info(begin_text)

            with Timer() as timer:
                for f in functions_to_run:
                    f()

            logger.info(complete_text + '. Elapsed time: {}'.format(timer.time_elapsed))

            logger.info('Next loop cycle at {}'.format(dt.datetime.now() + wait_after_success))

            time.sleep(wait_after_success.total_seconds())
        except Exception as e:
            logger.exception('Exception encountered')
            logger.warning('Loop cycle failed, retrying in {} seconds'.format(wait_after_failure.total_seconds()))

        time.sleep(wait_after_failure.total_seconds())


def grouper(iterable, n, fillvalue = None):
    """
    Collect data from iterable into fixed-length chunks or blocks of length n

    See https://docs.python.org/3/library/itertools.html#itertools-recipes

    :param iterable:
    :param n:
    :param fillvalue:
    :return:
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return it.zip_longest(*args, fillvalue = fillvalue)


def get_process_by_name(process_name):
    for proc in psutil.process_iter():
        if proc.name() == process_name:
            return proc

    raise ProcessLookupError('No process with name "{}" found'.format(process_name))


def figsize(scale, fig_width_pts = 498.66258, aspect_ratio = (np.sqrt(5.0) - 1.0) / 2.0):
    """
    Helper function for get_figure

    :param scale:
    :param fig_width_pts: get this from LaTeX using \the\textwidth
    :param aspect_ratio: height = width * ratio, defaults to golden ratio
    :return:
    """
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch

    fig_width = fig_width_pts * inches_per_pt * scale  # width in inches
    fig_height = fig_width * aspect_ratio  # height in inches
    fig_size = [fig_width, fig_height]

    return fig_size


def get_figure(scale = 0.9, fig_width_pts = 498.66258, aspect_ratio = (np.sqrt(5.0) - 1.0) / 2.0):
    """
    Get a matplotlib figure object with the desired scale relative to a full-text-width LaTeX page.

    scale = 'full' -> scale = 0.95
    scale = 'half' -> scale = 0.475

    :param scale: width of figure in LaTeX \textwidths
    :param fig_width_pts: get this from LaTeX using \the\textwidth
    :param aspect_ratio: height = width * ratio, defaults to golden ratio
    :return:
    """
    if scale == 'full':
        scale = 0.95
    elif scale == 'half':
        scale = .475

    fig = plt.figure(figsize = figsize(scale, fig_width_pts = fig_width_pts, aspect_ratio = aspect_ratio))

    return fig
