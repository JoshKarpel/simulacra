import os
import sys
import pickle
import gzip
import uuid
import logging
import datetime as dt
import functools
import multiprocessing as mp
from copy import deepcopy


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


LOG_FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt = '%y/%m/%d %H:%M:%S')  # global log format specification

ILLEGAL_FILENAME_CHARACTERS = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']  # these characters should be stripped from file names before use


def strip_illegal_characters(string):
    return ''.join([char for char in string if char not in ILLEGAL_FILENAME_CHARACTERS])


class Beet:
    """A superclass for anything that should be pickleable."""

    def __init__(self, name, file_name = None):
        """

        :param name:
        :param file_name:
        """
        self.name = name
        if file_name is None:
            file_name = strip_illegal_characters(name)
        self.file_name = file_name

        self.initialized_at = dt.datetime.now()
        self.uid = uuid.uuid4()

    def __str__(self):
        return '{}: {} ({}) [{}]'.format(self.__class__, self.name, self.file_name, self.uid)

    def __repr__(self):
        return '{}(name = {}, file_name = {}, uid = {})'.format(self.__class__, self.name, self.file_name, self.uid)

    def copy(self):
        return deepcopy(self)

    def save(self, target_dir = None, file_extension = '.beet'):
        """
        Atomically save the Beet to {target_dir}/{self.file_name}.{file_extension}, and gzip it for reduced disk usage.

        :param target_dir:
        :param file_extension:
        :return:
        """
        if target_dir is None:
            target_dir = os.getcwd()

        file_path = os.path.join(target_dir, self.file_name + file_extension)
        file_path_working = file_path + '.working'

        # ensure dir exists

        with gzip.open(file_path_working, mode = 'wb') as file:
            pickle.dump(self, file, protocol = -1)

        os.replace(file_path_working, file_path)

        logger.debug('Saved {} to {}'.format(self.file_name, file_path))

    @staticmethod
    def load(file_path):
        """
        Load a Beet from file_path.

        :param file_path:
        :return:
        """
        with gzip.open(file_path, mode = 'rb') as file:
            beet = Beet.load(file)

        logger.info('Loaded {} from {}'.format(beet.name, file_path))

        return beet


def ensure_dir_exists(path):
    split_path = os.path.splitext(path)
    if split_path[0] != path:  # path is file
        make_path = os.path.dirname(split_path[0])
    else:  # path is dir
        make_path = split_path[0]
    os.makedirs(make_path, exist_ok = True)


def ask_for_input(question, default = None, cast_to = str):
    input_str = input(question + ' [Default: {}]: '.format(default))

    trimmed = input_str.replace(' ', '')
    if trimmed == '':
        return cast_to(default)
    else:
        return cast_to(trimmed)


def multi_map(function, targets, processes = None):
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


def memoize(copy_output = False):
    class Memoize:
        def __init__(self, func):
            self.func = func
            self.memo = {}

            self.__doc__ = self.func.__doc__

        def __call__(self, *args):
            # memoized call to the wrapped function
            if args in self.memo:
                out = self.memo[args]
            else:
                value = self.func(*args)
                self.memo[args] = value
                out = value

            if copy_output:
                try:
                    out = out.copy()
                except AttributeError:
                    out = deepcopy(out)

            return out

        def __get__(self, obj, objtype):
            # support instance methods
            return functools.partial(self.__call__, obj)

    return Memoize
