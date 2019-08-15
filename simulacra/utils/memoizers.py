import functools
import logging
from typing import Optional, Union, NamedTuple, Callable, Iterable

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class cached_property:
    """
    A property that is only computed once per instance and then replaces itself with an ordinary attribute.
    Deleting the attribute resets the property.

    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    """

    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def hash_args_kwargs(*args, **kwargs):
    """Return the hash of a tuple containing the args and kwargs."""
    try:
        key = hash(args + tuple(kwargs.items()))
    except TypeError:  # unhashable type, see if we've got numpy arrays
        key = hash(
            tuple(tuple(a) if type(a) == np.ndarray else a for a in args)
            + tuple(kwargs.items())
        )

    return key


def memoize(func: Callable):
    """Memoize a function by storing a dictionary of {inputs: outputs}."""
    memo = {}

    @functools.wraps(func)
    def memoizer(*args, **kwargs):
        key = hash_args_kwargs(*args, **kwargs)
        try:
            v = memo[key]
        except KeyError:
            v = memo[key] = func(*args, **kwargs)

        return v

    return memoizer


def watched_memoize(watcher):
    """
    A decorator that memoizes the result of a method call until the watcher function returns a different value.

    The watcher function is passed the instance that the original method is bound to.

    Parameters
    ----------
    watcher
    """

    class Watcher:
        __slots__ = ("func", "memo", "watched_value", "__doc__")

        def __init__(self, func):
            self.func = func
            self.memo = {}
            self.watched_value = object()

            self.__doc__ = func.__doc__

        def __call__(self, *args, **kwargs):
            check = watcher(args[0])  # args[0] is the instance
            key = hash_args_kwargs(*args, **kwargs)

            if self.watched_value != check:
                self.memo = {}
                self.watched_value = check

            try:
                v = self.memo[key]
            except KeyError:
                v = self.memo[key] = self.func(*args, **kwargs)

            return v

        def __get__(self, instance, cls):
            # support instance methods
            return functools.partial(self.__call__, instance)

    return Watcher
