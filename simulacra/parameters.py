import logging
from typing import (
    Any,
    Optional,
    Union,
    List,
    Collection,
    Dict,
    Tuple,
    Callable,
)

import itertools
from copy import deepcopy
import textwrap

# these imports need to be here so that ask_for_eval works
import numpy as np
import scipy as sp
from simulacra import units as u

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Parameter:
    """A class that represents a parameter of a :class:`simulacra.Specification`."""

    def __init__(
        self,
        name: str,
        value: Union[Any, Collection[Any]] = None,
        expandable: bool = False,
    ):
        """
        Parameters
        ----------
        name
            The name of the Parameter, which should match a keyword argument of
            the target :class:`simulacra.Specification`.
        value
            The value of the Parameter, or an iterable of values.
        expandable
            If ``True``, :func:`expand_parameters` will expand along an iterable
            `value`.
        """
        self.name = name
        self.value = value
        self.expandable = expandable

    def __repr__(self):
        if not self.expandable:
            return (
                f"{self.__class__.__name__}(name = {self.name}, value = {self.value})"
            )
        else:
            header = f"{self.__class__.__name__}(name = {self.name}, expandable = {self.expandable}, values ="
            val_text = textwrap.wrap(", ".join(repr(v) for v in self.value), width=60)
            out = "\n    ".join([header] + val_text) + ")"

            return out


def expand_parameters(parameters: Collection[Parameter]) -> List[Dict[str, Any]]:
    """
    Expand an iterable of :class:`Parameter` to a list of dictionaries
    containing all of the combinations of parameter values.
    Each of these dictionaries can then be unpacked into a :class:`Specification`.

    If a :class:`Parameter` has ``expandable = True``, it will be expanded
    across the values in the outermost iterable in that :class:`Parameter`'s
    ``value``.

    Parameters
    ----------
    parameters
        The parameters to expand over.

    Returns
    -------
    expanded_parameters
        An list of dictionaries containing all of the combinations of parameters.
    """
    expanded: List[Dict[str, Any]] = [{}]

    for par in parameters:
        if par.expandable:
            expanded = [deepcopy(d) for d in expanded for _ in range(len(par.value))]
            for d, v in zip(expanded, itertools.cycle(par.value)):
                d[par.name] = v
        else:
            for d in expanded:
                d[par.name] = par.value

    return expanded


def ask_for_input(
    question: str, default: Any = None, callback: Callable[[Any], Any] = str
) -> Any:
    """
    Ask for input from the user at the command line.

    Parameters
    ----------
    question
        A string to display as a prompt for the user.
    default
        The default answer to the question.
    callback
        The return value of this callback is what is returned from this
        function. Useful for simple conversions, like receiving an integer
        instead of a raw string.

    Returns
    -------
    answer
        The input, passed through ``callback``.
    """
    while True:
        input_str = input(question + " [Default: {}] > ".format(default))

        trimmed = input_str.replace(" ", "")
        if trimmed == "":
            return default

        try:
            return callback(trimmed)
        except Exception as e:
            print(e)


def ask_for_choices(
    question: str,
    choices: Union[List[str], Tuple[str, ...], Dict[str, Any]],
    default: Optional[str] = None,
):
    """
    Ask for input from the user, restricted to a given set of options.

    Parameters
    ----------
    question
        A string to display as a prompt prompt for the user.
    choices
        The choices to present to the user. If it is a tuple or list of strings,
        these will be the choices and whichever one is chosen will be returned.
        If it is a dictionary, the user will be asked to choose from the keys,
        and the matching value will be returned.
    default
        The default answer to the question. If this is ``None``, the default
        will be the first element of the choices.

    Returns
    -------
    answer
        The input, interpreted as a boolean.
    """
    if default is None:
        default = list(choices)[0]

    while True:
        answer = ask_for_input(question + f' [{" | ".join(choices)}]', default=default)
        if answer not in choices:
            print(f"{answer} is not a valid choice, try again")
            continue

        try:
            return choices[answer]
        except TypeError:
            return answer
        except Exception as e:
            print(e)


TRUE_ANSWERS = {"true", "t", "yes", "y", "1", "on", True}
FALSE_ANSWERS = {"false", "f", "no", "n", "0", "off", False, ""}


def ask_for_bool(question: str, default: Union[str, bool, int] = "") -> bool:
    """
    Ask for input from the user, which will be interpreted
    as a boolean. The interpretation is case-insensitive.

    Synonyms for ``True``: ``true``, ``t``, ``yes``, ``y``, ``1``, ``on``

    Synonyms for ``False``: ``false``, ``f``, ``no``, ``n``, ``0``, ``off``, ```` (i.e., nothing)

    Parameters
    ----------
    question
        A string to display as a prompt prompt for the user.
    default
        The default answer to the question, which is ``False``.

    Returns
    -------
    answer
        The input, interpreted as a boolean.
    """
    while True:
        input_str = input(question + " [Default: {}] > ".format(default))

        trimmed = input_str.replace(" ", "").lower()
        if trimmed in TRUE_ANSWERS:
            return True
        elif trimmed in FALSE_ANSWERS:
            return False
        else:
            print(f"Answer could not be interpreted as a boolean, try again")


def ask_for_eval(question: str, default: str = "None") -> Any:
    """
    Ask for input from the user, which will be evaluated as a Python command.

    Numpy and Scipy's top-level interfaces (imported as ``np`` and ``sp``) and
    Simulacra's own unit module (imported as ``u``) are all available.
    For example, entering ``np.linspace(0, 1, 100)`` will produce the
    expected result of an array of 100 numbers evenly spaced between 0 and 1.

    .. warning ::

        This function is not safe!
        The user can execute arbitrary Python code!
        You should only expose this function to known, trusted users.

    Parameters
    ----------
    question
        A string to display as a prompt prompt for the user.
    default
        The default answer to the question.

    Returns
    -------
    answer
        The result of evaluating the user's input.
    """
    while True:
        input_str = input(question + " [Default: {}] (eval) > ".format(default))

        trimmed = input_str.replace(" ", "")
        if trimmed == "":
            input_str = str(default)

        try:
            return eval(input_str)
        except Exception as e:
            print(e)
