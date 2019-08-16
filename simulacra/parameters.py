import itertools
import logging
from copy import deepcopy
import textwrap
from typing import Any, Optional, Type, Union, List, Collection, Dict, Tuple

# these imports need to be here so that ask_for_eval works
import numpy as np
import scipy as sp
from simulacra import units as u

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Parameter:
    """A class that represents a parameter of a :class:`Specification`."""

    def __init__(self, name: str, value: Any = None, expandable: bool = False):
        """
        Parameters
        ----------
        name
            The name of the Parameter, which should match a keyword argument of
            the target :class:`Specification`.
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
    expanded_parameters :
        An list of dictionaries containing all of the combinations of parameters.
    """
    expanded = [{}]

    for par in parameters:
        if par.expandable:
            expanded = [deepcopy(d) for d in expanded for _ in range(len(par.value))]
            for d, v in zip(expanded, itertools.cycle(par.value)):
                d[par.name] = v
        else:
            for d in expanded:
                d[par.name] = par.value

    return expanded


def ask_for_input(question: str, default: Any = None, cast_to: Type = str) -> Any:
    """
    Ask for input from the user, with a default value, which will be cast to a specified type.

    Parameters
    ----------
    question
        A string to display on the command prompt for the user.
    default
        The default answer to the question.
    cast_to
        A type to cast the user's input to.

    Returns
    -------

    """
    while True:
        input_str = input(question + " [Default: {}] > ".format(default))

        trimmed = input_str.replace(" ", "")
        if trimmed == "":
            return default

        try:
            return cast_to(trimmed)
        except Exception as e:
            print(e)


def ask_for_choices(
    question: str,
    choices: Union[Tuple[str], Dict[str, Any]],
    default: Optional[str] = None,
):
    if default is None:
        default = list(choices.keys())[0]

    while True:
        answer = ask_for_input(question + f' [{" | ".join(choices)}]', default=default)
        if answer not in choices:
            print(f"{answer} is not a valid choice, try again")
            continue

        try:
            return choices[answer]
        except KeyError:
            return answer


TRUE_ANSWERS = ("true", "t", "yes", "y", "1", "on", True)
FALSE_ANSWERS = ("false", "f", "no", "n", "0", "off", False)


def ask_for_bool(question: str, default: Union[str, bool] = False) -> bool:
    """
    Ask for input from the user, with a default value, which will be interpreted as a boolean.

    Synonyms for True: 'true', 't', 'yes', 'y', '1', 'on'
    Synonyms for False: 'false', 'f', 'no', 'n', '0', 'off'

    Parameters
    ----------
    question
        A string to display on the command prompt for the user.
    default
        The default answer to the question.

    Returns
    -------
    answer :
        The input, interpreted as a boolean.
    """
    while True:
        input_str = input(question + " [Default: {}] > ".format(default))

        trimmed = input_str.replace(" ", "").lower()
        if trimmed == "":
            trimmed = default

        if trimmed in TRUE_ANSWERS:
            return True
        elif trimmed in FALSE_ANSWERS:
            return False
        else:
            print(f"Answer could not be interpreted as a boolean, try again")


def ask_for_eval(question: str, default: str = "None") -> Any:
    """
    Ask for input from the user, with a default value, which will be evaluated as a Python command.

    Numpy and Scipy's top-level interfaces (imported as ``np`` and ``sp``) and
    Simulacra's own unit module (imported as ``u``) are both available.
    For example, ``'np.linspace(0, twopi, 100)'`` will produce the expected
    result of 100 numbers evenly spaced between zero and twopi.

    NB: this function is not safe!
    The user can execute arbitrary Python code!

    Parameters
    ----------
    question
        A string to display on the command prompt for the user.
    default
        The default answer to the question.

    Returns
    -------
    answer

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
