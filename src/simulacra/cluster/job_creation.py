import itertools
import logging
import os
import pickle
import sys
from copy import deepcopy
from typing import Any, Iterable, Optional, Callable, Type, Tuple, Union, List

from tqdm import tqdm

from .. import sims, utils

import numpy as np  # needs to be here so that ask_for_eval works
import scipy as sp
from .. import units as u

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Parameter:
    """A class that represents a parameter of a :class:`Specification`."""

    def __init__(self, name: str, value: Any = None, expandable: bool = False):
        """
        Parameters
        ----------
        name : :class:`str`
            The name of the Parameter, which should match a keyword argument of the target :class:`Specification`.
        value : :class:`str`
            The value of the Parameter, or an iterable of values.
        expandable : :class:`Bool`
            If True, :func:`expand_parameters` will expand along an iterable `value`.
        """
        self.name = name
        self.value = value
        self.expandable = expandable

    def __str__(self):
        return '{} {} = {}'.format(self.__class__.__name__, self.name, self.value)

    def __repr__(self):
        if not self.expandable:
            return '{}(name = {}, value = {})'.format(self.__class__.__name__, self.name, self.value)
        else:
            return '{}(name = {} expandable: \n{})'.format(self.__class__.__name__, self.name, self.value, '\n  '.join(str(v) for v in self.value))


def expand_parameters(parameters: Iterable[Parameter]) -> List[dict]:
    """
    Expand an iterable of :class:`Parameter` to a list of dictionaries containing all of the combinations of parameter values.
    Each of these dictionaries can then be unpacked into a :class:`Specification`.

    If a :class:`Parameter` has ``expandable = True``, it will be expanded across the values in the outermost iterable in that :class:`Parameter`'s ``value``.

    Parameters
    ----------
    parameters : iterable of :class:`Parameter`
        The parameters to expand over.

    Returns
    -------
    list of :class:`dict`
        An list of dictionaries containing all of the combinations of parameters.
    """
    dicts = [{}]

    for par in parameters:
        name, value = par.name, par.value

        # make sure the value is an iterable that isn't a string and has a length
        if par.expandable and hasattr(par.value, '__iter__') and not isinstance(par.value, str) and hasattr(par.value, '__len__'):
            dicts = [deepcopy(d) for d in dicts for _ in range(len(value))]
            for d, v in zip(dicts, itertools.cycle(value)):
                d[name] = v
        else:
            for d in dicts:
                d[name] = value

    return dicts


def ask_for_input(question: str, default: Any = None, cast_to: Type = str) -> Any:
    """
    Ask for input from the user, with a default value, which will be cast to a specified type.

    Parameters
    ----------
    question : :class:`str`
        A string to display on the command prompt for the user.
    default
        The default answer to the question.
    cast_to
        A type to cast the user's input to.

    Returns
    -------

    """
    try:
        input_str = input(question + ' [Default: {}] > '.format(default))

        trimmed = input_str.replace(' ', '')
        if trimmed == '':
            out = cast_to(default)
        else:
            out = cast_to(trimmed)

        logger.debug('Got input from stdin for question "{}": {}'.format(question, out))

        return out
    except Exception as e:
        print(e)
        return ask_for_input(question, default = default, cast_to = cast_to)


def ask_for_bool(question: str, default: Union[str, bool] = False) -> bool:
    """
    Ask for input from the user, with a default value, which will be interpreted as a boolean.

    Synonyms for True: 'true', 't', 'yes', 'y', '1', 'on'
    Synonyms for False: 'false', 'f', 'no', 'n', '0', 'off'

    Parameters
    ----------
    question : :class:`str`
        A string to display on the command prompt for the user.
    default : :class:`str`
        The default answer to the question.

    Returns
    -------
    :class:`Bool`
        The input, interpreted as a boolean.
    """
    try:
        input_str = input(question + ' [Default: {}] > '.format(default))

        trimmed = input_str.replace(' ', '')
        if trimmed == '':
            input_str = str(default)

        logger.debug('Got input from stdin for question "{}": {}'.format(question, input_str))

        input_str_lower = input_str.lower()
        if input_str_lower in ('true', 't', 'yes', 'y', '1', 'on'):
            return True
        elif input_str_lower in ('false', 'f', 'no', 'n', '0', 'off'):
            return False
        else:
            raise ValueError('Invalid answer to question "{}"'.format(question))
    except Exception as e:
        print(e)
        return ask_for_bool(question, default = default)


def ask_for_eval(question: str, default: str = 'None') -> Any:
    """
    Ask for input from the user, with a default value, which will be evaluated as a Python command.

    Numpy's top-level interface (imported as np) and Simulacra's unit module (* imported) are both available. For example, ``'np.linspace(0, twopi, 100)'`` will produce the expected result.

    NB: this function is not safe! The user can execute arbitrary Python code.

    Parameters
    ----------
    question
        A string to display on the command prompt for the user.
    default
        The default answer to the question.

    Returns
    -------

    """
    input_str = input(question + ' [Default: {}] (eval) > '.format(default))

    trimmed = input_str.replace(' ', '')
    if trimmed == '':
        input_str = str(default)

    logger.debug('Got input from stdin for question "{}": {}'.format(question, input_str))

    try:
        return eval(input_str)
    except Exception as e:
        print(e)
        return ask_for_eval(question, default = default)


def abort_job_creation():
    """Abort job creation by exiting the script."""
    print('Aborting job creation...')
    logger.critical('Aborted job creation')
    sys.exit(0)


def create_job_subdirs(job_dir: str):
    """Create directories for the inputs, outputs, logs, and movies."""
    print('Creating job directory and subdirectories...')

    utils.ensure_dir_exists(job_dir)
    utils.ensure_dir_exists(os.path.join(job_dir, 'inputs'))
    utils.ensure_dir_exists(os.path.join(job_dir, 'outputs'))
    utils.ensure_dir_exists(os.path.join(job_dir, 'logs'))


def save_specifications(specifications: Iterable[sims.Specification], job_dir: str):
    """Save a list of Specifications."""
    print('Saving Specifications...')

    for spec in tqdm(specifications, ascii = True):
        spec.save(target_dir = os.path.join(job_dir, 'inputs/'))

    logger.debug('Saved Specifications')


def write_specifications_info_to_file(specifications: Iterable[sims.Specification], job_dir: str):
    """Write information from the list of the Specifications to a file."""
    print('Writing Specification info to file...')

    with open(os.path.join(job_dir, 'specifications.txt'), 'w', encoding = 'utf-8') as file:
        for spec in tqdm(specifications, ascii = True):
            file.write(str(spec.info()) + '\n')

    logger.debug('Wrote Specification information to file')


def write_parameters_info_to_file(parameters: Iterable[Parameter], job_dir: str):
    """Write information from the list of Parameters to a file."""
    print('Writing parameters to file...')

    with open(os.path.join(job_dir, 'parameters.txt'), 'w', encoding = 'utf-8') as file:
        for param in parameters:
            file.write(repr(param) + '\n')

    logger.debug('Wrote parameter information to file')


def specification_check(specifications: Iterable[sims.Specification], check: int = 3):
    """Ask the user whether some number of specifications look correct."""
    print('-' * 20)
    for s in specifications[0:check]:
        print(f'\n{s.info()}\n')
        print('-' * 20)

    print(f'Generated {len(specifications)} Specifications')

    check = ask_for_bool(f'Do the first {check} Specifications look correct?', default = 'No')
    if not check:
        abort_job_creation()


def write_job_info_to_file(job_info, job_dir: str):
    """Write job information to a file."""
    with open(os.path.join(job_dir, 'info.pkl'), mode = 'wb') as f:
        pickle.dump(job_info, f, protocol = -1)


def load_job_info_from_file(job_dir: str):
    """Load job information from a file."""
    with open(os.path.join(job_dir, 'info.pkl'), mode = 'rb') as f:
        return pickle.load(f)
