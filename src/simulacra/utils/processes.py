import multiprocessing
import subprocess
import logging
from typing import Optional, Union, NamedTuple, Callable, Iterable

import psutil

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        logger.debug('Suspended {}'.format(p))


def resume_processes(processes: Iterable[psutil.Process]):
    """
    Resume a list of processes.

    Parameters
    ----------
    processes : iterable of psutil.Process
    """
    for p in processes:
        p.resume()
        logger.debug('Resumed {}'.format(p))


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
