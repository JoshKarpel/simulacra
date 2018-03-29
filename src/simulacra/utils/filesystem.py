import os
import logging
from typing import Optional, Union, NamedTuple, Callable, Iterable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    split_path = os.path.splitext(path)
    if split_path[0] != path:  # path is file
        path_to_make = os.path.dirname(split_path[0])
    else:  # path is dir
        path_to_make = split_path[0]
    os.makedirs(path_to_make, exist_ok = True)

    logger.debug('Ensured dir {} exists'.format(path_to_make))

    return path_to_make


def get_file_size(file_path: str) -> int:
    """Return the size of the file at file_path."""
    return os.stat(file_path).st_size

