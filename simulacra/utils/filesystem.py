from typing import Union
import logging

import os
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ensure_parents_exist(path: os.PathLike):
    """
    Ensure that the directory tree to the path exists.

    Parameters
    ----------
    path
        A path to a file or directory.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_file_size(path: os.PathLike) -> int:
    """Return the size of the file at the given path."""
    return Path(path).stat().st_size
