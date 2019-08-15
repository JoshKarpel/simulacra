from pathlib import Path
import logging
from typing import Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ensure_parents_exist(path: Union[Path, str]):
    """
    Ensure that the directory tree to the path exists.

    Parameters
    ----------
    path
        A path to a file or directory.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_file_size(path: Union[Path, str]) -> int:
    """Return the size of the file at the given path."""
    return Path(path).stat().st_size
