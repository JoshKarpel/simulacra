import datetime
import os
from pathlib import Path
import logging
from typing import Optional, Union, NamedTuple, Callable, Iterable

from . import filesystem

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ILLEGAL_FILENAME_CHARACTERS = {'<', '>', ':', '"', '/', '\\', '|', '?', '*'}  # these characters should be stripped from file names before use


def strip_illegal_characters(string: str) -> str:
    """Strip characters that cannot be included in file names from a string."""
    return ''.join([char for char in string if char not in ILLEGAL_FILENAME_CHARACTERS])


def get_now_str() -> str:
    """Return a formatted string with the current year-month-day_hour-minute-second."""
    return datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')


def bytes_to_str(num_bytes: int) -> str:
    """Return a number of bytes as a human-readable string."""
    for unit in ('bytes', 'KB', 'MB', 'GB'):
        if num_bytes < 1024:
            return f'{num_bytes:.1f} {unit}'
        num_bytes /= 1024
    return f'{num_bytes:.1f} TB'


def get_file_size_as_string(path: Union[Path, str]) -> str:
    """Return the size of the file at file_path as a human-readable string."""
    return bytes_to_str(filesystem.get_file_size(Path(path)))


def table(headers: Iterable[str], rows: Iterable[Iterable]) -> str:
    """
    Return a string containing a simple table created from headers and rows of entries.

    Parameters
    ----------
    headers
        The column headers for the table.
    rows
        The entries for each row, for each column.
        Should be an iterable of iterables, with the outer level containing the rows, and each inner iterable containing the entries for each column.
        A ``None`` in the outer iterable produces a horizontal bar at that position.

    Returns
    -------
    table
        A string containing the table.
    """
    lengths = [len(h) for h in headers]
    rows = [[str(entry) for entry in row] if row is not None else None for row in rows]
    for row in rows:
        if row is None:
            continue

        lengths = [max(curr, len(entry)) for curr, entry in zip(lengths, row)]

    header = ' ' + ' │ '.join(h.center(l) for h, l in zip(headers, lengths)) + ' '
    bar = ''.join('─' if char != '│' else '┼' for char in header)
    bottom_bar = bar.replace('┼', '┴')

    lines = []
    for row in rows:
        if row is None:
            lines.append(bar)
        else:
            lines.append(' ' + ' │ '.join(f.center(l) for f, l in zip(row, lengths)))

    output = '\n'.join((
        header,
        bar,
        *lines,
        bottom_bar,
    ))

    return output
