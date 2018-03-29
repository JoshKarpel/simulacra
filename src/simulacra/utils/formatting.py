import datetime
import os
import logging
from typing import Optional, Union, NamedTuple, Callable, Iterable


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ILLEGAL_FILENAME_CHARACTERS = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']  # these characters should be stripped from file names before use


def strip_illegal_characters(string: str) -> str:
    """Strip characters that cannot be included in file names from a string."""
    return ''.join([char for char in string if char not in ILLEGAL_FILENAME_CHARACTERS])


def get_now_str() -> str:
    """Return a formatted string with the current year-month-day_hour-minute-second."""
    return datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')


def bytes_to_str(num: Union[float, int]) -> str:
    """Return a number of bytes as a human-readable string."""
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def get_file_size_as_string(file_path: str) -> str:
    """Return the size of the file at file_path as a human-readable string."""
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return bytes_to_str(file_info.st_size)


def table(headers: Iterable[str], rows: Iterable[Iterable]) -> str:
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
