import logging
from typing import Callable, Iterable, Dict, Any, Mapping, Optional


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def bytes_to_str(num_bytes: float) -> str:
    """Return a number of bytes as a human-readable string."""
    for unit in ("bytes", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def table(
    headers: Iterable[str],
    rows: Iterable[Iterable[Any]],
    fill: str = "",
    header_fmt: Optional[Callable[[str], str]] = None,
    row_fmt: Optional[Callable[[str], str]] = None,
    alignment: Optional[Dict[str, str]] = None,
) -> str:
    """
    Return a string containing a simple table created from headers and rows of entries.

    Parameters
    ----------
    headers
        The column headers for the table.
    rows
        The entries for each row, for each column.
        Should be an iterable of iterables or mappings, with the outer level containing the rows,
        and each inner iterable containing the entries for each column.
        An iterable-type row is printed in order.
        A mapping-type row uses the headers as keys to align the stdout and can have missing values,
        which are filled using the ```fill`` value.
    fill
        The string to print in place of a missing value in a mapping-type row.
    header_fmt
        A function to be called on the header string.
        The return value is what will go in the output.
    row_fmt
        A function to be called on each row string.
        The return value is what will go in the output.
    alignment
        A map of headers to string method names to use to align each column.

    Returns
    -------
    table :
        A string containing the table.
    """
    if header_fmt is None:
        header_fmt = lambda _: _
    if row_fmt is None:
        row_fmt = lambda _: _
    if alignment is None:
        alignment = {}

    headers = tuple(headers)
    lengths = [len(h) for h in headers]

    align_methods = [alignment.get(h, "center") for h in headers]

    processed_rows = []
    for row in rows:
        if isinstance(row, Mapping):
            processed_rows.append([str(row.get(key, fill)) for key in headers])
        else:
            processed_rows.append([str(entry) for entry in row])

    for row in processed_rows:
        lengths = [max(curr, len(entry)) for curr, entry in zip(lengths, row)]

    header = header_fmt(
        "  ".join(
            getattr(h, a)(l) for h, l, a in zip(headers, lengths, align_methods)
        ).rstrip()
    )

    lines = (
        row_fmt(
            "  ".join(getattr(f, a)(l) for f, l, a in zip(row, lengths, align_methods))
        )
        for row in processed_rows
    )

    output = "\n".join((header, *lines))

    return output
