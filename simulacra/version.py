from typing import Tuple, Optional

import re

__version__ = "0.2.0"

version_re = re.compile(
    r"^(\d+) \. (\d+) (\. (\d+))? ([ab](\d+))?$", re.VERBOSE | re.ASCII
)


def version() -> str:
    """Return a string containing human-readable version information."""
    return f"Simulacra version {__version__}"


def version_info() -> Tuple[int, int, int, Optional[str], Optional[int]]:
    """Return a tuple of version information: ``(major, minor, micro, prerelease, prelease_number)``."""
    return _version_info(__version__)


def _version_info(v: str) -> Tuple[int, int, int, str, int]:
    match = version_re.match(v)
    (major, minor, micro, prerelease, prerelease_num) = match.group(1, 2, 4, 5, 6)

    out = (
        int(major),
        int(minor),
        int(micro or 0),
        prerelease[0] if prerelease is not None else None,
        int(prerelease_num) if prerelease_num is not None else None,
    )

    return out
