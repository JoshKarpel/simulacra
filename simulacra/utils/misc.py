import logging
from typing import List

import enum
import subprocess


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class StrEnum(str, enum.Enum):
    """An :class:`enum.Enum` whose members are also strings."""

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.value.upper()}"

    def __str__(self):
        return self.value


class SubprocessManager:
    """A context manager that runs a process for the duration of its block."""

    def __init__(self, cmd: List[str], **subprocess_kwargs):
        """
        Parameters
        ----------
        cmd
            A list containing the executable and arguments, as strings.
        subprocess_kwargs
            Additional keyword arguments are passed to :func:`subprocess.Popen`.
        """
        self.cmd = cmd
        self.subprocess_kwargs = subprocess_kwargs

        self.name = self.cmd[0]

        self.subprocess = None

    def __enter__(self):
        self.subprocess = subprocess.Popen(self.cmd, **self.subprocess_kwargs)

        logger.debug(f"Opened subprocess {self.name}")

        return self.subprocess

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.subprocess.communicate()
            logger.debug(f"Closed subprocess {self.name}")
        except Exception:
            logger.warning(
                f"Exception while trying to close subprocess {self.name}, possibly not closed"
            )
