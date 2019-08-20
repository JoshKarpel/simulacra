import logging
from typing import Optional

import os
import sys
import datetime
from pathlib import Path

from . import filesystem

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)s] ~ %(message)s", datefmt="%y-%m-%d %H:%M:%S"
)


class LogManager:
    """
    A context manager to set up logging.

    Within a managed block, logging messages are intercepted if their
    logger is named in ``logger_names``.
    The object returned by the LogManager ``with`` statement can be used as a
    logger, with name given by ``manual_logger_name``.
    """

    def __init__(
        self,
        *logger_names,
        manual_logger_name: Optional[None],
        log_formatter: logging.Formatter = DEFAULT_LOG_FORMATTER,
        stdout_logs: bool = True,
        stdout_level: int = logging.DEBUG,
        file_logs: bool = False,
        file_level: int = logging.DEBUG,
        file_name: Optional[str] = None,
        file_dir: Optional[os.PathLike] = None,
        file_mode: str = "a",
    ):
        """
        Parameters
        ----------
        logger_names
            The names of loggers to intercept.
        manual_logger_name
            The name used by the logger returned by the :class:`LogManager`
            ``with`` statement. If not given, it will be the first logger
            passed in ``logger_names``.
        log_formatter
            A log formatter to use for all log messages.
        stdout_logs
            If ``True``, log messages will be displayed on stdout.
        stdout_level
            Sets the level of messages to display on stdout.
        file_logs
            If ``True`, log messages will be sent to a file.
        file_level
            Sets the level of messages to send to the file.
        file_name
            The name of the log file. If not given, it will named ``log__<timestamp>``.
        file_dir
            The directory to place the log file in. If not given, it will be in
            the current working directory.
        file_mode
            The file mode to open the log file with, defaults to 'a' (append).
        """
        self.logger_names = list(logger_names)
        if manual_logger_name is None:
            manual_logger_name = self.logger_names[0]
        if manual_logger_name not in self.logger_names:
            self.logger_names = [manual_logger_name] + self.logger_names

        self.stdout_logs = stdout_logs
        self.stdout_level = stdout_level

        self.file_logs = file_logs
        self.file_level = file_level

        if file_name is None:
            file_name = f"log__{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')}"
        self.file_name = file_name
        if not self.file_name.endswith(".log"):
            self.file_name += ".log"

        if file_dir is None:
            file_dir = Path.cwd()
        self.file_dir = Path(file_dir)

        self.file_mode = file_mode

        self.log_formatter = log_formatter

        self.logger = None

    def __enter__(self):
        """Gets a logger with the specified name, replace it's handlers with, and returns itself."""
        self.loggers = {name: logging.getLogger(name) for name in self.logger_names}

        new_handlers = [logging.NullHandler()]

        if self.stdout_logs:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(self.stdout_level)
            stdout_handler.setFormatter(self.log_formatter)

            new_handlers.append(stdout_handler)

        if self.file_logs:
            log_file_path = self.file_dir / self.file_name

            # NB: the log message emitted here will not be included in the
            # logger being created by this context manager
            filesystem.ensure_parents_exist(log_file_path)

            file_handler = logging.FileHandler(
                log_file_path, mode=self.file_mode, encoding="utf-8"
            )
            file_handler.setLevel(self.file_level)
            file_handler.setFormatter(self.log_formatter)

            new_handlers.append(file_handler)

        self.old_levels = {name: logger.level for name, logger in self.loggers.items()}
        self.old_handlers = {
            name: logger.handlers for name, logger in self.loggers.items()
        }

        for logger in self.loggers.values():
            logger.setLevel(logging.DEBUG)
            logger.handlers = new_handlers

        return self.loggers[self.logger_names[0]]

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.disable(logging.NOTSET)

        for name, logger in self.loggers.items():
            logger.level = self.old_levels[name]
            logger.handlers = self.old_handlers[name]
