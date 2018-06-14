import datetime
import gzip
import pickle
import uuid
from copy import deepcopy
from pathlib import Path
import abc
from typing import Optional, Union, Type

import logging
import os
from simulacra.info import Info

from . import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Beet:
    """
    A class that provides an easy interface for pickling and unpickling instances.

    Beets can be compared and hashed based on their :class:`Beet.uuid` value.

    Attributes
    ----------
    uuid
        A `Universally Unique Identifier <https://en.wikipedia.org/wiki/Universally_unique_identifier>`_ for the :class:`Beet`.
    """

    def __init__(self, name: str, file_name: Optional[str] = None):
        """
        Parameters
        ----------
        name
            The internal name of the :class:`Beet`.
        file_name
            The desired external name of the :class:`Beet`.
            Automatically derived from ``name`` if ``None``.
            Either way, illegal characters are stripped and spaces are replaced with underscores.
        """
        self.name = str(name)
        if file_name is None:
            file_name = self.name

        file_name_stripped = utils.strip_illegal_characters(str(file_name).replace(' ', '_'))
        if file_name_stripped != file_name:
            logger.warning('Using file name {} instead of {} for {}'.format(file_name_stripped, file_name, self.name))
        self.file_name = file_name_stripped

        self.initialized_at = datetime.datetime.utcnow()
        self.uuid = uuid.uuid4()

        logger.info('Initialized {}'.format(repr(self)))

    def __eq__(self, other: 'Beet'):
        """Two Beets are equal if they have the same UUID."""
        return isinstance(other, self.__class__) and self.uuid == other.uuid

    def __hash__(self):
        """The hash of the Beet is the hash of its UUID."""
        return hash(self.uuid)

    def clone(self, **kwargs) -> 'Beet':
        """
        Return a deepcopy of the :class:`Beet`.

        If any kwargs are passed, they will be interpreted as key-value pairs and ``clone`` will try to :func:`setattr` them on the new Beet.

        The new :class:`Beet` will have a different UUID.

        Parameters
        ----------
        kwargs
            Key-value pairs to update attributes on the new Beet.

        Returns
        -------
        Beet
            The new (possibly modified) :class:`Beet`.
        """
        new_beet = deepcopy(self)
        new_beet.__dict__.update(kwargs)
        new_beet.uuid = uuid.uuid4()

        return new_beet

    def save(
        self,
        target_dir: Optional[Path] = None,
        file_extension: str = 'beet',
        compressed: bool = True,
    ) -> str:
        """
        Atomically pickle the :class:`Beet` to a file.

        Parameters
        ----------
        target_dir : :class:`str`
            The directory to save the Beet to.
        file_extension : :class:`str`
            The file extension to name the Beet with (for keeping track of things, no actual effect).
        compressed : :class:`bool`
            Whether to compress the Beet using gzip.

        Returns
        -------
        str
            The path to the saved :class:`Beet`.
        """
        if target_dir is None:
            target_dir = Path.cwd()

        path = (Path(target_dir).absolute() / f'{self.file_name}.{file_extension}')
        working_path = path.with_name(f'{path.name}.working')

        utils.ensure_parents_exist(working_path)

        op = gzip.open if compressed else open
        with op(working_path, mode = 'wb') as file:
            pickle.dump(self, file, protocol = -1)

        os.replace(working_path, path)

        logger.debug(f'Saved {self} to {path}')

        return path

    @classmethod
    def load(cls, path: str) -> 'Beet':
        """
        Load a Beet from `file_path`.

        Parameters
        ----------
        path
            The path to load a Beet from.

        Returns
        -------
        :class:`Beet`
            The loaded Beet.
        """
        path = Path(path)
        try:
            with gzip.open(path, mode = 'rb') as file:
                beet = pickle.load(file)
        except OSError:  # file is not gzipped
            with path.open(mode = 'rb') as file:
                beet = pickle.load(file)

        logger.debug(f'Loaded {beet} from {path}')

        return beet

    def __repr__(self):
        return f"{self.__class__.__name__}(name = '{self.name}', file_name = '{self.file_name}', uuid = {self.uuid})"

    def __str__(self):
        if self.name != self.file_name:
            return f'{self.__class__.__name__}({self.name}, file_name = {self.file_name})'
        else:
            return f'{self.__class__.__name__}({self.name})'

    def info(self) -> Info:
        info = Info(header = str(self))
        info.add_field('UUID', self.uuid)

        return info


class Status(utils.StrEnum):
    UNINITIALIZED = 'uninitialized'
    INITIALIZED = 'initialized'
    RUNNING = 'running'
    FINISHED = 'finished'
    PAUSED = 'paused'
    ERROR = 'error'


class Simulation(Beet, abc.ABC):
    """
    A class that represents a single simulation.

    It should be subclassed and customized for each variety of simulation.

    Simulations should generally be instantiated using Specification.to_sim() to avoid possible mismatches.

    Attributes
    ----------
    uuid
        A `Universally Unique Identifier <https://en.wikipedia.org/wiki/Universally_unique_identifier>`_ for the :class:`Simulation`.
    status : Status
        The status of the Simulation.
    """

    def __init__(self, spec):
        """
        Parameters
        ----------
        spec
            The :class:`Specification` for the Simulation.
        """
        super().__init__(spec.name, file_name = spec.file_name)  # inherit name and file_name from spec

        self.spec = spec

        # diagnostic data
        self.runs = 0
        self.init_time = None
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.latest_run_time = None
        self.running_time = datetime.timedelta()

        self._status = Status.UNINITIALIZED
        self.status = Status.INITIALIZED

    @property
    def status(self) -> Status:
        return self._status

    @status.setter
    def status(self, status: Status):
        """
        Set the status of the :class:`Simulation`.

        Defined statuses are ``STATUS_INI`` (initialized), ``STATUS_RUN`` (running), ``STATUS_FIN`` (finished), ``STATUS_PAU`` (paused), and ``STATUS_ERR`` (error).
        These statuses can have side effects on the simulation's time diagnostics.

        Parameters
        ----------
        status
            The new status for the :class:`Simulation`.
        """
        if not isinstance(status, Status):
            raise TypeError(f'{status} is not a member of Status')

        old_status = self.status
        now = datetime.datetime.utcnow()

        if status == Status.INITIALIZED:
            self.init_time = now
        elif status == Status.RUNNING:
            if self.start_time is None:
                self.start_time = now
            self.latest_run_time = now
            self.runs += 1
        elif status == Status.PAUSED:
            if self.latest_run_time is not None:
                self.running_time += now - self.latest_run_time
        elif status == Status.FINISHED:
            if self.latest_run_time is not None:
                self.running_time += now - self.latest_run_time
            self.end_time = now
            self.elapsed_time = self.end_time - self.init_time

        self._status = status

        logger.debug(f'{self} status set to {self.status} from {old_status}')

    def save(
        self,
        target_dir: Optional[str] = None,
        file_extension: str = 'sim',
        **kwargs,
    ) -> str:
        """
        Atomically pickle the :class:`Simulation` to a file.

        Parameters
        ----------
        target_dir
            The directory to save the :class:`Simulation` to.
        file_extension
            The file extension to name the :class:`Simulation` with (for keeping track of things, no actual effect).

        Returns
        -------
        :class:`str`
            The path to the saved Simulation.
        """
        if self.status != Status.FINISHED:
            self.status = Status.PAUSED

        return super().save(target_dir = target_dir, file_extension = file_extension, **kwargs)

    @abc.abstractmethod
    def run(self):
        """Hook method for running the Simulation, whatever that may entail."""
        raise NotImplementedError

    def __str__(self):
        return super().__str__() + f' {{{self.status}}}'

    def info(self) -> Info:
        """Return a string describing the parameters of the Simulation and its associated Specification."""
        info = super().info()

        info.add_field('UUID', self.uuid)

        info_diag = Info(header = f'Status: {self.status}')
        info_diag.add_field('Initialization Time', self.init_time)
        info_diag.add_field('Latest Run Time', self.latest_run_time)
        info_diag.add_field('End Time', self.end_time)
        info_diag.add_field('Running Time', self.running_time)
        info_diag.add_field('Elapsed Time', self.elapsed_time)

        info.add_infos(info_diag, self.spec.info())

        return info


class Specification(Beet, abc.ABC):
    """
    A class that contains the information necessary to run a simulation.

    It should be subclassed for each type of simulation and all additional information necessary to run that kind of simulation should be added via keyword arguments.

    Any number of additional keyword arguments can be passed to the constructor.
    They will be stored as attributes if they don't conflict with any attributes already set.

    Attributes
    ----------
    simulation_type
        A class attribute which determines what kind of :class:`Simulation` will be generated via :func:`Specification.to_sim`.
    uuid
        A `Universally Unique Identifier <https://en.wikipedia.org/wiki/Universally_unique_identifier>`_ for the :class:`Specification`.
    """

    simulation_type: Type[Simulation] = Simulation

    def __init__(self, name: str, file_name: Optional[str] = None, **kwargs):
        """
        Parameters
        ----------
        name : :class:`str`
            The internal name of the Specification.
        file_name : :class:`str`
            The desired external name of the Specification.
            Automatically derived from `name` if ``None`` is passed.
            Illegal characters are stripped before use, and spaces are replaced with underscores.
        kwargs
            Any number of keyword arguments, which will be stored as attributes.
        """
        super().__init__(name, file_name = file_name)

        self._extra_attr_keys = list()

        for k, v in ((k, v) for k, v in kwargs.items() if k not in self.__dict__):
            setattr(self, k, v)
            self._extra_attr_keys.append(k)
            logger.debug('{} stored additional attribute {} = {}'.format(self.name, k, v))

    def save(
        self,
        target_dir: Optional[str] = None,
        file_extension: str = 'spec',
        **kwargs,
    ) -> str:
        """
        Pickle the :class:`Specification` to a file.

        Parameters
        ----------
        target_dir
            The directory to save the :class:`Specification` to.
        file_extension
            The file extension to name the Specification with (for keeping track of things, no actual effect).
        compressed
            Whether to compress the Beet using gzip.

        Returns
        -------
        path
            The path to the saved :class:`Specification`.
        """
        return super().save(target_dir = target_dir, file_extension = file_extension, **kwargs)

    def to_sim(self):
        """Return a :class:`Simulation` of the type associated with the :class:`Specification` type, generated from this instance."""
        return self.simulation_type(self)

    def info(self) -> Info:
        info = super().info()

        if len(self._extra_attr_keys) > 0:
            info_extra = Info(header = f'Extra Attributes')

            for k in self._extra_attr_keys:
                info_extra.add_field(k, getattr(self, k))

            info.add_info(info_extra)

        return info
