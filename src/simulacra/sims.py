import datetime
import gzip
import pickle
import uuid
from copy import deepcopy
from typing import Optional, Union
import abc

import logging
import os
from simulacra.info import Info

from . import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Beet:
    """
    A class that provides an easy interface for pickling and unpickling instances.

    Two Beets compare and hash equal if they have the same uid attribute, a uuid4 generated during initialization.

    Attributes
    ----------
    uuid
        A `Universally Unique Identifier <https://en.wikipedia.org/wiki/Universally_unique_identifier>`_ for the :class:`Beet`.
    """

    def __init__(self, name: str, file_name: Optional[str] = None):
        """
        Parameters
        ----------
        name : :class:`str`
            The internal name of the Beet.
        file_name : :class:`str`
            The desired external name of the Beet. Automatically derived from `name` if ``None``. Illegal characters are stripped before use, and spaces are replaced with underscores.
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

    def __repr__(self):
        return f"{self.__class__.__name__}(name = '{self.name}', file_name = '{self.file_name}', uuid = {self.uuid})"

    def __str__(self):
        if self.name != self.file_name:
            return f'{self.__class__.__name__}({self.name}, file_name = {self.file_name})'
        else:
            return f'{self.__class__.__name__}({self.name})'

    def __eq__(self, other: 'Beet'):
        """Two Beets are equal if they have the same UUID."""
        return isinstance(other, self.__class__) and self.uuid == other.uuid

    def __hash__(self):
        """The hash of the Beet is the hash of its UUID."""
        return hash(self.uuid)

    def clone(self, **kwargs) -> 'Beet':
        """
        Return a deepcopy of the Beet.

        If any kwargs are passed, they will be interpreted as key-value pairs and ``clone`` will try to :func:`setattr` them on the new Beet.

        Parameters
        ----------
        kwargs
            Key-value pairs to modify attributes on the new Beet.

        Returns
        -------
        :class:`Beet`
            The new (possibly modified) :class:`Beet`.
        """
        new_beet = deepcopy(self)

        for k, v in kwargs.items():
            setattr(new_beet, k, v)

        return new_beet

    def save(self, target_dir: Optional[str] = None, file_extension: str = '.beet', compressed: bool = True, ensure_dir_exists: bool = True) -> str:
        """
        Atomically pickle the Beet to a file.

        Parameters
        ----------
        target_dir : :class:`str`
            The directory to save the Beet to.
        file_extension : :class:`str`
            The file extension to name the Beet with (for keeping track of things, no actual effect).
        compressed : :class:`bool`
            Whether to compress the Beet using gzip.
        ensure_dir_exists : :class:`bool`
            Whether to ensure that the target directory exists before saving.

        Returns
        -------
        :class:`str`
            The path to the saved Beet.
        """
        if target_dir is None:
            target_dir = os.getcwd()

        file_path = os.path.join(target_dir, self.file_name + file_extension)
        file_path_working = file_path + '.working'

        if ensure_dir_exists:
            utils.ensure_dir_exists(file_path_working)

        if compressed:
            op = gzip.open
        else:
            op = open

        with op(file_path_working, mode = 'wb') as file:
            pickle.dump(self, file, protocol = -1)

        os.replace(file_path_working, file_path)

        logger.debug('Saved {} {} to {}'.format(self.__class__.__name__, self.name, file_path))

        return file_path

    @classmethod
    def load(cls, file_path: str) -> 'Beet':
        """
        Load a Beet from `file_path`.

        Parameters
        ----------
        file_path
            The path to load a Beet from.

        Returns
        -------
        :class:`Beet`
            The loaded Beet.
        """
        try:
            with gzip.open(file_path, mode = 'rb') as file:
                beet = pickle.load(file)
        except OSError:  # file is not gzipped
            with open(file_path, mode = 'rb') as file:
                beet = pickle.load(file)

        logger.debug('Loaded {} {} from {}'.format(beet.__class__.__name__, beet.name, file_path))

        return beet

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
    status : :class:`str`
        The status of the Simulation. One of ``'initialized'``, ``'running'``, ``'finished'``, ``'paused'``, or ``'error'``.
    """

    def __init__(self, spec):
        """
        Parameters
        ----------
        spec : :class:`Specification`
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
    def status(self):
        return self._status

    @status.setter
    def status(self, status: Status):
        """
        Set the status of the :class:`Simulation`.

        Defined statuses are ``STATUS_INI`` (initialized), ``STATUS_RUN`` (running), ``STATUS_FIN`` (finished), ``STATUS_PAU`` (paused), and ``STATUS_ERR`` (error).
        These statuses can have side effects on the simulation's time diagnostics.

        Parameters
        ----------
        status : Status
            The new status for the simulation.
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

        logger.debug(f'{self.__class__.__name__} {self.name} status set to {self.status} from {old_status}')

    def __str__(self):
        return super().__str__() + f' {{{self.status}}}'

    def save(self, target_dir: Optional[str] = None, file_extension: str = '.sim', **kwargs) -> str:
        """
        Atomically pickle the Simulation to a file.

        Parameters
        ----------
        target_dir : :class:`str`
            The directory to save the Simulation to.
        file_extension : :class:`str`
            The file extension to name the Simulation with (for keeping track of things, no actual effect).
        compressed : :class:`bool`
            Whether to compress the Beet using gzip.

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


class Specification(Beet):
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

    simulation_type = Simulation

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

    def save(self, target_dir: Optional[str] = None, file_extension: str = '.spec', **kwargs) -> str:
        """
        Atomically pickle the Specification to a file.

        Parameters
        ----------
        target_dir : :class:`str`
            The directory to save the Specification to.
        file_extension : :class:`str`
            The file extension to name the Specification with (for keeping track of things, no actual effect).
        compressed : :class:`bool`
            Whether to compress the Beet using gzip.

        Returns
        -------
        :class:`str`
            The path to the saved Specification.
        """
        return super().save(target_dir = target_dir, file_extension = file_extension, **kwargs)

    def to_sim(self):
        """Return a Simulation of the type associated with the Specification type, generated from this instance."""
        return self.simulation_type(self)

    def info(self) -> Info:
        info = super().info()

        if len(self._extra_attr_keys) > 0:
            info_extra = Info(header = f'Extra Attributes')

            for k in self._extra_attr_keys:
                info_extra.add_field(k, getattr(self, k))

            info.add_info(info_extra)

        return info
