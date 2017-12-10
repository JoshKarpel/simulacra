"""
Simulacra core functionality.


Copyright 2017 Josh Karpel

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import datetime
import gzip
import pickle
import uuid
import collections
from copy import deepcopy
from typing import Optional, Union, List, Tuple, Iterable, Any

import logging
import os

from . import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SimulacraException(Exception):
    """Base :class:`Exception` for all Simulacra exceptions."""
    pass


class Info:
    """
    A class that represents a hierarchical collection of information.

    Each :class:`Info` contains a header and a dictionary of children.
    The header is a string that will be written at the top-level of this Info.
    Each child is either a field, which will be written out as ``'{key}: {value}'``, or another Info, which will display itself.

    Field names are unique.
    """

    def __init__(self, *, header: str):
        """
        Parameters
        ----------
        header
            The header for this :class:`Info`.
        """
        self.header = header
        self.children = collections.OrderedDict()

    def __str__(self) -> str:
        field_strings = [self.header]

        for field, value in self.children.items():
            if isinstance(value, Info):
                info_strings = str(value).split('\n')
                field_strings.append('├─ ' + info_strings[0])
                field_strings.extend('│  ' + info_string for info_string in info_strings[1:])
            else:
                field_strings.append(f'├─ {field}: {value}')

        # this loop goes over the field strings in reverse, cleaning up the tail of the structure indicators
        for index, field_string in reversed(list(enumerate(field_strings))):
            if field_string[0] == '├':  # this is the last branch on this level, replace it with endcap and break
                field_strings[index] = field_string.replace('├', '└')
                break
            else:  # not yet at last branch, continue cleanup
                field_strings[index] = field_string.replace('│', ' ', 1)

        return '\n'.join(field_strings)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.header})'

    def log(self, level = logging.INFO):
        """Emit a log message containing the formatted info output."""
        logger.log(level, '\n' + str(self))

    def json(self):
        # TODO: json output from Info
        raise NotImplementedError

    def add_field(self, name: str, value: Any):
        """
        Add a field to the :class:`Info`, which will be displayed as ``'{name}: {value}'``.

        Parameters
        ----------
        name : :class:`str`
            The name of the field.
        value : :class:`str`
            The value of the field.
        """
        self.children[name] = value

    def add_fields(self, name_value_pairs):
        """
        Add a list of fields to the :class:`Info`.

        Parameters
        ----------
        name_value_pairs : iterable
            An iterable or dict of ``(name, value)`` pairs to add as fields.
        """
        self.children.update(dict(name_value_pairs))

    def add_info(self, info: 'Info'):
        """
        Add a sub-Info to the :class:`Info`, which will be displayed at a deeper indentation level.

        Parameters
        ----------
        info : :class:`Info`
            An :class:`Info` to be added as a sub-Info.
        """
        self.children[info.header] = info

    def add_infos(self, *infos):
        """
        Add a list of Infos to this Info as sub-Infos.

        Parameters
        ----------
        infos : iterable
            An iterable of :class:`Info`
        """
        self.children.update({id(info): info for info in infos})


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

    def __str__(self):
        if self.name != self.file_name:
            return f'{self.__class__.__name__}({self.name}, file_name = {self.file_name})'
        else:
            return f'{self.__class__.__name__}({self.name})'

    def __repr__(self):
        return utils.field_str(self, 'name', 'file_name', 'uuid')

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
    INITIALIZED = 'initialized'
    RUNNING = 'running'
    FINISHED = 'finished'
    PAUSED = 'paused'
    ERROR = 'error'


class Simulation(Beet):
    """
    A class that represents a single simulation.

    It should be subclassed and customized for each variety of simulation.

    Simulations should generally be instantiated using Specification.to_simulation() to avoid possible mismatches.

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

        self._status = None
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

    def run_simulation(self):
        """Hook method for running the Simulation, whatever that may entail."""
        raise NotImplementedError

    def info(self) -> Info:
        """Return a string describing the parameters of the Simulation and its associated Specification."""
        info = super().info()

        info.add_field('UUID', self.uuid)

        info_diag = Info(header = f'Status: {self.status}')
        info_diag.add_fields({
            'Initialization Time': self.init_time,
            'Latest Run Time': self.latest_run_time,
            'End Time': self.end_time,
            'Running Time': self.running_time,
            'Elapsed Time': self.elapsed_time,
        })

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
        A class attribute which determines what kind of :class:`Simulation` will be generated via :func:`Specification.to_simulation`.
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

    def to_simulation(self) -> 'Simulation':
        """Return a Simulation of the type associated with the Specification, generated from this instance."""
        return self.simulation_type(self)

    def info(self) -> Info:
        info = super().info()

        if len(self._extra_attr_keys) > 0:
            info_extra = Info(header = f'Extra Attributes')

            for k in self._extra_attr_keys:
                info_extra.add_field(k, getattr(self, k))

            info.add_info(info_extra)

        return info


class Summand:
    """
    An object that can be added to other objects that it shares a superclass with.
    """

    def __init__(self, *args, **kwargs):
        self.summation_class = Sum

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def __iter__(self):
        """When unpacked, yield self, to ensure compatability with Sum's __add__ method."""
        yield self

    def __add__(self, other: Union['Summand', 'Sum']):
        return self.summation_class(*self, *other)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def info(self) -> Info:
        return Info(header = self.__class__.__name__)


class Sum(Summand):
    """
    A class that represents a sum of Summands.

    Calls to __call__ are passed to the contained Summands and then added together and returned.
    """

    container_name = 'summands'

    def __init__(self, *summands, **kwargs):
        setattr(self, self.container_name, summands)
        super().__init__(**kwargs)

    @property
    def _container(self):
        return getattr(self, self.container_name)

    def __str__(self):
        return '({})'.format(' + '.join([str(s) for s in self._container]))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join([repr(p) for p in self._container]))

    def __iter__(self):
        yield from self._container

    def __getitem__(self, item):
        return self._container[item]

    def __add__(self, other: Union[Summand, 'Sum']):
        """Return a new Sum, constructed from all of the contents of self and other."""
        return self.__class__(*self, *other)

    def __call__(self, *args, **kwargs):
        return sum(x(*args, **kwargs) for x in self._container)

    def info(self) -> Info:
        info = super().info()

        for x in self._container:
            info.add_info(x.info())

        return info
