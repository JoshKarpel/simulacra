import datetime
import gzip
import pickle
import uuid
import collections
from copy import deepcopy

import logging
import os

from . import utils


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SimulacraException(Exception):
    """Base exception for all Simulacra exceptions."""
    pass


class Info:
    """
    A class that represents a hierarchical collection of information.

    Each :class:`Info` contains a header and a dictionary of children.
    The header is a string that will be written at the top-level of this Info.
    Each child is either a field, which will be written out as "{key}: {value}", or another Info, which will display itself.
    """
    def __init__(self, *,
                 header: str,
                 indentation: int = 2):
        self.header = header
        self.indentation = indentation

        self.children = collections.OrderedDict()

    def _field_strs(self):
        s = []
        for field, value in self.children.items():
            try:
                s.extend(value._field_strs())
            except AttributeError:
                s.append(f'{field}: {value}')
        s = [self.header] + [' ' * self.indentation + f for f in s]

        return s

    def __str__(self):
        field_strs = self._field_strs()
        return f'\n{field_strs[0]}\n' + '\n'.join('|' + s[1:] for s in field_strs[1:])

    def __repr__(self):
        return f'{self.__class__.__name__}({self.header})'

    def add_field(self, name, value):
        self.children[name] = value

    def add_fields(self, name_value_pairs):
        self.children.update({k: v for k, v in name_value_pairs})

    def add_info(self, info):
        self.children[id(info)] = info

    def add_infos(self, infos):
        self.children.update({id(info): info for info in infos})


class Beet:
    """
    A class that provides an easy interface for pickling and unpickling instances.

    Two Beets compare and hash equal if they have the same uid attribute, a uuid4 generated during initialization.
    """

    def __init__(self, name, file_name = None):
        """
        Construct a Beet with the given name and file_name.

        The file_name is automatically derived from the name if None is given.

        Parameters
        ----------
        name : :class:`str`
            The internal name of the Beet.
        file_name : :class:`str`
            The desired external name of the Beet. Illegal characters are stripped before use.
        """
        self.name = str(name)
        if file_name is None:
            file_name = self.name

        file_name_stripped = utils.strip_illegal_characters(str(file_name))
        if file_name_stripped != file_name:
            logger.warning('Using file name {} instead of {} for {}'.format(file_name_stripped, file_name, self.name))
        self.file_name = file_name_stripped

        self.initialized_at = datetime.datetime.utcnow()
        self.uid = uuid.uuid4()

        logger.info('Initialized {}'.format(repr(self)))

    def __str__(self):
        if self.name != self.file_name:
            return f'{self.__class__.__name__}({self.name}, file_name = {self.file_name}) [{self.uid}]'
        else:
            return f'{self.__class__.__name__}({self.name}) [{self.uid}]'

    def __repr__(self):
        return utils.field_str(self, 'name', 'file_name', 'uid')

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)

    def clone(self, **kwargs):
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

    def save(self, target_dir = None, file_extension = '.beet', compressed = True):
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

        Returns
        -------
        :class:`str`
            The path to the saved Beet.
        """
        if target_dir is None:
            target_dir = os.getcwd()

        file_path = os.path.join(target_dir, self.file_name + file_extension)
        file_path_working = file_path + '.working'

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
    def load(cls, file_path):
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
        except OSError:
            with open(file_path, mode = 'rb') as file:
                beet = pickle.load(file)

        logger.debug('Loaded {} {} from {}'.format(beet.__class__.__name__, beet.name, file_path))

        return beet

    def info(self):
        return Info(header = str(self))


class Specification(Beet):
    """
    A class that contains the information necessary to run a simulation.

    It should be subclassed for each type of simulation and all additional information necessary to run that kind of simulation should be added via keyword arguments.
    """

    simulation_type = None

    def __init__(self, name, file_name = None, **kwargs):
        """
        Construct a Specification.

        Any number of additional keyword arguments can be passed. They will be stored as attributes if they don't conflict with any attributes already set.

        Parameters
        ----------
        name : :class:`str`
            The internal name of the Specification.
        file_name : :class:`str`
        kwargs
            Any number of keyword arguments, which will be stored as attributes.
        """
        super().__init__(name, file_name = file_name)

        self._extra_attr_keys = list()

        for k, v in ((k, v) for k, v in kwargs.items() if k not in self.__dict__):
            setattr(self, k, v)
            self._extra_attr_keys.append(k)
            logger.debug('{} stored additional attribute {} = {}'.format(self.name, k, v))

    def save(self, target_dir = None, file_extension = '.spec', compressed = True):
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
        return super().save(target_dir = target_dir, file_extension = file_extension, compressed = compressed)

    def to_simulation(self):
        """Return a Simulation of the type associated with the Specification, generated from this instance."""
        try:
            return self.simulation_type(self)
        except TypeError:
            return Simulation(self)

    def info(self):
        """Return a string describing the parameters of the Simulation and its associated Specification."""
        info = super().info()

        if len(self._extra_attr_keys) > 0:
            info_extra = Info(header = f'Extra Attributes')

            for k in self._extra_attr_keys:
                info_extra.add_field(' '.join(k.split('_').title()), getattr(self, k))

            info.add_info(info_extra)

        return info


# Simulation status names
STATUS_INI = 'initialized'
STATUS_RUN = 'running'
STATUS_FIN = 'finished'
STATUS_PAU = 'paused'
STATUS_ERR = 'error'


class Simulation(Beet):
    """
    A class that represents a single simulation.

    It should be subclassed and customized for each variety of simulation.

    Attributes
    ----------
    status : :class:`str`
        The status of the Simulation. One of ``'initialized'``, ``'running'``, ``'finished'``, ``'paused'``, or ``'error'``.
    """

    _status = utils.RestrictedValues('status', {'', STATUS_INI, STATUS_RUN, STATUS_FIN, STATUS_PAU, STATUS_ERR})

    def __init__(self, spec):
        """
        Construct a Simulation from a Specification.

        Simulations should generally be instantiated using Specification.to_simulation() to avoid possible mismatches.

        Parameters
        ----------
        spec : :class:`Specification`
            The :class:`Specification` for the Simulation.
        """
        self.spec = spec

        super().__init__(spec.name, file_name = spec.file_name)  # inherit name and file_name from spec

        # diagnostic data
        self.runs = 0
        self.init_time = None
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.latest_run_time = None
        self.running_time = datetime.timedelta()

        self._status = ''
        self.status = STATUS_INI

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        """
        Set the status of the :class:`Simulation`.

        Specially defined statuses are ``STATUS_INI`` (initialized), ``STATUS_RUN`` (running), ``STATUS_FIN`` (finished), ``STATUS_PAU`` (paused), and ``STATUS_ERR`` (error).
        These statuses have side effects on the simulation's time diagnostics.

        Parameters
        ----------
        status : :class:`str`
            The new status for the simulation
        """
        if status == self.status:
            raise ValueError('Tried to set status of {} to its current status'.format(self.name))

        now = datetime.datetime.utcnow()

        if status == STATUS_INI:
            self.init_time = now
        elif status == STATUS_RUN:
            if self.latest_run_time is None:
                self.start_time = now
            self.latest_run_time = now
            self.runs += 1
        elif status == STATUS_PAU:
            if self.latest_run_time is not None:
                self.running_time += now - self.latest_run_time
        elif status == STATUS_FIN:
            if self.latest_run_time is not None:
                self.running_time += now - self.latest_run_time
            self.end_time = now
            self.elapsed_time = self.end_time - self.init_time

        self._status = status

        logger.debug("{} {} ({}) status set to {}".format(self.__class__.__name__, self.name, self.file_name, status))

    def __str__(self):
        return super().__str__() + f' ~ {self.status}'

    def save(self, target_dir = None, file_extension = '.sim', compressed = True):
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
        if self.status != STATUS_FIN:
            self.status = STATUS_PAU

        return super().save(target_dir = target_dir, file_extension = file_extension, compressed = compressed)

    def run_simulation(self):
        """Hook method for running the Simulation, whatever that may entail."""
        raise NotImplementedError

    def info(self):
        """Return a string describing the parameters of the Simulation and its associated Specification."""
        info = super().info()

        info.add_info(self.spec.info())

        info_diag = Info(header = f'Status: {self.status}')
        info_diag.add_field('Start Time', self.init_time)
        info_diag.add_field('Latest Run Time', self.latest_run_time)
        info_diag.add_field('End Time', self.end_time)
        info_diag.add_field('Elapsed Time', self.elapsed_time)
        info_diag.add_field('Run Time', self.running_time)
        info.add_info(info_diag)

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

    def __add__(self, other):
        return self.summation_class(*self, *other)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def info(self):
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

    def __add__(self, other):
        """Return a new Sum, constructed from all of the contents of self and other."""
        return self.__class__(*self, *other)  # TODO: no protection against adding together non-similar types

    def __call__(self, *args, **kwargs):
        return sum(x(*args, **kwargs) for x in self._container)

    def info(self):
        info = super().info()

        for x in self._container:
            try:
                info.add_info(x.info())
            except AttributeError:
                info.add_field(x.__class__.__name__, str(x))

        return info
