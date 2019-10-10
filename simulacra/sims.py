import logging
from typing import Optional, Type, Set, Any

import datetime
import gzip
import pickle
import uuid
from copy import deepcopy
from pathlib import Path
import abc
import functools

from .info import Info

from . import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Beet:
    """
    A class that provides an easy interface for pickling and unpickling
    instances.

    Beets can be compared and hashed based on their :attr:`Beet.uuid` value.
    """

    file_extension = "beet"

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name
            The name of the :class:`Beet`.
        """
        self.name = str(name)
        self._uuid = uuid.uuid4()

    @property
    def uuid(self) -> uuid.UUID:
        """A unique identifier for the :class:`Beet`."""
        return self._uuid

    def __eq__(self, other: Any):
        if not isinstance(other, Beet):
            return NotImplemented
        return isinstance(other, self.__class__) and self._uuid == other._uuid

    def __hash__(self):
        return hash((self.__class__, self._uuid))

    def clone(self, **kwargs) -> "Beet":
        """
        Return a deepcopy of the :class:`Beet`.

        If any kwargs are passed, they will be interpreted as key-value pairs
        and ``clone`` will try to :func:`setattr` them on the new Beet.

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
        new_beet._uuid = uuid.uuid4()

        return new_beet

    def save(self, target_dir: Optional[Path] = None) -> Path:
        """
        Atomically save the :class:`Beet` to a file.

        Parameters
        ----------
        target_dir
            The directory to save the Beet to.

        Returns
        -------
        path :
            The path to the saved :class:`Beet`.
        """
        target_dir = Path(target_dir or Path.cwd())
        final_path = target_dir.absolute() / f"{self.name}.{self.file_extension}"
        working_path = final_path.with_name(f"{final_path.name}.working")

        utils.ensure_parents_exist(working_path)

        with gzip.open(working_path, mode="wb") as file:
            pickle.dump(self, file, protocol=-1)

        working_path.replace(final_path)

        logger.debug(f"Saved {self} to {final_path}")

        return final_path

    @classmethod
    def load(cls, path: Path) -> "Beet":
        """
        Load a Beet from `file_path`.

        Parameters
        ----------
        path
            The path to load a Beet from.

        Returns
        -------
        beet :
            The loaded Beet.
        """
        with gzip.open(path, mode="rb") as file:
            beet = pickle.load(file)

        logger.debug(f"Loaded {beet} from {path}")

        return beet

    def __repr__(self):
        return f"{self.__class__.__name__}(name = {self.name}, uuid = {self._uuid})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

    def info(self) -> Info:
        """Return the :class:`Info` for the :class:`Beet`."""
        info = Info(header=str(self))
        info.add_field("UUID", self._uuid)

        return info


class Status(utils.StrEnum):
    """
    An enumeration of the possible states of a :class:`Simulation`.
    """

    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    RUNNING = "running"
    FINISHED = "finished"
    PAUSED = "paused"
    ERROR = "error"


class Simulation(Beet, abc.ABC):
    """
    A class that represents a single simulation.

    It should be subclassed and customized for each variety of simulation.

    Simulations should generally be instantiated using
    :meth:`Specification.to_sim` to avoid possible mismatches.
    """

    file_extension = "sim"

    def __init__(self, spec):
        """
        Parameters
        ----------
        spec
            The :class:`Specification` for the Simulation.
        """
        super().__init__(spec.name)
        self.spec = spec

        self.runs: int = 0
        self.init_time: Optional[datetime.datetime] = None
        self.start_time: Optional[datetime.datetime] = None
        self.end_time: Optional[datetime.datetime] = None
        self.elapsed_time: Optional[datetime.timedelta] = None
        self.latest_run_time: Optional[datetime.datetime] = None
        self.running_time: datetime.timedelta = datetime.timedelta()

        self._status = Status.UNINITIALIZED
        self.status = Status.INITIALIZED

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """
        This method handles adding a decorator around subclass run methods
        that implements the correct status behavior for them.
        It is called automatically during subclass creation; no user code is
        necessary.
        """

        def wrapper(func):
            @functools.wraps(func)
            def wrapped(self, *args, **kwargs):
                self.status = Status.RUNNING
                out = func(self, *args, **kwargs)
                self.status = Status.FINISHED
                return out

            return wrapped

        cls.run = wrapper(cls.run)

        return cls

    @property
    def status(self) -> Status:
        """The current state of the simulation."""
        return self._status

    @status.setter
    def status(self, new_status: Status):
        """
        Set the status of the :class:`Simulation`.

        Setting the statuses can have side effects on the simulation's time
        diagnostics.

        Parameters
        ----------
        new_status
            The new status for the :class:`Simulation`.
        """
        if not isinstance(new_status, Status):
            raise TypeError(f"{new_status} is not a member of Status")

        if new_status is self.status:
            return

        old_status = self.status
        now = datetime.datetime.utcnow()

        if new_status == Status.INITIALIZED:
            self.init_time = now
        elif new_status == Status.RUNNING:
            if self.start_time is None:
                self.start_time = now
            self.latest_run_time = now
            self.runs += 1
        elif new_status == Status.PAUSED:
            if self.latest_run_time is not None:
                self.running_time += now - self.latest_run_time
        elif new_status == Status.FINISHED:
            if self.latest_run_time is not None:
                self.running_time += now - self.latest_run_time
            self.end_time = now
            self.elapsed_time = self.end_time - self.init_time

        self._status = new_status

        logger.debug(f"{self} status changed to {self.status} from {old_status}")

    @abc.abstractmethod
    def run(self, **kwargs):
        """Hook method for running the Simulation, whatever that may entail."""
        raise NotImplementedError

    def __repr__(self):
        return f"{super().__repr__()} {{{self.status}}}"

    def info(self) -> Info:
        """Return the :class:`Info` for the :class:`Simulation`."""
        info = super().info()

        info.add_field("UUID", self._uuid)

        info_diag = Info(header=f"Status: {self.status}")
        info_diag.add_field("Initialization Time", self.init_time)
        info_diag.add_field("Latest Run Time", self.latest_run_time)
        info_diag.add_field("End Time", self.end_time)
        info_diag.add_field("Running Time", self.running_time)
        info_diag.add_field("Elapsed Time", self.elapsed_time)

        info.add_infos(info_diag, self.spec.info())

        return info


class Specification(Beet, abc.ABC):
    """
    A class that contains the information necessary to run a simulation.

    It should be subclassed for each type of simulation and all additional
    information necessary to run that kind of simulation should be added via
    keyword arguments.

    Any number of additional keyword arguments can be passed to the constructor.
    They will be stored as attributes if they don't conflict with any attributes
    already set.

    Attributes
    ----------
    simulation_type
        A class attribute which determines what kind of :class:`Simulation` will
        be generated via :meth:`Specification.to_sim`.
        Should be overridden in concrete subclasses.
    """

    file_extension = ".spec"

    simulation_type: Type[Simulation] = Simulation

    def __init__(self, name: str, **kwargs):
        """
        Parameters
        ----------
        name
            The internal name of the Specification.
        kwargs
            Any number of keyword arguments, which will be stored as attributes.
        """
        super().__init__(name)

        self._extra_attr_keys: Set[str] = set()

        for k, v in ((k, v) for k, v in kwargs.items() if k not in self.__dict__):
            setattr(self, k, v)
            self._extra_attr_keys.add(k)
            logger.debug("{} stored additional attribute {} = {}".format(self, k, v))

    def to_sim(self) -> Simulation:
        """
        Return a :class:`Simulation` of the concrete subtype associated with
        the :class:`Specification` type, generated from this instance.
        """
        return self.simulation_type(self)

    def info(self) -> Info:
        """Return the :class:`Info` for the :class:`Specification`."""
        info = super().info()

        if len(self._extra_attr_keys) > 0:
            info_extra = Info(header=f"Extra Attributes")

            for k in self._extra_attr_keys:
                info_extra.add_field(k, getattr(self, k))

            info.add_info(info_extra)

        return info


def find_sim_or_init(
    spec: Specification, search_dir: Optional[Path] = None
) -> Simulation:
    """
    Try to load a :class:`simulacra.Simulation` by looking for a pickled
    :class:`simulacra.core.Simulation` named
    ``{search_dir}/{spec.file_name}.sim``.
    If that fails, create a new Simulation from `spec`.

    Parameters
    ----------
    spec
        The filename of this specification is what will be searched for.
    search_dir
        The directory to look for the simulation in.
    Returns
    -------
    sim
        The simulation, either loaded or initialized.
    """
    search_dir = Path(search_dir or Path.cwd())
    path = search_dir / f"{spec.name}.{SIM_FILE_EXTENSION}"
    try:
        sim = Simulation.load(path=path)
    except FileNotFoundError:
        sim = spec.to_sim()

    return sim


def run_from_cache(
    spec: Specification, cache_dir: Optional[Path] = None, **kwargs
) -> Simulation:
    """
    Runs simulations from a cache, which is a directory where simulations are
    stored. If a simulation with the same name as the ``spec`` is in the cache,
    it will be loaded, run if it is not finished, and returned. If not, a new
    simulation will be created using the ``spec`` and run.

    Parameters
    ----------
    spec
        The specification to use as a template for the simulation.
    cache_dir
        The path to the cache directory.
    kwargs
        Additional keyword arguments are passed to the simulation's
        :meth:`Simulation.run` method.

    Returns
    -------
    sim
        The completed simulation.
    """
    sim = find_sim_or_init(spec, search_dir=cache_dir)

    if sim.status != Status.FINISHED:
        sim.run(**kwargs)
        sim.save(target_dir=cache_dir)

    return sim
