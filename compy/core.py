import datetime as dt
import logging

from compy import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Specification(utils.Beet):
    """
    A class that contains the information necessary to run a simulation.

    It should be subclassed for each type of simulation and all additional information necessary to run that kind of simulation should be added via keyword arguments.
    """

    def __init__(self, name, file_name = None, simulation_type = None, **kwargs):
        """
        Construct a Specification.

        Any number of additional keyword arguments can be passed. They will be stored in an attribute called extra.

        :param name: the internal name of the Specification
        :param file_name: the desired external name, used for pickling. Illegal characters are stripped before use.
        :param kwargs: extra arguments, stored in Specification.extra
        """
        super(Specification, self).__init__(name, file_name = file_name)

        self.simulation_type = simulation_type
        self.extra_args = kwargs

    def save(self, target_dir = None, file_extension = '.spec'):
        """
        Atomically pickle the Specification to {target_dir}/{self.file_name}.{file_extension}, and gzip it for reduced disk usage.

        :param target_dir: directory to save the Specification to
        :param file_extension: file extension to name the Specification with
        :return: None
        """
        super(Specification, self).save(target_dir, file_extension)

    def to_simulation(self):
        return self.simulation_type(self)


class Simulation(utils.Beet):
    """
    A class that represents a simulation.

    It should be subclassed and customized for each variety of simulation.
    Ideally, actual computation should be handed off to another object, while the Simulation itself stores the data produced by that object.
    """

    def __init__(self, spec, initial_status = 'initialized'):
        """
        Construct a Simulation from a Specification.

        :param spec: the Specification for the Simulation
        :param initial_status: an initial status for the simulation, defaults to 'initialized'
        """
        self.spec = spec
        self.status = initial_status

        super(Simulation, self).__init__(spec.name, file_name = spec.file_name)  # inherit name and file_name from spec
        self.spec.simulation_type = self.__class__

        # diagnostic data
        self.evictions = 0
        self.start_time = dt.datetime.now()
        self.end_time = None
        self.elapsed_time = None
        self.latest_load_time = dt.datetime.now()
        self.run_time = dt.timedelta()

    def save(self, target_dir = None, file_extension = '.sim'):
        """
        Atomically pickle the Simulation to {target_dir}/{self.file_name}.{file_extension}, and gzip it.

        :param target_dir: directory to save the Simulation to
        :param file_extension: file extension to name the Simulation with
        :return: None
        """
        if self.status != 'finished':
            self.run_time += dt.datetime.now() - self.latest_load_time
            self.latest_load_time = dt.datetime.now()

        return super(Simulation, self).save(target_dir, file_extension)

    @classmethod
    def load(cls, file_path):
        """
        Load a Simulation from file_path.

        :param file_path: the path to try to load a Simulation from
        :return: the loaded Simulation
        """
        sim = super(Simulation, cls).load(file_path)

        sim.latest_load_time = dt.datetime.now()

        return sim

    def __str__(self):
        return '{}: {} ({}) [{}]  |  {}'.format(self.__class__.__name__, self.name, self.file_name, self.uid, self.spec)

    def __repr__(self):
        return '{}(spec = {}, uid = {})'.format(self.__class__.__name__, repr(self.spec), self.uid)

    def run_simulation(self):
        raise NotImplementedError

    def info(self):
        diag = ['Status: {}'.format(self.status),
                '   Start Time: {}'.format(self.start_time),
                '   Latest Load Time: {}'.format(self.latest_load_time),
                '   End Time: {}'.format(self.end_time),
                '   Elapsed Time: {}'.format(self.elapsed_time),
                '   Run Time: {}'.format(self.run_time)]

        return '\n'.join((str(self), *diag, self.spec.info()))
