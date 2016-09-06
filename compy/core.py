import logging

from compy import utils


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Parameters(utils.Beet):
    """
    A class that contains the information necessary to run a simulation.

    It should be subclassed for each type of simulation and all additional information necessary to run that kind of simulation should be added via keyword arguments.
    """
    def __init__(self, name, file_name = None):
        super(Parameters, self).__init__(name, file_name = file_name)

    def save(self, target_dir = None, file_extension = '.par'):
        super(Parameters, self).save(target_dir, file_extension)


class Simulation(utils.Beet):
    """
    A class that represents a simulation.

    It should be subclassed and customized for each variety of simulation. Ideally, actual computation should be handed off to another object, while the Simulation simply stores the data produced by that object.
    """

    def __init__(self, parameters, initial_status = 'initialized'):
        """
        Construct a Simulation from a Parameters.

        :param parameters: the Parameters for the Simulation
        :param initial_status: an initial status for the simulation, defaults to 'initialized'
        """

        self.parameters = parameters
        self.status = initial_status

        super(Simulation, self).__init__(parameters.name, file_name = parameters.file_name)  # inherit name and file_name from parameters

    def save(self, target_dir = None, file_extension = '.sim'):
        super(Simulation, self).save(target_dir, file_extension)

    def __str__(self):
        return '{}: {} ({}) [{}] | {}'.format(self.__class__.__name__, self.name, self.file_name, self.uid, str(self.parameters))

    def __repr__(self):
        return '{}(parameters = {}, uid = {})'.format(self.__class__.__name__, repr(self.parameters), self.uid)

    def run_simulation(self):
        raise NotImplementedError
