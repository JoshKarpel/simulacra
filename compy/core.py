import logging

from compy import utils


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Parameters(utils.Beet):
    def __init__(self, name, file_name = None):
        super(Parameters, self).__init__(name, file_name = file_name)

    def save(self, target_dir = None, file_extension = '.par'):
        super(Parameters, self).save(target_dir, file_extension)


class Simulation(utils.Beet):
    def __init__(self, name, file_name = None):
        super(Simulation, self).__init__(name, file_name = file_name)

        self.status = 'initialized'

    def save(self, target_dir = None, file_extension = '.sim'):
        super(Simulation, self).save(target_dir, file_extension)

    def run_simulation(self):
        raise NotImplementedError
