import logging
from copy import deepcopy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QuantumMesh:
    def __init__(self, parameters, simulation):
        self.parameters = parameters
        self.simulation = simulation

    def __str__(self):
        return '{} for {}'.format(self.__class__.__name__, str(self.simulation))

    def __repr__(self):
        return '{}(parameters = {}, simulation = {})'.format(self.__class__.__name__, repr(self.parameters), repr(simulation))

    @property
    def norm(self):
        raise NotImplementedError

    def __abs__(self):
        return self.norm

    def copy(self):
        return deepcopy(self)