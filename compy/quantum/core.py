import logging
from copy import deepcopy

import numpy as np

import compy.units as un

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class IllegalQuantumState(Exception):
    pass


def electron_energy_from_wavenumber(k):
    return (un.hbar * k) ** 2 / (2 * un.electron_mass)


def electron_wavenumber_from_energy(energy):
    return np.sqrt(2 * un.electron_mass * energy) / un.hbar


class QuantumMesh:
    def __init__(self, parameters, simulation):
        self.parameters = parameters
        self.simulation = simulation

    def __str__(self):
        return '{} for {}'.format(self.__class__.__name__, str(self.simulation))

    def __repr__(self):
        return '{}(parameters = {}, simulation = {})'.format(self.__class__.__name__, repr(self.parameters), repr(self.simulation))

    @property
    def norm(self):
        raise NotImplementedError

    def __abs__(self):
        return self.norm

    def copy(self):
        return deepcopy(self)