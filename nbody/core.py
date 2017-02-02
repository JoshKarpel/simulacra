import datetime as dt
import functools
import logging
import itertools as it

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

import compy as cp
from compy.units import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Particle:
    def __init__(self, position, velocity, mass, fixed = False):
        self.mass = mass
        self.position = position
        self.velocity = velocity

        self.fixed = False


class ChargedParticle(Particle):
    def __init__(self, position, velocity, mass, charge, fixed = False):
        super().__init__(position, velocity, mass, fixed = fixed)

        self.charge = charge


class NBodySpecification(cp.core.Specification):
    raise NotImplementedError


class NBodySimulation(cp.core.Simulation):
    raise NotImplementedError
