import logging

import numpy as np

import compy.units as un

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def photon_wavelength_from_frequency(frequency):
    return un.c / frequency