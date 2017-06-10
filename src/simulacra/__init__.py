__all__ = ['core', 'math', 'utils', 'units', 'vis']

import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

import matplotlib


matplotlib.use('Agg')

mpl_rcParams_update = {
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    'xtick.top': True,
    'xtick.bottom': True,
    'ytick.right': True,
    'ytick.left': True,
}

matplotlib.rcParams.update(mpl_rcParams_update)

import numpy as _np
_np.set_printoptions(linewidth = 200)  # screw character limits

from simulacra.core import *
from simulacra import math, utils, vis
