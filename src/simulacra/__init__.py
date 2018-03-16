import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

import os

# see https://github.com/ContinuumIO/anaconda-issues/issues/905
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

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

from .sims import *
from .info import Info
from . import math, utils, vis, units, summables, exceptions
