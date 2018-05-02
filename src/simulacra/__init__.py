import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

import os

# see https://github.com/ContinuumIO/anaconda-issues/issues/905
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import matplotlib

matplotlib.use('Agg')

from .sims import *
from .info import Info
from . import math, utils, vis, cluster, units, summables, exceptions
