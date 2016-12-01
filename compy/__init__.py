__all__ = ['core', 'math', 'utils', 'units']

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

import matplotlib

matplotlib.use('agg')

matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt

plt.set_cmap(plt.cm.inferno)

# TODO: use latex to get siunitx for correct unit formatting (but what about cluster plotting? not that I ever actually do that...)

# set up platform-independent runtime cython compilation and imports
import numpy as np
import pyximport
pyx_dir = os.path.join(os.path.dirname(__file__), '.pyxbld')
pyximport.install(setup_args = {"include_dirs": np.get_include()},
                  build_dir = pyx_dir,
                  language_level = 3)

np.set_printoptions(linewidth = 200)

from compy.core import *
from compy import math, utils, cluster
