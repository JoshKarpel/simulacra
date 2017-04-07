__all__ = ['core', 'math', 'utils', 'units']

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

import matplotlib as mpl

mpl.use('Agg')

mpl_rcParams_update = {
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    'xtick.top': True,
    'xtick.bottom': True,
    'ytick.right': True,
    'ytick.left': True,
}

mpl.rcParams.update(mpl_rcParams_update)

import matplotlib.pyplot as _plt

_plt.set_cmap(_plt.cm.inferno)  # set default colormap

# set up platform-independent runtime cython compilation and imports
import numpy as _np
import pyximport

pyx_dir = os.path.join(os.path.dirname(__file__), '.pyxbld')
pyximport.install(setup_args = {"include_dirs": _np.get_include()},
                  build_dir = pyx_dir,
                  language_level = 3)

_np.set_printoptions(linewidth = 200)  # screw character limits

from compy.core import *
from compy import math, utils, cluster
