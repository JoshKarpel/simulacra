"""
Simulacra __init__ file.


Copyright 2017 Josh Karpel

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__all__ = ['core', 'math', 'utils', 'units', 'vis']

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

import os

# see https://github.com/ContinuumIO/anaconda-issues/issues/905
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import matplotlib

try:
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
except Exception as e:
    logger.exception('Failed to set matplotlib options')

import numpy as _np

_np.set_printoptions(linewidth = 200)  # screw character limits

from simulacra.core import *
from simulacra import math, utils, vis
