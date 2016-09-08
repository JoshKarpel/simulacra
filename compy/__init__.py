import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

from compy import core, math, units, utils

# set up platform-independent runtime cython compilation and imports
import numpy as np
import pyximport
pyx_dir = os.path.join(os.path.dirname(__file__), '.pyxbld')
pyximport.install(setup_args = {"include_dirs": np.get_include()},
                  build_dir = pyx_dir,
                  language_level = 3)
