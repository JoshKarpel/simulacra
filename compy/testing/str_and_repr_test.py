import logging
import os

import compy as cp
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', stdout_logs = True, stdout_level = logging.DEBUG, file_dir = OUT_DIR, file_logs = False) as logger:
        print(logger)
        print(repr(logger))

        b = cp.utils.Beet('hi')
        print(b)
        print(repr(b))

        s = cp.Specification('spec', test = 1000 * m)

        print(cp.utils.field_str(s, 'name', ('test', 'km')))
