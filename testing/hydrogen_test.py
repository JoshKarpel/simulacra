import os
import sys
import logging

import compy as cp
from compy.quantum.core import IllegalQuantumState
import compy.quantum.hydrogenic as hyd

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

cp.utils.ensure_dir_exists(OUT_DIR)

logger = logging.getLogger('compy')
logger.setLevel(logging.DEBUG)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(cp.utils.LOG_FORMATTER)
logger.addHandler(stdout_handler)

if __name__ == '__main__':
    bs = hyd.BoundState(5, 0)
    logger.info(bs)
    logger.info(repr(bs))

    bss = hyd.BoundStateSuperposition({hyd.BoundState(n, l, 0): 1 for n in range(3) for l in range(n)})

    logger.info(bss)
    logger.info(repr(bss))

    try:
        bs2 = hyd.BoundState(2, 3, 5)
    except IllegalQuantumState as err:
        logger.exception('expected excepted')
