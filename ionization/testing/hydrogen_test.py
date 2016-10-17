import logging
import os
import random
import sys

import compy as cp

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

    print(hyd.BoundState(1, 0, 0) < hyd.BoundState(3, 2, 1))
    print(hyd.BoundState(1, 0, 0) <= hyd.BoundState(3, 2, 1))
    print(hyd.BoundState(3, 2, 1) > hyd.BoundState(3, 2, 0))
    print(hyd.BoundState(3, 2, 1) >= hyd.BoundState(3, 2, 0))

    sortable = [hyd.BoundState(n, l, m) for n in range(4) for l in range(n) for m in range(-l, l + 1)]
    random.shuffle(sortable)

    sort = sorted(sortable, key = hyd.BoundState.sort_key)

    for state, s in zip(sortable, sort):
        s1 = '{}'.format(state).rjust(10)
        s2 = '{}'.format(s).ljust(10)
        print(s1 + ' | ' + s2)

