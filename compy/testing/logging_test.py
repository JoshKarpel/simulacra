import logging
import os
import sys

import compy.quantum.hydrogenic as hyd

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

file_handler = logging.FileHandler(os.path.join(OUT_DIR, '{}.log'.format(FILE_NAME)), mode = 'w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(cp.utils.LOG_FORMATTER)
logger.addHandler(file_handler)

if __name__ == '__main__':
    logger.info('hi from script')

    par = cp.core.Specification('test', file_name = 'test_file_name')
    logger.info(par)
    logger.info(repr(par))

    par.save(target_dir = OUT_DIR)

    par2 = cp.core.Specification.load(os.path.join(OUT_DIR, 'test_file_name.par'))

    # cp.utils.ask_for_input('who are you?', cast_to = str)

    sim = cp.core.Simulation(par)
    logger.critical(sim)
    logger.critical(repr(sim))

    h = hyd.ElectricFieldSimulation(par)
    h.save(target_dir = OUT_DIR)
    h2 = hyd.ElectricFieldSimulation.load(os.path.join(OUT_DIR, 'test_file_name.sim'))

    logger.critical(h2)

    try:
        raise ValueError
    except ValueError as err:
        logger.exception('ouch')

    print(logger.handlers)

    with cp.utils.Logger(file_logs = True, file_dir = OUT_DIR) as withlogger:
        print(withlogger.handlers)
        print(withlogger.handlers)
        withlogger.info('im a fancy logger. should print only once')

    print(logger.handlers)

    logger.debug('only once as well')
