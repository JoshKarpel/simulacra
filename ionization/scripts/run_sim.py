import matplotlib

matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend

import os
import argparse
import logging
import datetime as dt
import socket
import json

import compy as cp

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run a simulation.')
    parser.add_argument('sim_name',
                        type = str,
                        help = 'the name of the sim')

    args = parser.parse_args()

    with cp.utils.Logger('__main__', 'compy', 'ionization',
                         stdout_logs = False,
                         file_logs = True, file_level = logging.INFO, file_name = '{}'.format(args.sim_name)) as logger:
        try:
            logger.info('Loaded onto execute node {} at {}.'.format(socket.gethostname(), dt.datetime.now()))
            logger.debug('Local directory contents: {}'.format(os.listdir(os.getcwd())))

            # try to find existing checkpoint, and start from scratch if that fails
            try:
                sim_path = os.path.join(os.getcwd(), '{}.sim'.format(args.sim_name))
                sim = cp.Simulation.load(sim_path)
                logger.info('Checkpoint found at {}, recovered simulation {}'.format(sim_path, sim))
                logger.debug('Checkpoint size is {}'.format(cp.utils.get_file_size_as_string(sim_path)))
            except (FileNotFoundError, EOFError):
                sim = cp.Specification.load(os.path.join(os.getcwd(), '{}.spec'.format(args.sim_name))).to_simulation()
                logger.info('Checkpoint not found, started simulation {}'.format(sim))

            # run the simulation and save it
            logger.info(sim.info())
            sim.run_simulation()
            logger.info(sim.info())
            sim.save()
        except Exception as e:
            logger.exception(e)
            raise e
