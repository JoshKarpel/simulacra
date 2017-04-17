import matplotlib

matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend

import os
import argparse
import logging
import datetime as dt
import socket

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
                         file_logs = True, file_level = logging.INFO, file_name = '{}'.format(args.sim_name), file_mode = 'a') as log:
        try:
            log.info('Loaded onto execute node {} at {}.'.format(socket.gethostname(), dt.datetime.now()))
            log.debug('Local directory contents: {}'.format(os.listdir(os.getcwd())))

            # try to find existing checkpoint, and start from scratch if that fails
            try:
                sim_path = os.path.join(os.getcwd(), '{}.sim'.format(args.sim_name))
                sim = cp.Simulation.load(sim_path)
                log.info('Checkpoint found at {}, recovered simulation {}'.format(sim_path, sim))
                log.debug('Checkpoint size is {}'.format(cp.utils.get_file_size_as_string(sim_path)))
            except (FileNotFoundError, EOFError):
                sim = cp.Specification.load(os.path.join(os.getcwd(), '{}.spec'.format(args.sim_name))).to_simulation()
                log.info('Checkpoint not found, started simulation {}'.format(sim))

            # run the simulation and save it
            log.info(sim.info())

            sim.run_simulation()

            log.info(sim.info())

            sim.save()
        except Exception as e:
            log.exception(e)
            raise e
