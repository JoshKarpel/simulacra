import matplotlib

matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend

import os
import argparse
import logging.handlers
import datetime as dt
import socket

import compy as cp

if __name__ == '__main__':
    # get the sim name
    parser = argparse.ArgumentParser(description = 'Run a simulation.')
    parser.add_argument('sim_name',
                        type = str,
                        help = 'the name of the sim')

    args = parser.parse_args()

    with cp.utils.Logger(stdout_logs = False, file_logs = True, file_level = logging.DEBUG, file_name = '{}.log'.format(args.sim_name)) as logger:
        try:
            logger.info('Loaded onto execute node {} at {}.'.format(socket.gethostname(), dt.datetime.now()))
            logger.info('Local directory contents: {}'.format(os.listdir(os.getcwd())))

            # try to find existing checkpoint, and start from scratch if that fails
            try:
                sim = cp.Simulation.load(os.path.join(os.getcwd(), '{}.sim'.format(args.sim_name)))
                logger.info('Checkpoint found, recovering simulation')
            except FileNotFoundError:
                spec = cp.Specification.load(os.path.join(os.getcwd(), '{}.spec'.format(args.sim_name)))
                sim = spec.to_simulation()
                logger.info('Checkpoint not found, beginning simulation')

            # run the simulation and save it
            logger.info(sim.info())
            sim.run_simulation()
            logger.info(sim.info())
            sim.save()
        except Exception as e:
            logger.exception(e)
