import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        sim = ion.SphericalHarmonicSpecification('speed_test',
                                                 r_bound = 100 * bohr_radius,
                                                 r_points = 800, l_bound = 300,
                                                 test_states = (), use_numeric_eigenstates_as_basis = False,
                                                 time_initial = 0, time_final = 1000 * asec, time_step = 1 * asec,
                                                 dipole_gauges = (),
                                                 store_data_every = -1,
                                                 ).to_simulation()

        logger.info(sim.info())
        with si.utils.BlockTimer() as timer:
            sim.run_simulation()
        logger.info(sim.info())

        mesh_points = sim.mesh.mesh_points
        time_points = sim.time_steps - 1
        space_time_points = mesh_points * time_points

        logger.info(f'Number of Space Points: {mesh_points}')
        logger.info(f'Number of Time Points: {time_points}')
        logger.info(f'Number of Space-Time Points: {space_time_points}')
        logger.info(f'Sim Runtime: {sim.running_time}')
        logger.info(f'BlockTimer: {timer}')

        logger.info(f'Space-Time Points / Runtime (according to Sim): {round(space_time_points / sim.running_time.total_seconds())}')
        logger.info(f'Space-Time Points / Runtime (according to BlockTimer): {round(space_time_points / timer.proc_time_elapsed)}')
