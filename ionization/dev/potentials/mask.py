import logging
import os

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.LogManager('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG, file_dir = OUT_DIR, file_logs = False) as logger:
        electric = ion.Rectangle(start_time = 25 * asec, end_time = 100 * asec, amplitude = 1 * atomic_electric_field)

        mask = ion.RadialCosineMask(inner_radius = 40 * bohr_radius, outer_radius = 49 * bohr_radius)
        sim = ion.SphericalHarmonicSpecification('mask',
                                                 time_final = 200 * asec,
                                                 r_bound = 50 * bohr_radius, r_points = 50 * 8,
                                                 l_bound = 100,
                                                 electric_potential = electric,
                                                 test_states = [ion.HydrogenBoundState(n, l) for n in range(6) for l in range(n)],
                                                 mask = mask).to_simulation()

        sim.run_simulation()
        logger.info(sim.info())
        print(sim.info())

        print(sim.mesh.norm)
        print(sim.mesh.state_overlap(sim.spec.initial_state))

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)
        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR, log = True)
