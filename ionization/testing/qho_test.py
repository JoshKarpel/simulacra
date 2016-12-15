import logging
import os

import numpy as np
import scipy.sparse as sparse

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        pot = ion.HarmonicOscillatorPotential.from_energy_and_mass()
        mass = electron_mass
        sim = ion.LineSpecification('qho',
                                    internal_potential = pot,
                                    test_mass = mass,
                                    test_states = (ion.QHOState(omega = pot.omega(mass), mass = mass, n = n) for n in range(10)),
                                    dipole_gauges = (),
                                    initial_state = ion.QHOState(omega = pot.omega(mass), mass = mass, n = 1),
                                    time_initial = 0, time_final = 100 * fsec, time_step = 1 * fsec,
                                    ).to_simulation()

        print(sim.mesh.norm)

        sim.run_simulation()

        print(sim.mesh.norm)
        print(sim.energy_expectation_value_vs_time_internal / eV)
