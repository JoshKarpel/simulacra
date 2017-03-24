import os
import logging

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG)

if __name__ == '__main__':
    with log as logger:
        energy_spacing = 1 * eV
        mass = electron_mass

        qho = ion.HarmonicOscillator.from_energy_spacing_and_mass(energy_spacing = energy_spacing, mass = mass)

        initial_state = ion.QHOState.from_QHO_potential_and_mass(qho, mass, n = 0)
        test_states = [ion.QHOState.from_QHO_potential_and_mass(qho, mass, n = n) for n in range(10)]

        efield = ion.SineWave.from_photon_energy(energy_spacing, amplitude = .01 * atomic_electric_field)
        # efield = ion.NoElectricField()

        ani = [
            ion.animators.LineAnimator(target_dir = OUT_DIR, postfix = '_full'),
            ion.animators.LineAnimator(target_dir = OUT_DIR, plot_limit = 10 * nm, postfix = '_10'),
        ]

        sim = ion.LineSpecification('spectral',
                                    x_bound = 100 * nm, x_points = 2 ** 14,
                                    internal_potential = qho,
                                    initial_state = initial_state,
                                    test_states = test_states,
                                    test_mass = mass,
                                    electric_potential = efield,
                                    time_initial = 0,
                                    time_final = efield.period * 10,
                                    time_step = 10 * asec,
                                    animators = ani,
                                    ).to_simulation()

        sim.mesh.plot_g(target_dir = OUT_DIR)

        sim.run_simulation()
        logger.info(sim.info())

        sim.plot_test_state_overlaps_vs_time(target_dir = OUT_DIR,
                                             grouped_free_states = {})
