import logging
import os

import numpy as np
import scipy.sparse as sparse

import simulacra as si
import ionization as ion
from simulacra.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('compy', 'ionization', stdout_level = logging.INFO) as logger:
        mass = electron_mass
        pot = ion.HarmonicOscillator.from_energy_spacing_and_mass(energy_spacing = 1 * eV, mass = mass)

        # init = ion.Superposition({ion.QHOState(omega = pot.omega(mass), mass = mass, n = 0): 1,
        #                           ion.QHOState(omega = pot.omega(mass), mass = mass, n = 1): 1})
        init = ion.QHOState.from_potential(pot, mass, n = 20)

        electric = ion.SineWave.from_photon_energy(1 * eV, amplitude = .01 * atomic_electric_field)

        ani = [
            ion.animators.LineAnimator(target_dir = OUT_DIR, postfix = '_full'),
            ion.animators.LineAnimator(target_dir = OUT_DIR, postfix = '_full_log', log_metrics = True),
            ion.animators.LineAnimator(target_dir = OUT_DIR, plot_limit = 10 * nm, postfix = '_10'),
        ]

        sim = ion.LineSpecification('qho',
                                    x_bound = 50 * nm, x_points = 2 ** 14,
                                    internal_potential = pot,
                                    # internal_potential = ion.NoPotentialEnergy(),
                                    electric_potential = electric,
                                    test_mass = mass,
                                    test_states = (ion.QHOState.from_potential(pot, mass, n = n) for n in range(60)),
                                    dipole_gauges = (),
                                    initial_state = init,
                                    time_initial = 0, time_final = 10 * fsec, time_step = 10 * asec,
                                    animators = ani
                                    ).to_simulation()

        # delta = 2 * bohr_radius
        # sim.mesh.psi = np.exp(-0.5 * ((sim.mesh.x / delta) ** 2)).astype(np.complex128)

        print('init norm', sim.mesh.norm())

        sim.mesh.plot_g(name_postfix = '_init', target_dir = OUT_DIR)

        sim.run_simulation()

        print(sim.info())
        print('norm', sim.mesh.norm())
        print('energy EV', sim.energy_expectation_value_vs_time_internal / eV)
        # print(pot.omega(init.mass), init.omega)
        # print('period: {} fs'.format(uround(init.period, fsec, 3)))

        sim.mesh.plot_g(name_postfix = '_post', target_dir = OUT_DIR)
        # sim.plot_wavefunction_vs_time(target_dir = OUT_DIR, x_unit = 'fsec')
        sim.plot_test_state_overlaps_vs_time(target_dir = OUT_DIR, x_unit = 'fsec')

        # print(sim.mesh.wavenumbers)
        # print(sim.mesh.free_evolution_prefactor)
